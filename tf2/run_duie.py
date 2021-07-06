import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertTokenizer
from mymodel import MyModel
from warmup import LinearDecayWithWarmup

from data_loader import DuIEDataset, DataCollator
from utils import decoding, find_entity, get_precision_recall_f1, write_prediction_results

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
tf.compat.v1.set_random_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--predict_data_file", default="./data/test_data.json", type=str, required=False,
                    help="Path to data.")
parser.add_argument("--output_dir", default="./checkpoints", type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


# yapf: enable


class BCELossForDuIE(tf.keras.layers.Layer):
    def __init__(self):
        super(BCELossForDuIE, self).__init__()
        self.criterion = tf.keras.losses.BinaryCrossentropy(reduction='none')

    def call(self, inputs, **kwargs):
        logits, labels, mask = inputs
        loss = self.criterion(logits, labels)  # 2D
        mask = tf.cast(mask, 'float32')
        loss = loss * mask
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(mask, axis=1)
        return loss


# def set_random_seed(seed):
#     """sets random seed"""
#     random.seed(seed)
#     np.random.seed(seed)
#     paddle.seed(seed)


@tf.function
def evaluate(model, criterion, data_loader, test_loss, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    probs_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    for batch in tqdm(data_loader):
        eval_steps += 1
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        loss = criterion((logits, labels, mask))
        loss_all += test_loss(loss).result()
        probs = logits
        if probs_all is None:
            probs_all = probs.numpy()
            seq_len_all = seq_len.numpy()
            tok_to_orig_start_index_all = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_all = tok_to_orig_end_index.numpy()
        else:
            probs_all = np.append(probs_all, probs.numpy(), axis=0)
            seq_len_all = np.append(seq_len_all, seq_len.numpy(), axis=0)
            tok_to_orig_start_index_all = np.append(
                tok_to_orig_start_index_all,
                tok_to_orig_start_index.numpy(),
                axis=0)
            tok_to_orig_end_index_all = np.append(
                tok_to_orig_end_index_all,
                tok_to_orig_end_index.numpy(),
                axis=0)
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)
    formatted_outputs = decoding(file_path, id2spo, probs_all, seq_len_all,
                                 tok_to_orig_start_index_all,
                                 tok_to_orig_end_index_all)
    if mode == "predict":
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")


def do_train():
    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    train_num = len(open(os.path.join(args.data_path, 'train_data.json')).readlines())

    # Loads pretrained model ERNIE
    model = MyModel(num_classes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    criterion = BCELossForDuIE()

    # Loads dataset.
    train_generator = DuIEDataset.from_file(
        os.path.join(args.data_path, 'train_data.json'), tokenizer,
        args.max_seq_length, True)
    train_dataset = tf.data.Dataset.from_generator(train_generator,
                                                   (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                                   (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None]),
                                                    tf.TensorShape([None]), tf.TensorShape([None, 1]))).shuffle(
        10000).batch(args.batch_size)

    eval_file_path = os.path.join(args.data_path, 'dev_data.json')
    test_generator = DuIEDataset.from_file(eval_file_path, tokenizer,
                                           args.max_seq_length, True)
    test_dataset = tf.data.Dataset.from_generator(test_generator,
                                                  (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                                  (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None]),
                                                   tf.TensorShape([None]), tf.TensorShape([None, 1]))).batch(
        args.batch_size)

    # Defines learning rate strategy.
    steps_by_epoch = math.ceil(train_num / args.batch_size)
    num_training_steps = steps_by_epoch * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_ratio)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        tvar for tvar in tf.compat.v1.trainable_variables() if not any(nd in tvar.name for nd in ["bias", "Norm"])
    ]
    optimizer = tfa.optimizers.AdamW(weight_decay=args.weight_decay, learning_rate=lr_scheduler)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(batch):
        input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        with tf.GradientTape() as tape:
            logits = model(input_ids=input_ids)
            mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
            loss = criterion((logits, labels, mask))
        gradients = tape.gradient(loss, decay_params)
        optimizer.apply_gradients(zip(gradients, decay_params))

        train_loss(loss)

    # Starts training.
    global_step = 0
    logging_steps = 50
    save_steps = 10000
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        train_loss.reset_states()
        for step, batch in enumerate(train_dataset):
            train_step(batch)
            loss_item = train_loss.result()
            global_step += 1

            if global_step % logging_steps == 0:
                print("epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                      % (epoch, args.num_train_epochs, step, steps_by_epoch, loss_item,
                         logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % save_steps == 0:
                print("\n=====start evaluating ckpt of %d steps=====" % global_step)
                test_loss.reset_states()
                precision, recall, f1 = evaluate(model, criterion, test_dataset, test_loss, eval_file_path, "eval")
                print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" % (100 * precision, 100 * recall, 100 * f1))
                print("saving checkpoing model_%d.pdparams to %s " % (global_step, args.output_dir))
                model.save(os.path.join(args.output_dir, "model_%d.pdparams.h5" % global_step))

        tic_epoch = time.time() - tic_epoch
        print("epoch time footprint: %d hour %d min %d sec" % (
            tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

    # Does final evaluation.
    print("\n=====start evaluating last ckpt of %d steps=====" % global_step)
    test_loss.reset_states()
    precision, recall, f1 = evaluate(model, criterion, test_dataset, test_loss, eval_file_path, "eval")
    print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" % (100 * precision, 100 * recall, 100 * f1))
    model.save(os.path.join(args.output_dir, "model_%d.pdparams.h5" % global_step))
    print("\n=====training complete=====")


def do_predict():
    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    # model = MyModel(num_classes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    criterion = BCELossForDuIE()

    # Loads dataset.
    test_generator = DuIEDataset.from_file(os.path.join(args.data_path, 'test_data.json'), tokenizer,
                                           args.max_seq_length, True)
    test_dataset = tf.data.Dataset.from_generator(test_generator,
                                                  (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                                  (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None]),
                                                   tf.TensorShape([None]), tf.TensorShape([None, 1]))).batch(
        args.batch_size)

    # Loads model parameters.
    if not (os.path.exists(args.init_checkpoint) and os.path.isfile(args.init_checkpoint)):
        sys.exit("wrong directory: init checkpoints {} not exist".format(args.init_checkpoint))
    model = tf.keras.models.load_model(sorted(os.listdir(args.init_checkpoint))[-1])
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Does predictions.
    print("\n=====start predicting=====")
    test_loss.reset_states()
    evaluate(model, criterion, test_dataset, test_loss, args.predict_data_file, "predict")
    print("=====predicting complete=====")


if __name__ == "__main__":
    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()
