set -eux

export BATCH_SIZE=32
export LR=2e-5
export EPOCH=3

unset CUDA_VISIBLE_DEVICES
python run_duie.py \
  --device gpu \
  --seed 42 \
  --do_train \
  --data_path ../data \
  --max_seq_length 128 \
  --batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCH \
  --learning_rate $LR \
  --warmup_ratio 0.06 \
  --output_dir ./checkpoints
