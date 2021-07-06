import tensorflow as tf
from transformers import TFBertModel
from transformers.modeling_tf_utils import get_initializer


class MyModel(tf.keras.Model):

    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-chinese')
        self.dropout = tf.keras.layers.Dropout(.1)
        self.classifier = tf.keras.layers.Dense(units=num_labels,
                                                activation=tf.nn.sigmoid,
                                                kernel_initializer=get_initializer(0.02))

    def call(self, inputs, training=None, mask=None):
        x = self.bert(inputs)[0]
        if training:
            x = self.dropout(x, training=training)
        return self.classifier(x)
