import tensorflow as tf 
from tensorflow.keras.applications import EfficientNetB7 #Requires latest build of nighly tensorflow

from transformer import Transformer


class detr(tf.keras.models.Model):
    def __init__(self, num_classes, hidden_dim, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):

        super(detr, self).__init__()

        #Create EfficientNet Backbone
        self.backbone = EfficientNetB7(include_top=False)

        # conversion layer
        self.conv = tf.keras.layers.Conv2D(2048, hidden_dim, 1)

        # transformer
        self.transformer = Transformer()

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = tf.keras.layers.Dense(num_classes+1,input_dim=hidden_dim)
        self.linear_bbox = tf.keras.layers.Dense(4, input_dim=hidden_dim)

        # output positional encodings (object queries)
        self.query_pos = tf.random.uniform(100, hidden_dim)

        # spatial position encodings
        self.row_embed = tf.random.uniform(50, hidden_dim // 2)
        self.col_embed = tf.random.uniform(50, hidden_dim // 2)

    def call(self, inputs):
        # progagate inputs thru EfficientNet up to avg-pool layer
        x = self.backbone(inputs)

        #convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = 