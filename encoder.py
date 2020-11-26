from positional_encoding import create_pe
from attention_mechs import MultiHeadAttention, point_wise_feed_forward_network
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dp_rate=0.1):

        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dp_rate)

    def call(self, x, training, mask):

        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, dp_rate=0.1):

        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.d_model = d_model

        self.pos_encoding = create_pe(self.d_model, maximum_position_encoding)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dp_rate) for i in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dp_rate)

    def call(self, x, training, mask):

        x = tf.one_hot(x, int(self.d_model))
        x = tf.cast(x, tf.float32) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
