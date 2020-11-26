from positional_encoding import create_pe
from attention_mechs import MultiHeadAttention, point_wise_feed_forward_network
import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dp_rate=0.1):

        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dp_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attn_output1, attn_weights1 = self.mha(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_output1 = self.dropout(attn_output1, training=training)
        out1 = self.layernorm(attn_output1 + x)  # (batch_size, target_seq_len, d_model)

        attn_output2, attn_weights2 = self.mha(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn_output2 = self.dropout(attn_output2, training=training)
        out2 = self.layernorm(attn_output2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout(ffn_output, training=training)
        out3 = self.layernorm(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights1, attn_weights2


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, dp_rate=0.1):

        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.d_model = d_model

        self.pos_encoding = create_pe(self.d_model, maximum_position_encoding)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dp_rate) for i in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dp_rate)

    def call(self, y, enc_output, training, look_ahead_mask, padding_mask):

        attention_weights = {}

        y = tf.one_hot(y, self.d_model)
        y = tf.cast(y, tf.float32) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        y += self.pos_encoding[:, :tf.shape(y)[1], :]

        y = self.dropout(y, training=training)

        for i in range(self.num_layers):
            y, block1, block2 = self.dec_layers[i](y, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return y, attention_weights
