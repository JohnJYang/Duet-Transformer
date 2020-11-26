from encoder import Encoder
from decoder import Decoder
import tensorflow as tf


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, pe_x, pe_y, dp_rate=0.1):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_x, dp_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_y, dp_rate)

        self.final_layer = tf.keras.layers.Dense(d_model)

    def call(self, x, y, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(x, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(y, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_model)

        return final_output, attention_weights
