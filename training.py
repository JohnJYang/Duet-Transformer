from positional_encoding import read_quanta_docs, add_padding, one_hot_data, create_pe
from attention_mechs import create_padding_mask, create_look_ahead_mask
from transformer import Transformer
import tensorflow as tf
import numpy as np
import time

# tf.random.set_seed(7)

num_layers = 6
d_model = 510
num_heads = 6
dff = 2048
dropout_rate = 0.1


def temp_add_one(data):
    # temporary, use until quanta files are re-created
    new = []
    for quanta in data:
        quanta = np.asarray(quanta)
        quanta += 1
        new.append(quanta.tolist())
    return new


def acquire(file, batch, batch_size):
    if file == 1:
        return add_padding(temp_add_one(read_quanta_docs('data/quanta_docs/0')[batch:batch+batch_size]), 7308)
    else:
        return add_padding(temp_add_one(read_quanta_docs('data/quanta_docs/1')[batch:batch+batch_size]), 5416)


# solos = tf.cast(one_hot_data(add_padding(temp_add_one(read_quanta_docs('data/quanta_docs/0')), 7308), d_model), dtype=tf.float32)
# duets = tf.cast(one_hot_data(add_padding(temp_add_one(read_quanta_docs('data/quanta_docs/1')), 5416), d_model), dtype=tf.float32)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):

        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


def create_masks(x, y):

    enc_padding_mask = create_padding_mask(x)  # use in encoder
    dec_padding_mask = create_padding_mask(x)  # use in 2 attn block of decoder to mask enc_outputs

    dec_target_padding_mask = create_padding_mask(y)
    look_ahead_mask = create_look_ahead_mask(tf.shape(y)[1])  # use in 1 attn block of decoder
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, dec_padding_mask, combined_mask


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer = Transformer(num_layers, d_model, num_heads, dff, pe_x=7310, pe_y=5420, dp_rate=dropout_rate)

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


@tf.function(input_signature=train_step_signature)
def train_step(x, y):

    y_inp = y[:, :-1]
    y_real = y[:, 1:]

    enc_padding_mask, dec_padding_mask, combined_mask = create_masks(x, y)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(x, y_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(y_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(y_real, predictions)


def optimize(epoch, batch_size):

    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for e in range(epoch):
        for batch in range(0, 968, batch_size):

            train_step(acquire(1, batch, batch_size), acquire(2, batch, batch_size))

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


optimize(1, 22)
