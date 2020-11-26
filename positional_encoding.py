import glob
import math
import tensorflow as tf
import matplotlib.pyplot as plt


def read_quanta_docs(file_path):
    # solos longest = 7308
    # duets longest = 5416
    quanta_list = []

    for docs in sorted(glob.glob(file_path + '/*.txt')):
        file = open(docs)
        all_quanta = file.readlines()
        file.close()

        for quanta in all_quanta:
            put = []

            for q in quanta[:-2].split(','):
                put.append(int(q) + 1)

            quanta_list.append(put)

    return quanta_list


def add_padding(data, length):

    padded = []

    for quanta in data:
        for i in range(length - len(quanta)):
            quanta.append(0)
        padded.append(quanta)

    return padded


def one_hot_data(data, dim):

    one_hotted = []

    for quanta_list in data:
        one_hotted.append(tf.one_hot(quanta_list, dim))

    return one_hotted


def create_pe(d_model, max_length):

    pe = [[0 for x in range(d_model)] for y in range(max_length)]

    for pos in range(max_length):
        for i in range(0, d_model, 2):
            div_term = math.exp(i * -(math.log(10000) / d_model))
            pe[pos][i] = math.sin(pos * div_term)
            pe[pos][i + 1] = math.cos(pos * div_term)

    return tf.cast([pe], dtype=tf.float32)


def add_pe(data, pe):

    new = []

    for quanta in data:
        new.append(quanta + pe[len(quanta[0])])

    return new


def create_heatmap(data):
    plt.pcolormesh(data, cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
