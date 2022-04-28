import numpy as np
import os
import codecs
import logging
import jieba
import MeCab
from tempfile import mkstemp
from nltk.translate.bleu_score import *
from utils import DataUtil, AttrDict
import yaml
import tensorflow as tf


def create_dataset(dirname, filename):
    with open(os.path.join(dirname, filename), 'r') as fin:
        sentences = fin.readlines()
    fin.close()
    titles, ja_sentences, zh_sentences = [], [], []

    for line in sentences:
        parallel = line.split("|||")
        title = "-".join(parallel[0].split("-")[:-1])
        titles.append(title.strip())
        ja_sentences.append(parallel[1].strip())
        zh_sentences.append(parallel[2].strip())

    jout = open(os.path.join(dirname, "ja_doc.txt"), "w")
    zout = open(os.path.join(dirname, "zh_doc.txt"), "w")
    previous_title = titles[0]
    ja_doc, zh_doc = [], []
    ja_par, zh_par = [], []

    for title, ja, zh in zip(titles, ja_sentences, zh_sentences):
        if title != previous_title:
            ja_doc.append(ja_par)
            zh_doc.append(zh_par)
            ja_par, zh_par = [], []
        print(title, " ja:", ja, " zh:", zh)
        ja_par.append(ja)
        zh_par.append(zh)
        previous_title = title


    for ja_par, zh_par in zip(ja_doc, zh_doc):
        if len(ja_par) > 2:
            for length in range(2, len(ja_par)):
                jout.write("||".join(ja_par[length - 2:length+1])+"\n")
                zout.write("||".join(zh_par[length - 2:length + 1])+"\n")
    '''
    for title, ja, zh in zip(titles, ja_sentences, zh_sentences):
        print(title, " ja:", ja, " zh:", zh)
        if title!=previous_title:
            jout.write('\n')
            zout.write('\n')
        jout.write(ja+'\n')
        zout.write(zh+'\n')
        previous_title = title
    '''
    jout.close()
    zout.close()

def attention_bias_lower_traingle_l(length, l):
    lower_triangle = tf.zeros([length, length])
    def recurrency(i, lower_triangle, l):
        new_block = tf.concat([tf.ones([l, i+l]), tf.zeros([l, length-(i+l)])], 1)
        lower_triangle = tf.concat([lower_triangle[:i,:], new_block, lower_triangle[i+l:,:]], 0)
        return i+l, lower_triangle, l

    initial_i = 0
    _, lower_triangle, _ = tf.while_loop(
        cond=lambda a, _1, _2: a < length,
        body=recurrency,
        loop_vars=(initial_i, lower_triangle, l),
    )

    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])

if __name__ == "__main__":
    #make_vocab(hp.source_train, "de.vocab.tsv")
    #make_vocab(hp.target_train, "en.vocab.tsv")
    #make_ja_vocab(hp.ja_train, "ja.vocab.tsv")
    #make_zh_vocab(hp.zh_train, "zh.vocab.tsv")

    create_dataset("./ASPEC-JC/dev", "dev.txt")
    print("Done!")
    '''
    x = tf.placeholder(tf.int32, None)
    mask = attention_bias_lower_traingle_l(x, 3)
    with tf.Session() as sess:
        c = tf.constant(30, tf.int32)
        a = sess.run(mask, feed_dict={x: 30})
        print(a)
    
    tower_grads = [[(1,2),(2,3),(3,4)], [(4,8),(5,10),(6,12)]]
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = np.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = np.concatenate(grads, axis=0)
            grad = np.mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    print(average_grads)
    
    config = AttrDict(yaml.load(open("./configs/config_generator_train.yaml")))
    du = DataUtil(config=config)
    du.load_vocab()
    count = 0
    while True:
        for batch in du.get_training_batches_doc():
            count += 1
            print(len(batch[0]))
        print(count)
        count = 0
    '''

    """
    c = [[1,5,6,7,8], [3,4,5]]

    for length in range(1, 10):
        a = []
        b = []
        for p in c:
            if len(p) > length:
                a.append(p[:length])
                b.append(p[length])
        print(a, b)
    """

