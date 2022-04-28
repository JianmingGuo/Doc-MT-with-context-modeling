import numpy as np
import os
import codecs
import logging
import jieba
import MeCab
from tempfile import mkstemp
from nltk.translate.bleu_score import *

class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class DataUtil():
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('util')

    def load_vocab(self, src_vocab=None, dst_vocab=None, src_vocab_size=None, dst_vocab_size=None):
        """
        Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """
        def vocab(fpath, vocab_size):
            print("vocab_size",vocab_size )
            vocab = [line.split()[0] for line in codecs.open(fpath, "r", "utf-8").read().splitlines()]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        if src_vocab and dst_vocab and src_vocab_size and dst_vocab_size:
            self.logger.debug('Load set vocabularies as %s and %s.' % (src_vocab, dst_vocab))
            self.src2idx, self.idx2src = vocab(src_vocab, src_vocab_size)
            self.dst2idx, self.idx2dst = vocab(dst_vocab, dst_vocab_size)
        else:
            self.logger.debug('Load vocabularies %s and %s.' % (self.config.src_vocab, self.config.dst_vocab))
            self.src2idx, self.idx2src = vocab(self.config.src_vocab, self.config.src_vocab_size)
            self.dst2idx, self.idx2dst = vocab(self.config.dst_vocab, self.config.dst_vocab_size)

    def get_training_batches(self, shuffle=True, set_train_src_path=None, set_train_dst_path=None,
                             set_batch_size=None, set_max_length=None):
        if set_train_src_path and set_train_dst_path:
            src_path = set_train_src_path
            dst_path = set_train_dst_path
        else:
            src_path = self.config.train.src_path
            dst_path = self.config.train.dst_path

        if set_batch_size:
            batch_size = set_batch_size
        else:
            batch_size = self.config.train.batch_size

        if set_max_length:
            max_length = set_max_length
        else:
            max_length = self.config.train.max_length
        while True:
            if shuffle:
                src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
            else:
                src_shuf_path = src_path
                dst_shuf_path = dst_path

            sources, targets = [], []
            source_sents = [line for line in codecs.open(src_shuf_path, "r", "utf-8").read().split("\n") if
                            line and line[0] != "<"]
            target_sents = [line for line in codecs.open(dst_shuf_path, "r", "utf-8").read().split("\n") if
                            line and line[0] != "<"]

            for source_sent, target_sent in zip(source_sents, target_sents):
                x = [word for word in source_sent.split()]
                y = [word for word in target_sent.split()]
                if len(x) <= max_length and len(y) <= max_length:
                    sources.append(x)
                    targets.append(y)
                if len(sources) >= batch_size:
                    yield self.create_batch(sources, o='src'), self.create_batch(targets, o='dst')
                    sources, targets = [], []

        #if sources and targets:
        #    yield self.create_batch(sources, o="src"), self.create_batch(targets, o="dst")

            if shuffle:
                os.remove(src_shuf_path)
                os.remove(dst_shuf_path)
                print("shuffle again")
                break

    def get_training_batches_with_buckets(self, shuffle=True):
        """
        Generate batches according to bucket setting.
        """

        buckets = [(i, i) for i in range(10, 100, 5)] + [(self.config.train.max_length, self.config.train.max_length)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return (l1, l2)
            return None

        # Shuffle the training files.
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        if shuffle:
            self.logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

        source_sents = [line for line in codecs.open(src_shuf_path, "r", "utf-8").read().split("\n") if
                        line and line[0] != "<"]
        target_sents = [line for line in codecs.open(dst_shuf_path, "r", "utf-8").read().split("\n") if
                        line and line[0] != "<"]

        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [word for word in source_sent.split()]
            y = [word for word in target_sent.split()]

            bucket = select_bucket(len(x), len(y))
            if bucket is None:  # No bucket is selected when the sentence length exceed the max length.
                continue

            caches[bucket][0].append(x)
            caches[bucket][1].append(y)
            caches[bucket][2] += len(x)
            caches[bucket][3] += len(y)

            if max(caches[bucket][2], caches[bucket][3]) >= self.config.train.tokens_per_batch:
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
                self.logger.debug(
                    'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                yield batch
                caches[bucket] = [[], [], 0, 0]

        # Clean remain sentences.
        for bucket in buckets:
            # Ensure each device at least get one sample.
            if len(caches[bucket][0]) > len(self.config.train.devices.split(',')):
                batch = (
                self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
            self.logger.debug(
                'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
            yield batch

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    def get_training_batches_with_buckets_doc(self, shuffle=False):
        """
        Generate batches according to bucket setting.
        """

        buckets = [(i, i) for i in range(10, 100, 5)] + [(self.config.train.max_length, self.config.train.max_length)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return (l1, l2)
            return None

        # Shuffle the training files.
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        if shuffle:
            self.logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0, [], []]  # src sentences, dst sentences, src tokens, dst tokens

        source_sents = [line.split("||")[-1] for line in (codecs.open(src_shuf_path, "r", "utf-8").read().split("\n")) if
                        line and line[0] != "<"]
        target_sents = [line.split("||")[-1] for line in (codecs.open(dst_shuf_path, "r", "utf-8").read().split("\n")) if
                        line and line[0] != "<"]

        source_contexts = [line.split("||")[:-1] for line in (codecs.open(src_shuf_path, "r", "utf-8").read().split("\n")) if
                        line and line[0] != "<"]
        target_contexts = [line.split("||")[:-1] for line in (codecs.open(dst_shuf_path, "r", "utf-8").read().split("\n")) if
                        line and line[0] != "<"]

        for source_sent, target_sent, source_context, target_context in zip(source_sents, target_sents, source_contexts, target_contexts):
            x = [word for word in source_sent.split()]
            y = [word for word in target_sent.split()]
            x_c = [[word for word in source_c_sent.split()] for source_c_sent in source_context]
            y_c = [[word for word in target_c_sent.split()] for target_c_sent in target_context]

            bucket = select_bucket(len(x), len(y))
            if bucket is None:  # No bucket is selected when the sentence length exceed the max length.
                continue

            caches[bucket][0].append(x)
            caches[bucket][1].append(y)
            caches[bucket][2] += len(x)
            caches[bucket][3] += len(y)
            caches[bucket][4].append(x_c)
            caches[bucket][5].append(y_c)

            if max(caches[bucket][2], caches[bucket][3]) >= self.config.train.tokens_per_batch:
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'),
                         self.create_batch_context(caches[bucket][4], o='src'), self.create_batch_context(caches[bucket][5], o='dst'))
                self.logger.debug(
                    'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                yield batch
                caches[bucket] = [[], [], 0, 0, [], []]

        # Clean remain sentences.
        for bucket in buckets:
            # Ensure each device at least get one sample.
            if len(caches[bucket][0]) > len(self.config.train.devices.split(',')):
                batch = (
                self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'),
                self.create_batch_context(caches[bucket][4], o='src'), self.create_batch_context(caches[bucket][5], o='dst'))
            self.logger.debug(
                'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
            yield batch

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    def get_training_batches_doc(self, shuffle=False):
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        batch_size = self.config.train.batch_size
        if shuffle:
            self.logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        ss = open(src_shuf_path, "r")
        tt = open(dst_shuf_path, "r")
        source_contexts, target_contexts = [], []
        try:
            while True:
                source = ss.readline()
                target = tt.readline()
                if not source or not target:
                    break
                source_context = [[word for word in source_c_sent.split()] for source_c_sent in source.split("||")]
                target_context = [[word for word in target_c_sent.split()] for target_c_sent in target.split("||")]

                source_contexts.append(source_context)
                target_contexts.append(target_context)
                if len(source_contexts) >= batch_size:
                    yield self.create_batch_context(source_contexts, o='src'), self.create_batch_context(target_contexts, o='dst')
                    source_contexts, target_contexts = [], []

            if source_contexts and target_contexts:
                yield self.create_batch_context(source_contexts, o='src'), self.create_batch_context(target_contexts, o='dst')
        except IOError:
            pass

## for contra pro
    def get_training_batches_doc_single(self, shuffle=False):
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        batch_size = self.config.train.batch_size
        if shuffle:
            self.logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        ss = open(src_shuf_path, "r")
        tt = open(dst_shuf_path, "r")
        source_contexts, target_contexts = [], []
        try:
            while True:
                source = ss.readline()
                target = tt.readline()
                if not source or not target:
                    break
                source_context = [[word for word in source_c_sent.split()] for source_c_sent in source.split("||")]
                target_context = [[word for word in target_c_sent.split()] for target_c_sent in target.split("||")]

                source_contexts.append(source_context)
                target_contexts.append(target_context)
                if len(source_contexts) >= batch_size:
                    yield self.create_batch_context_test(source_contexts, o='src'), self.create_batch_context_test(target_contexts, o='dst')
                    source_contexts, target_contexts = [], []

            if source_contexts and target_contexts:
                yield self.create_batch_context_test(source_contexts, o='src'), self.create_batch_context_test(target_contexts, o='dst')
        except IOError:
            pass

    @staticmethod
    def shuffle(list_of_files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')
        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("|||||".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fds = [open(ff + '.{}.shuf'.format(os.getpid()), 'w') for ff in list_of_files]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('|||||')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return [ff + '.{}.shuf'.format(os.getpid()) for ff in list_of_files]

    def get_test_batches_doc(self, set_src_path=None, set_batch=None):
        if set_src_path and set_batch:
            src_path = set_src_path
            batch_size = set_batch
        else:
            src_path = self.config.test.src_path
            batch_size = self.config.test.batch_size

        ss = open(src_path, "r")
        source_contexts = []
        try:
            while True:
                source = ss.readline()
                if not source:
                    break
                source_context = [[word for word in source_c_sent.split()] for source_c_sent in
                                  source.split("||")]

                source_contexts.append(source_context)
                if len(source_contexts) >= batch_size:
                    yield self.create_batch_context_test(source_contexts, o='src')
                    source_contexts = []

            if source_contexts:
                yield self.create_batch_context(source_contexts, o='src')
        except IOError:
            pass

    def get_test_batches_with_target(self,
                                     set_test_src_path=None,
                                     set_test_dst_path=None,
                                     set_batch_size=None):

        if set_test_src_path and set_test_dst_path and set_batch_size:
            src_path = set_test_src_path
            dst_path = set_test_dst_path
            batch_size = set_batch_size

        else:
            src_path = self.config.test.src_path
            dst_path = self.config.test.dst_path
            batch_size = self.config.test.batch_size


        ss = open(src_path, "r")
        tt = open(dst_path, "r")

        source_contexts = []
        target_contexts = []
        try:
            while True:
                source = ss.readline()
                target = tt.readline()
                if not source:
                    break
                if not target:
                    break

                source_context = [[word for word in source_c_sent.split()] for source_c_sent in
                                  source.split("||")]
                target_context = [[word for word in target_c_sent.split()] for target_c_sent in
                                  target.split("||")]

                source_contexts.append(source_context)
                target_contexts.append(target_context)

                if len(source_contexts) >= batch_size and len(target_contexts) >= batch_size:
                    # print(len(source_contexts), len(target_contexts),  "begin yield")
                    yield self.create_batch_context_test(source_contexts, o='src'), self.create_batch_context_test(target_contexts, o='dst')
                    source_contexts = []
                    target_contexts = [] 
    

            if source_contexts:
                yield self.create_batch_context_test(source_contexts, o='src'), self.create_batch_context_test(target_contexts, o='dst')
        except IOError:
            pass

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def create_batch_context(self, contexts, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        p_indices = []
        max_indices = 0
        for sents in contexts:
            indices = []
            for sent in sents:
                x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
                indices.append(x)
            max_indices = len(indices) if len(indices)> max_indices else max_indices
            # print(len(indices))
            p_indices.append(indices)

        # Pad to the same length.
        #maxlen = max(sum([[len(s) for s in indices] for indices in p_indices], []))
        max_length = self.config.train.max_length
        X = np.zeros([len(p_indices), 10, max_length], np.int32)
        for i, indices in enumerate(p_indices):
            if  len(indices)<=10:
                for j, x in enumerate(indices):
                    if len(x)<max_length :
                        # print("##################")
                        # print(X.shape)
                        # print(i,j,x)
                        X[i, j, :len(x)] = x
                    elif len(x)>=max_length:
                        X[i, j, :max_length] = x[:max_length]
            else:
                for j in range(10):
                    if len(indices[j])<max_length :
                        # print("##################")
                        # print(X.shape)
                        # print(i,j,indices[j])
                        X[i, j, :len(indices[j])] = indices[j]
                    elif len(indices[j])>=max_length:
                        X[i, j, :max_length] = indices[j][:max_length]
                
        return X

    def create_batch_context_test(self, contexts, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        p_indices = []
        max_indices = 0
        for sents in contexts:
            indices = []
            for sent in sents:
                x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
                indices.append(x)
            max_indices = len(indices) if len(indices)> max_indices else max_indices
            # print(len(indices))
            p_indices.append(indices)

        # Pad to the same length.
        #maxlen = max(sum([[len(s) for s in indices] for indices in p_indices], []))
        max_length = self.config.train.max_length
        X = np.zeros([len(p_indices), len(p_indices[0]), max_length], np.int32)
        # print(len(p_indices[0]))
        for i, indices in enumerate(p_indices):
            for j, x in enumerate(indices):
                if len(x)<max_length :
                    X[i, j, :len(x)] = x
                elif len(x)>=max_length:
                    X[i, j, :max_length] = x[:max_length]

        return X

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y:  # for each sentence
            sent = []
            for i in y:  # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents

    def indices_to_words_del_pad(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        pad_index = idx2word
        sents = []
        for y in Y:
            sent = []
            for i in y:
                if i > 0:
                    w = idx2word[i]
                    sent.append(w)
            # print(sent)
            sents.append(' '.join(sent))
        return sents

def calc_bleu_2(ref, translation):
    with open(ref, 'r', encoding='utf-8') as fileread:
        tgt_lines = fileread.readlines()

    with open(translation, 'r', encoding='utf-8') as fileread:
        pred_lines = fileread.readlines()

    length = len(tgt_lines)
    assert length == len(pred_lines)
    smoother = SmoothingFunction()

    score = 0
    for i in range(length):
        candidate = pred_lines[i]
        # reference_list.append(candidate)
        # print(reference_list)
        # print(candidate)
        temp = sentence_bleu([tgt_lines[i].split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method7)
        #print(temp)
        #print(tgt_lines, candidate)
        score += temp

    final = float(score) / length
    print(final)

