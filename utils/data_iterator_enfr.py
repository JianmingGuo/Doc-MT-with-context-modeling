import codecs
import gzip
import numpy
import jieba
import MeCab
import nltk

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class disThreeTextIterator:
    def __init__(self, positive_data, negative_data, source_data, dic_target, dic_source, batch=1, maxlen=50, n_words_target=-1, n_words_source=-1):
        self.positive = fopen(positive_data, 'r')
        self.negative = fopen(negative_data, 'r')
        self.source = fopen(source_data, 'r')

        vocab_source = [line.split()[0] for line in codecs.open(dic_source, "r", "utf-8").read().splitlines()]
        self.dic_source = {word: idx for idx, word in enumerate(vocab_source)}
        vocab_target = [line.split()[0] for line in codecs.open(dic_target, "r", "utf-8").read().splitlines()]
        self.dic_target = {word: idx for idx, word in enumerate(vocab_target)}


        self.batch_size = batch
        assert self.batch_size % 2 == 0
        self.maxlen = maxlen
        self.n_words_trg = n_words_target
        self.n_words_src = n_words_source
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.positive.seek(0)
        self.negative.seek(0)
        self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        positive = []
        negative = []
        source = []
        x = []
        xs = []
        y = []

        try:
            while True:
                ss = self.positive.readline()
                if ss == "":
                    raise IOError
                # ss = (" ".join(jieba.cut(ss.strip(), False))).split()
                ss = nltk.word_tokenize(ss)
                ss = [self.dic_target[w] if w in self.dic_target else 1 for w in ss]
                if self.n_words_trg > 0:
                    ss = [w if w < self.n_words_trg else 1 for w in ss]

                tt = self.negative.readline()
                if tt == "":
                    raise IOError
                tt = nltk.word_tokenize(tt)
                tt = [self.dic_target[w] if w in self.dic_target else 1 for w in tt]
                if self.n_words_trg > 0:
                    tt = [w if w < self.n_words_trg else 1 for w in tt]

                ll = self.source.readline()
                if ll == "":
                    raise IOError
                ll = nltk.word_tokenize(ll)
                ll = [self.dic_source[w] if w in self.dic_source else 1 for w in ll]
                if self.n_words_src > 0:
                    ll = [w if w < self.n_words_src else 1 for w in ll]

                if len(ss) > self.maxlen or len(tt) > self.maxlen or len(ll) > self.maxlen:
                    continue

                positive.append(ss)
                negative.append(tt)
                source.append(ll)

                x = positive + negative

                # positive_labels = [[0, 1] for _ in positive]
                # negative_labels = [[1, 0] for _ in negative]
                positive_labels = [[0, 1] for _ in positive]
                negative_labels = [[1, 0] for _ in negative]
                y = positive_labels + negative_labels

                xs = source + source

                shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                x_np = numpy.array(x,dtype=object)
                y_np = numpy.array(y,dtype=object)
                xs_np = numpy.array(xs,dtype=object)

                x_np_shuffled = x_np[shuffle_indices]
                y_np_shuffled = y_np[shuffle_indices]
                xs_np_shuffled = xs_np[shuffle_indices]

                x_shuffled = x_np_shuffled.tolist()
                y_shuffled = y_np_shuffled.tolist()
                xs_shuffled = xs_np_shuffled.tolist()

                if len(x_shuffled) >= self.batch_size and len(y_shuffled) >= self.batch_size and len(
                        xs_shuffled) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(positive)<=0 or len(negative)<=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return x_shuffled, y_shuffled, xs_shuffled

class disTextIterator:
    def __init__(self, positive_data, negative_data, dis_dict, batch=1, maxlen=30, n_words_target=-1):
        self.positive = fopen(positive_data, 'r')
        self.negative = fopen(negative_data, 'r')
        vocab_dis = [line.split()[0] for line in codecs.open(dis_dict, "r", "utf-8").read().splitlines()]
        self.dis_dict = {idx: word for idx, word in enumerate(vocab_dis)}

        self.batch_size = batch
        assert self.batch_size % 2 == 0, 'the batch size of disTextIterator is not an even number'

        self.maxlen = maxlen
        self.n_words_target = n_words_target
        self.end_of_data = False


    def __iter__(self):
        return self


    def reset(self):
        self.positive.seek(0)
        self.negative.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        positive = []
        negative = []
        x = []
        y = []
        try:
            while True:
                ss = self.positive.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dis_dict[w] if w in self.dis_dict else 1 for w in ss]
                if self.n_words_target > 0:
                    ss = [w if w < self.n_words_target else 1 for w in ss]

                tt = self.negative.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dis_dict[w] if w in self.dis_dict else 1 for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                positive.append(ss)
                negative.append(tt)
                x = positive + negative
                positive_labels = [[0, 1] for _ in positive]
                negative_labels = [[1, 0] for _ in negative]
                y = positive_labels + negative_labels
                shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                x_np = numpy.array(x)
                y_np = numpy.array(y)
                x_np_shuffled = x_np[shuffle_indices]
                y_np_shuffled = y_np[shuffle_indices]

                x_shuffled = x_np_shuffled.tolist()
                y_shuffled = y_np_shuffled.tolist()

                if len(x_shuffled) >= self.batch_size and len(y_shuffled) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if len(positive) <= 0 or len(negative) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return x_shuffled, y_shuffled

class genTextIterator():
    def __init__(self, train_data, source_dict, batch_size=1, maxlen=30, n_words_source=-1):
        self.source = fopen(train_data, 'r')

        vocab_source = [line.split()[0] for line in codecs.open(source_dict, "r", "utf-8").read().splitlines()]
        self.source_dict = {idx: word for idx, word in enumerate(vocab_source)}

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        try:
            while True:
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                if len(ss) > self.maxlen:
                    continue

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source

class TextIterator():
    def __init__(self, source, target, source_dict, target_dict, batch_size=128, maxlen=100, n_words_source=-1, n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        vocab_source = [line.split()[0] for line in codecs.open(source_dict, "r", "utf-8").read().splitlines()]
        self.source_dict = {idx: word for idx, word in enumerate(vocab_source)}
        vocab_target = [line.split()[0] for line in codecs.open(target_dict, "r", "utf-8").read().splitlines()]
        self.target_dict = {idx: word for idx, word in enumerate(vocab_target)}

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:
            while True:
                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from target file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if len(source)<=0 or len(target)<=0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target