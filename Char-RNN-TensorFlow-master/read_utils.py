import numpy as np
import copy
import time
import tensorflow as tf
import pickle
import jieba


def general_vocabulary_zhcn(vocab_file,output_file, min_count=2):
    """
    :param vocab_file: traindata_file,each line contains [question,answer]
    :param min_count:
    :return: key:words value:id
    """
    words_count = {}
    for line in open(vocab_file, 'r', encoding='utf-8'):
        line = line.strip()
        for word in line:
            if word not in words_count:
                words_count[word] = 1
            else:
                words_count[word] += 1

    for line in open(output_file, 'r', encoding='utf-8'):
        line = line.strip()
        for word in line:
            if word not in words_count:
                words_count[word] = 1
            else:
                words_count[word] += 1

    # filter uncommon words
    words = [word for word, freq in words_count.items() if freq >= min_count]
    words = sorted(words, key=words_count.get, reverse=True)
    gen_vocab = {words[i]: i for i in range(len(words))}
    gen_vocab[u'UNK'] = len(gen_vocab)

    return gen_vocab

#TODO: need select zhcn
def seq2id_train(question_vocab, data_file):
    data = []
    f = open(data_file, 'r', encoding='utf-8')
    question_max_length = 0
    answer_max_length = 0
    for line in f.readlines():
        wordData=[]
        q_words = line.strip()
        question_max_length = max(question_max_length, len(q_words))
        for word in q_words:
            wordData.append(question_vocab.get(word, question_vocab[u'UNK']))
        data.append(wordData)
    return data

def batch_iter(input,output, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    input = np.array(input)
    output=np.array(output)
    input_data_size = len(input)
    num_batches_per_epoch = int((len(input)-1)/batch_size) + 1
    for epoch in range(num_epochs):

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, input_data_size)
            if (end_index == input_data_size):
                start_index = end_index-batch_size
            x=input[start_index:end_index]
            y=output[start_index:end_index]
            yield x,y

def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)

    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)

            text=text.replace(" ","").replace("\n","").replace("\r","")

            '''
            segText = jieba.cut(text)
            segText=set(segText)
            text=segText
            vocab=segText
            '''
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    '''
    def text_to_arrPredict(self, text):
        arr = []
        text = text.replace(" ", "").replace("\n", "").replace("\r", "")
        segText = jieba.cut(text)
        segText = list(segText)
        for word in segText:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    '''
    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
