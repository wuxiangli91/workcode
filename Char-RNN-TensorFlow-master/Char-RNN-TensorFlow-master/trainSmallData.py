import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs
import read_utils

FLAGS = tf.flags.FLAGS
'''
input_file=r'D:/shakespeare.txt'
name='shakespeare'
num_steps=50
num_seqs = 32
learning_rate=0.01
max_steps=20000
'''
tf.flags.DEFINE_string('name', 'baiduClean3', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 25, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 400, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 300, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.4, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', r'D:/data/baiduClean3.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 20000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 500, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 200000, 'max char number')


filePathInput=r'd:/data/lubinsplitinput.txt'
filePathOutput=r'd:/data/lubinsplitoutput.txt'



model_path = os.path.join('model', FLAGS.name)
if os.path.exists(model_path) is False:
    os.makedirs(model_path)

converter = TextConverter(filePathInput, filePathOutput,FLAGS.max_vocab)
converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

'''
arr = converter.text_to_arr(text)
g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
'''
input_data=read_utils.seq2id_train(converter.vocab,filePathInput)
output_data=read_utils.seq2id_train(converter.vocab,filePathOutput)

g=read_utils.batch_iter(input_data,output_data,FLAGS.num_seqs,FLAGS.num_steps)

print(converter.vocab_size)
model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )



