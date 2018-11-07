import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
import codecs

from IPython import embed
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

FLAGS = tf.flags.FLAGS
'''
tf.flags.DEFINE_integer('lstm_size', 400, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 300, 'size of embedding')
tf.flags.DEFINE_string('converter_path', r'model/baiduClean3/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/baiduClean3/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')
'''

tf.flags.DEFINE_integer('lstm_size', 400, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 300, 'size of embedding')
tf.flags.DEFINE_string('converter_path', r'model/baiduClean3/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/baiduClean3/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')


FLAGS.start_string = FLAGS.start_string
converter = TextConverter(filename=FLAGS.converter_path)
if os.path.isdir(FLAGS.checkpoint_path):
    FLAGS.checkpoint_path =\
        tf.train.latest_checkpoint(FLAGS.checkpoint_path)

model = CharRNN(converter.vocab_size, sampling=True,
                lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                use_embedding=FLAGS.use_embedding,
                embedding_size=FLAGS.embedding_size)

model.load(FLAGS.checkpoint_path)

mystr =r'如果这五万对你来说很重要的话就不要轻易投'
mystr1=r'如果这万说来五对要你重很的话就要轻不易投'
mystr2=r'天刀团本怎么找队长老一有红圈就躲开'
mystr3=r'天刀团本怎找么长队老一有就躲开红圈'
mystr4=r'打开电视'
mystr5=r'打开苹果'
start = converter.text_to_arr(FLAGS.start_string)

filePath=r'd:/data/testrawsplit.txt'
fileWrite=r'd:/data/testrawsplitWrite.txt'

'''
with codecs.open(filePath,encoding='utf-8') as f:
    with open(fileWrite, 'w', encoding='utf-8') as fw:
        data=f.read().split('\n')
        cnt=len(data)
        cntCorrect=0
        for line in data:
            if len(line.strip().split('\t'))>1:
                str1=line.strip().split('\t')[0]
                str2=line.strip().split('\t')[1]
                ans1=model.samplePredictProbility(FLAGS.max_length, start, converter.vocab_size,str1,converter)
                ans2=model.samplePredictProbility(FLAGS.max_length, start, converter.vocab_size, str2, converter)
                if ans1>ans2:
                    cntCorrect+=1
                    print('1')
                else:
                    print('0')

                fw.write(str1)
                fw.write("\t")
                fw.write(str(ans1))
                fw.write("\t")
                fw.write((str2))
                fw.write("\t")
                fw.write(str(ans2))
                fw.write("\n")

    print(cntCorrect*1.0/cnt)
'''

#arr = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr,converter)

arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
arr1 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr,converter)
arr2 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr1,converter)
arr3 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr2,converter)
arr4 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr3,converter)
arr5 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr4,converter)
arr6 = model.samplePredict(FLAGS.max_length, start, converter.vocab_size,mystr5,converter)

print(converter.arr_to_text(arr))
print(converter.arr_to_text(arr1))
print(converter.arr_to_text(arr2))
print(converter.arr_to_text(arr3))
print(converter.arr_to_text(arr4))
print(converter.arr_to_text(arr5))
print(converter.arr_to_text(arr6))

