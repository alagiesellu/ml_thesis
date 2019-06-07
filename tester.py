#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import requests
import zipapp
import requests
import os
import datetime
import random
import tensorflow as tf


# In[2]:


len_per_section = 50
skip = int(len_per_section / 10)
batch_size = 25
max_steps = 1000000
log_every = 500
learning_rate = 10.
names = []
added = {}

dataset_source = 'data/mini-shakespeare.txt'
send_backup_url='https://tools.sofora.net/index.php'


# In[3]:


file = open(dataset_source)

for line in file.readlines():

    name = line.split(':')[0]

    if added.get(name) is None:
        names.append(name)

    added[name] = True

names.sort()
added = None


file = open(dataset_source)

text = file.read()[:500000]
file.close()

text_len = len(text)


# In[60]:


chars = list(set(text))
char_size = len(chars)

char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))


# In[66]:


# suggested model =>> Nh = 2/3 * (Ni + No) from 
# https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
hidden_nodes = int(2/3 * (len_per_section * char_size))


# In[67]:


sections = []
next_chars = []

for i in range(0, text_len - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])


# In[68]:


text = None


# In[3]:


X = np.zeros((len(sections), len_per_section, char_size), dtype=int)
y = np.zeros((len(sections), char_size), dtype=int)


# In[33]:


for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1.
    y[i, char2id[next_chars[i]]] = 1.


# In[70]:


def lstm(i, o, s):
    input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
    forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
    output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
    memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

    s = forget_gate * s + input_gate * memory_cell
    o = output_gate * tf.tanh(s)

    return o, s


# In[72]:


graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0)
    
    data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
    
    labels = tf.placeholder(tf.float32, [batch_size, char_size])
    
    w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32))
    w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32))
    b_i = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32))

    w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32))
    w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32))
    b_f = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32))

    w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32))
    w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32))
    b_o = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32))

    w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1, tf.float32))
    w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1, tf.float32))
    b_c = tf.Variable(tf.zeros([1, hidden_nodes], tf.float32))
    
    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])
    
    for i in range(len_per_section):

        output, state = lstm(data[:, i, :], output, state)

        if i == 0:  # if first section
            outputs_all_i = output  # make current output the start
            labels_all_i = data[:, i + 1, :]  # make next input as the start

        elif i != len_per_section - 1:  # not first or last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)  # append the current output
            labels_all_i = tf.concat([labels_all_i, data[:, i + 1, :]], 0)  # append the next input

        else:  # the last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)  # append the current output
            labels_all_i = tf.concat([labels_all_i, labels], 0)  # append empty label

    w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))

    logits = tf.matmul(outputs_all_i, w) + b
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_all_i))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ###########
    # Test Graph tensors
    ###########

    test_data = tf.placeholder(tf.float32, shape=[1, char_size])

    test_output = tf.Variable(tf.zeros([1, hidden_nodes]))

    test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

    reset_test_state = tf.group(
        test_output.assign(tf.zeros([1, hidden_nodes])),
        test_state.assign(tf.zeros([1, hidden_nodes]))
    )

    test_output, test_state = lstm(test_data, test_output, test_state)
    test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)


# In[73]:


logs_directory = 'logs'


class Helper:

    def __init__(self, send_backup_url=None):

        if tf.gfile.Exists(logs_directory) is False:
            tf.gfile.MakeDirs(logs_directory)

        self.url = send_backup_url
        self.dir_count = str(len(tf.gfile.ListDirectory(logs_directory)) - 1)

        self.send_zip = 'send_' + self.dir_count + '.zip'

        self.logs_directory = logs_directory + "/" + self.dir_count
        self.model_dir = self.logs_directory + "/model"
        self.output_file = self.logs_directory + "/__main__.py"

        tf.gfile.MakeDirs(self.model_dir)

    def backup(self):
        """
        After every model saving, text generated, loss, step, and time log is saved.
        Also with the model. All is zipped and send online for backup.
        :return:
        """
        if self.url is not None:

            # zip backup folder
            zipapp.create_archive(self.logs_directory, self.send_zip)

            # then send zipped folder to the URL
            try:
                requests.post(self.url, files={
                    'uploaded_file': (os.path.basename(self.send_zip), open(self.send_zip, 'rb')),
                })
            except requests.exceptions.ConnectionError as error:
                print(error)

    def get_ckpt_dir(self):
        return self.model_dir + "/model"

    def write_file(self, _content, print_it=False):
        if print_it:
            print(_content)
        _file = open(self.output_file, 'a')
        _file.write(_content)
        _file.close()


# In[74]:


helper = Helper()


def sample(prediction):

    char_id = 0

    for i in range(len(prediction)):

        if prediction[char_id] < prediction[i]:
            char_id = i

    char_one_hot = np.zeros(shape=[char_size])

    # that characters ID encoded
    # https://image.slidesharecdn.com/latin-150313140222-conversion-gate01/95/representation-learning-of-vectors-of-words-and-phrases-5-638.jpg?cb=1426255492
    char_one_hot[char_id] = 1.

    return char_one_hot


def _sample(prediction):
    # Samples are uniformly distributed over the half-open interval
    r = random.uniform(0, 1)
    # store prediction char
    s = 0
    # since length > indices starting at 0
    char_id = len(prediction) - 1
    # for each char prediction probabilty
    for i in range(len(prediction)):
        # assign it to S
        s += prediction[i]
        # check if probability greater than our randomly generated one
        if s >= r:
            # if it is, thats the likely next char
            char_id = i
            break
    # dont try to rank, just differentiate
    # initialize the vector
    char_one_hot = np.zeros(shape=[char_size])

    # that characters ID encoded
    # https://image.slidesharecdn.com/latin-150313140222-conversion-gate01/95/representation-learning-of-vectors-of-words-and-phrases-5-638.jpg?cb=1426255492
    char_one_hot[char_id] = 1.

    return char_one_hot


# In[76]:


with tf.Session(graph=graph) as sess:

    # init graph, load model
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(helper.get_ckpt_dir())
    saver = tf.train.Saver()
    saver.restore(sess, model)

    while True:
        reset_test_state.run()  # reset the output and state

        # initialize an empty char store
        test_X = np.zeros((1, char_size))

        _input = input('Generate >>')

        if _input == 'exit()':
            exit()

        _input = names[random.randint(0, len(names)-1)] + ': '

        # start feeding all chars in the test_start text, to start a sequence
        for i in range(len(_input)):
            char_id = char2id[_input[i]]
            # store it in id from
            test_X[0, char_id] = 1.

            # feed it to model, test_prediction is the output value
            _ = sess.run(test_prediction, feed_dict={test_data: test_X})

            # reset char store
            test_X[0, char_id] = 0.

        # where we store encoded char predictions
        test_X[0, char2id[_input[-1]]] = 1. # store the last char of input

        next_char = ''
        text_generated = _input

        # generate until an end of line
        while next_char != '\n':
            # get each prediction probability
            prediction = test_prediction.eval({test_data: test_X})[0]

            # one hot encode it
            next_char_one_hot = sample(prediction)

            # get the indices of the max values (highest probability)  and convert to char
            next_char = id2char[np.argmax(next_char_one_hot)]

            # add each char to the output text iteratively
            text_generated += next_char

            # update the
            test_X = next_char_one_hot.reshape((1, char_size))

        print('AI   >>   ' + text_generated)

