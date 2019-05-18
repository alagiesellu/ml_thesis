#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import zipapp
import requests
import os
import datetime
import tensorflow as tf


# In[57]:


len_per_section = 50
skip = int(len_per_section / 10)
batch_size = 25
max_steps = 1000000
log_every = 500
learning_rate = 10.

dataset_source = 'data/mini-shakespeare.txt'
send_backup_url='https://tools.sofora.net/index.php'


# In[59]:

file = open(dataset_source)
text = file.read()

text_len = len(text)


# In[60]:


chars = list(set(text))
char_size = len(chars)

char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))


# In[66]:


# suggested model Nh = 2/3 * (Ni + No) from 
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
        self.dir_count = str(len(tf.gfile.ListDirectory(logs_directory)))

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


helper = Helper(send_backup_url=send_backup_url)

helper.write_file("Step,Training Lost,Timestamp")


# In[76]:


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver()
    X_length = len(X)

    # make steps
    for step in range(max_steps):

        offset = offset % X_length

        if offset <= (X_length - batch_size):
            batch_data = X[offset: offset + batch_size]
            batch_labels = y[offset: offset + batch_size]
            offset += batch_size
        else:
            to_add = batch_size - (X_length - offset)
            batch_data = np.concatenate((X[offset:X_length], X[0: to_add]))
            batch_labels = np.concatenate((y[offset:X_length], y[0: to_add]))
            offset = to_add

        _, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})

        if (step + 1) % log_every == 0:
            saver.save(sess, helper.get_ckpt_dir(), global_step=step)
            
            helper.write_file(
                "\n" + str(step) +
                "," + str(training_loss) +
                "," + str(datetime.datetime.now()),
                print_it=True
            )

            helper.backup()

