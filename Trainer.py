import numpy as np
import tensorflow as tf
import datetime
import time


"""
    Load data and get all the chars in text
"""
text = open('clean').read()
chars = sorted(list(set(text)))
char_size = len(chars)

"""
    create dictionary to link each char to an id, and vice vercer
"""
char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))


"""
    Text data have to be arranged into set of sections of {len_per_section} characters text 
    with the next character following the section as the output of the section.
    
    Then from the starting of the previous section, {skip} characters are skipped to start the next
    section that will form the input of the following input.
    
    Considering {len_per_section} to be 50.
    section_1 = text[n:n+50]        =>      next_char_1 = text[n+50]
    section_2 = text[n+1:n+1+50]    =>      next_char_2 = text[n+1+50]
    ......
    ....
"""
len_per_section = 50
skip = 2
sections = []
next_chars = []

for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])

"""
    Create two vectors of zeros to
    
    X   => to store sections, 3 dimension.
            1-D to store a char,
            2-D to store a specific section of characters
            3-D to store all the sections
    
    y   =>  to store the next chars, 2 dimension.
            1-D to store a char,
            2-D to store all the next chars
            
    dtype   =>  int
                smallest number data type. Only need to store 1 or 0
"""
X = np.zeros((len(sections), len_per_section, char_size), dtype=int)
y = np.zeros((len(sections), char_size), dtype=int)

"""
    Transfer the text data stored in the;
        sections list   =>  X vector
        next char list  =>  y vector
        
    by signing a 0 to 1 in a specific index in the 1-D. Each char have a unique id which indicates index in the 
    1-D to be signed to 1
"""
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1

"""

"""
batch_size = 500
max_steps = 1000
log_every = 10
save_every = 100
hidden_nodes = 1024

"""
    Directory to store a trained model
"""
checkpoint_directory = 'ckpt.' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

"""
    Graph data structure to be defined to outline how data will be presented to model.
    Also to outline the operations to be done to the data.
"""
graph = tf.Graph()
with graph.as_default():

    global_step = tf.Variable(0)

    # placeholders (no data in it during graph const), but will hold data during a session
    data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])    # input data
    labels = tf.placeholder(tf.float32, [batch_size, char_size])    # output data

    w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_i = tf.Variable(tf.zeros([1, hidden_nodes]))

    w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_f = tf.Variable(tf.zeros([1, hidden_nodes]))

    w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_o = tf.Variable(tf.zeros([1, hidden_nodes]))

    w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_c = tf.Variable(tf.zeros([1, hidden_nodes]))

    def lstm(i, o, state):

        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
        memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

        state = forget_gate * state + input_gate * memory_cell
        output = output_gate * tf.tanh(state)

        return output, state

    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])

    for i in range(len_per_section):

        output, state = lstm(data[:, i, :], output, state)

        if i == 0:
            outputs_all_i = output
            labels_all_i = data[:, i + 1, :]
        elif i != len_per_section - 1:
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, data[:, i + 1, :]], 0)
        else:
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, labels], 0)

    w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))

    logits = tf.matmul(outputs_all_i, w) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_all_i))

    optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)


with tf.Session(graph=graph) as sess:

    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver()
    X_length = len(X)

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

        if step % log_every == 0:
            print('training loss at step %d: %.2f (%s)' % (step, training_loss, datetime.datetime.now()))

            if step % save_every == 0:
                saver.save(sess, checkpoint_directory + '/model', global_step=step)
