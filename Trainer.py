import numpy as np
import tensorflow as tf
import datetime
import random

"""
    Shape(row, column)
"""


"""
    Load data and get all the chars in text.
"""
text = open('clean').read()
chars = sorted(list(set(text)))
char_size = len(chars)
print('char_size : ' + str(char_size))

"""
    create dictionary to link each char to an id, and vice versa
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
batch_size = 50
max_steps = 3
log_every = 1
save_every = 3
hidden_nodes = 2

"""
    Directory to store a trained model
"""
checkpoint_directory = 'ckpt/model' # + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

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
    """
    1D  :   Store batch
    2D  :   Store input chars section
    3D  :   Store char
    """
    data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])    # input data

    """
        1D  :   Store batch
        3D  :   Store output char
        """
    labels = tf.placeholder(tf.float32, [batch_size, char_size])    # output data

    """
        Why Weights is random with;
            mean = -0.1 and standard deviation = 0.1?
        If weight starts with 0, then signal will be cancelled and will output 0. Then there will be no weight change,
        and weight will continue being 0.
        
        Bias starts with 0.
        
        Weights for Input Signals (w_ii, w_fi, w_oi & w_ci):
            Dimension   -   Weight for each char (in this case 50 chars) and in every laywer
    """
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

    """
        lstm(i, o, s):
        Take in Inputs i and Outpus o, and Previous State s in batches and compute the
        state and output of the current state.
        _____________________________________________________        
            Input   i   :   shape=(batch_size, char_size)
            Output  o   :   shape=(batch_size, hidden_layers)
            State   s   :   shape=(batch_size, hidden_layers)
            
            i * w_i     :   shape=(batch_size, hidden_nodes)
            o * w_o     :   shape=(batch_size, hidden_nodes)
            Bias b      :   shape=(1, hidden_nodes)
            
            gate        =   (i * w_i) + (o * w_o) + b
            gate        :   shape=(batch_size, hidden_layers)
        _____________________________________________________
    """
    def lstm(i, o, s):
        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
        memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

        s = forget_gate * s + input_gate * memory_cell
        o = output_gate * tf.tanh(s)

        return o, s

    # start with initial empty output and state
    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])

    # loop through all the sections.
    for i in range(len_per_section):

        """
            data[:, i, :]   :   i(th) section from all batches as input
            output          :   output from previous input
            state           :   state  from previous input
        """
        output, state = lstm(data[:, i, :], output, state)

        """
            outputs_all_i   :   
            labels_all_i    :   
        """

        if i == 0:  # if first section
            outputs_all_i = output  # make current output the start
            labels_all_i = data[:, i + 1, :]    # make next input as the start

        elif i != len_per_section - 1:  # not first or last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)   # append the current output
            labels_all_i = tf.concat([labels_all_i, data[:, i + 1, :]], 0)  # append the next input

        else:   # the last section
            outputs_all_i = tf.concat([outputs_all_i, output], 0)   # append the current output
            labels_all_i = tf.concat([labels_all_i, labels], 0)     # append empty label

    w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))

    logits = tf.matmul(outputs_all_i, w) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_all_i))

    optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)

    ###########
    # Test
    ###########

    test_data = tf.placeholder(tf.float32, shape=[1, char_size])
    test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
    test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

    # Reset at the beginning of each test
    reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])),
                                test_state.assign(tf.zeros([1, hidden_nodes])))

    """
        Input   i   :   shape=(batch_size, char_size)
        Output  o   :   shape=(batch_size, hidden_layers)
        State   s   :   shape=(batch_size, hidden_layers)
    """
    # LSTM
    # test_output, test_state = lstm(test_data, test_output, test_state)
    w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))
    test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)


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


def sample(prediction):
    #Samples are uniformly distributed over the half-open interval
    r = random.uniform(0,1)
    #store prediction char
    s = 0
    #since length > indices starting at 0
    char_id = len(prediction) - 1
    #for each char prediction probabilty
    for i in range(len(prediction)):
        #assign it to S
        s += prediction[i]
        #check if probability greater than our randomly generated one
        if s >= r:
            #if it is, thats the likely next char
            char_id = i
            break
    #dont try to rank, just differentiate
    #initialize the vector
    char_one_hot = np.zeros(shape=[char_size])
    #that characters ID encoded
    #https://image.slidesharecdn.com/latin-150313140222-conversion-gate01/95/representation-learning-of-vectors-of-words-and-phrases-5-638.jpg?cb=1426255492
    char_one_hot[char_id] = 1.0
    return char_one_hot

test_start = 'I plan to make the world a better place for every one'


with tf.Session(graph=graph) as sess:
    # init graph, load model
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(checkpoint_directory)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # set input variable to generate chars from
    reset_test_state.run()
    test_generated = test_start

    """

    test_X = np.zeros((len_per_section, char_size), dtype=int)

    for i in range(len_per_section):
        test_X[i][char2id[test_start[i]]] = 1

    prediction = sess.run(test_prediction, feed_dict={test_data: test_X})

    print(len(prediction))
    print(len(prediction[0]))
    
    """

    # for every char in the input sentennce
    for i in range(len(test_start) - 1):
        # initialize an empty char store
        test_X = np.zeros((1, char_size))
        # store it in id from
        test_X[0, char2id[test_start[i]]] = 1
        # feed it to model, test_prediction is the output value
        _ = sess.run(test_prediction, feed_dict={test_data: test_X})

    # where we store encoded char predictions
    test_X = np.zeros((1, char_size))
    test_X[0, char2id[test_start[-1]]] = 1.

    # lets generate 500 characters
    for i in range(500):
        # get each prediction probability
        prediction = test_prediction.eval({test_data: test_X})[0]
        # one hot encode it
        next_char_one_hot = sample(prediction)
        # get the indices of the max values (highest probability)  and convert to char
        next_char = id2char[np.argmax(next_char_one_hot)]
        # add each char to the output text iteratively
        test_generated += next_char
        # update the
        test_X = next_char_one_hot.reshape((1, char_size))

    print(test_generated)
