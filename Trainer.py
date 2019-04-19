import numpy as np
import tensorflow as tf
import datetime

"""
    Training session
"""
def train(vars):

    max_steps = vars['max_steps']
    graph = vars['graph']
    X = vars['X']
    batch_size = vars['batch_size']
    optimizer = vars['optimizer']
    loss = vars['loss']
    data = vars['data']
    labels = vars['labels']
    checkpoint_directory = vars['checkpoint_directory']

    vars = None

    log_every = max_steps / 100
    save_every = log_every / 10

    with tf.Session(graph=graph) as sess:

        tf.global_variables_initializer().run()
        offset = 0
        saver = tf.train.Saver()
        X_length = len(X)

        """
            For every training step feed in data with size batch_size.
        """
        for step in range(max_steps):

            offset = offset % X_length

            # if batch size is less than offset to length of X
            if offset <= (X_length - batch_size):
                batch_data = X[offset:offset+batch_size]
                batch_labels = y[offset:offset+batch_size]
                offset += batch_size

            # else if not less, take remaining from the beginning of X again.
            else:
                to_add = batch_size - (X_length - offset)
                batch_data = np.concatenate((X[offset:X_length], X[0: to_add]))
                batch_labels = np.concatenate((y[offset:X_length], y[0: to_add]))
                offset = to_add

            # feed in data and label for training with optimizer and loss function
            _, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})

        if step % log_every == 0:

            # print log message
            print('training loss at step %d: %.2f (%s)' % (step, training_loss, datetime.datetime.now()))

            if step % save_every == 0:

                # save trained model
                saver.save(sess, checkpoint_directory + '/model', global_step=step)
