import tensorflow as tf
import numpy as np

def test(graph, checkpoint_directory):

    with tf.Session(graph=graph) as sess:
        # init graph, load model
        tf.global_variables_initializer().run()
        model = tf.train.latest_checkpoint(checkpoint_directory)
        saver = tf.train.Saver()
        saver.restore(sess, model)

        # set input variable to generate chars from
        reset_test_state.run()
        test_generated = test_start

        # for every char in the input sentennce
        for i in range(len(test_start) - 1):
            # initialize an empty char store
            test_X = np.zeros((1, char_size))
            # store it in id from
            test_X[0, char2id[test_start[i]]] = 1.
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
