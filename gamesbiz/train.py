import os
import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.externals import joblib

from gamesbiz.resolve import paths


def read_config_file(config_json):
    """This function reads in a json file like hyperparameters.json or resourceconfig.json
    :param config_json: this is a string path to the location of the file (for both sagemaker or local)
    :return: a python dict is returned"""

    config_path = paths.config(config_json)
    if os.path.exists(config_path):
        json_data = open(config_path).read()
        return(json.loads(json_data))


def entry_point():
    """
    This function acts as the entry point for a docker container that an be used to train
    the model either locally or on Sagemaker depending in whichever context its called in as per resolve.paths class.

    """
    # Turn off TensorFlow warning messages in program output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # load training data set from csv file
    training_data_df = pd.read_csv(paths.input('training', 'sales_data_training.csv'), dtype=float)

    X_training = training_data_df.drop('total_earnings', axis=1).values
    Y_training = training_data_df[['total_earnings']].values

    # load testing data set from csv file
    test_data_df = pd.read_csv(paths.input('testing', 'sales_data_test.csv'), dtype=float)

    X_testing = test_data_df.drop('total_earnings', axis=1).values
    Y_testing = test_data_df[['total_earnings']].values

    # scale the data by first creating scalers for inputs and outputs and then scaling them both

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled_training = X_scaler.fit_transform(X_training)
    Y_scaled_training = Y_scaler.fit_transform(Y_training)

    # remember to scale the training and test data set with the same scaler
    X_scaled_testing = X_scaler.transform(X_testing)
    Y_scaled_testing = Y_scaler.transform(Y_testing)

    # create a simple json file to be later used by the inference image
    xscaler_filename = paths.model("X_scaler.save")
    yscaler_filename = paths.model("Y_scaler.save")

    joblib.dump(X_scaler, xscaler_filename)
    joblib.dump(Y_scaler, yscaler_filename)

    # create an empty dict to hold epoch, training cost and testing cost
    master_cost_holder = dict()

    # read in hyperparameters from hyperparameters.json file
    hyper_params = read_config_file('hyperparameters.json')

    # define model parameters
    RUN_NAME = "run 1 with 20 nodes"
    learning_rate = float(hyper_params['learning_rate'])
    training_epochs = int(hyper_params['training_epochs'])


    # define the number of inputs and outputs in the neural network
    number_of_inputs = 9
    number_of_outputs = 1

    # how many neurons do we want in each layer of the network
    layer_1_nodes = int(hyper_params['layer_1_nodes'])
    layer_2_nodes = int(hyper_params['layer_2_nodes'])
    layer_3_nodes = int(hyper_params['layer_3_nodes'])

    # section one: define the layers of the NN itself
    # input layer
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

    # layer 1
    with tf.variable_scope('layer_1'):
        weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
        layer_1_outputs = tf.nn.relu(tf.add(tf.matmul(X, weights), biases))

    # layer 2
    with tf.variable_scope('layer_2'):
        weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        layer_2_outputs = tf.nn.relu(tf.add(tf.matmul(layer_1_outputs, weights), biases))

    # layer 3
    with tf.variable_scope('layer_3'):
        weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        layer_3_outputs = tf.nn.relu(tf.add(tf.matmul(layer_2_outputs, weights), biases))

    # output layer
    with tf.variable_scope('output'):
        weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, number_of_outputs],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
        prediction = tf.nn.relu(tf.add(tf.matmul(layer_3_outputs, weights), biases))

    # cost function with scalar output
    with tf.variable_scope('cost'):
        Y = tf.placeholder(tf.float32, shape=(None, 1))
        cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

    # optimization operation on the graph
    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # logging tr.summary objects
    with tf.variable_scope('logging'):
        tf.summary.scalar('current cost', cost)
        tf.summary.histogram('predicted_value', prediction)
        summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    # start a session to perform operation on the graph
    with tf.Session() as session:
        # initialize all wa
        session.run(tf.global_variables_initializer())

        training_writer = tf.summary.FileWriter(paths.output('./logs/{}/training'.format(RUN_NAME)), session.graph)
        testing_writer = tf.summary.FileWriter(paths.output('./logs/{}/testing'.format(RUN_NAME)), session.graph)

        for epoch in range(training_epochs):
            session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

            if epoch % 5 == 0:
                training_cost, training_summary = session.run([cost, summary],
                                                              feed_dict={X: X_scaled_training, Y: Y_scaled_training})
                testing_cost, testing_summary = session.run([cost, summary],
                                                            feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
                print(epoch, training_cost, testing_cost)

                # write out training cost and testing cost per epoch to be read into dynamo later
                per_epoch_cost = {
                   epoch: {"training_cost": str(training_cost), "testing_cost": str(testing_cost)}
                }

                master_cost_holder.update(per_epoch_cost)

                training_writer.add_summary(training_summary, epoch)
                testing_writer.add_summary(testing_summary, epoch)

        final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

        print("Final Training Cost: {}".format(final_training_cost))
        print("Final Testing Cost: {}".format(final_testing_cost))

        # Now that you have the trained model test its predictive power
        Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})

        # unscale the data back to it's original units (dollars)
        Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

        real_earnings = test_data_df['total_earnings'].values[0]
        predicted_earnings = Y_predicted[0][0]

        print("The actual earnings of Game #1 were ${}".format(real_earnings))
        print("Our neural network predicted earnings of ${}".format(predicted_earnings))

        save_path = saver.save(session, paths.output('logs/trained_model.ckpt'))
        print("Model saved: {}".format(save_path))

        # saving the model using SavedModelBuilder
        model_builder = tf.saved_model.builder.SavedModelBuilder(paths.model('exported_model'))

        inputs = {
            'input': tf.saved_model.utils.build_tensor_info(X)
        }
        outputs = {
            'earnings': tf.saved_model.utils.build_tensor_info(prediction)
        }

        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        model_builder.add_meta_graph_and_variables(
            session,
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def},
            tags=[tf.saved_model.tag_constants.SERVING]
        )
        model_builder.save()

    with open(paths.model('cost.json'), 'w') as outfile:
        json.dump(master_cost_holder, outfile)


if __name__ == "__main__":
    entry_point()
