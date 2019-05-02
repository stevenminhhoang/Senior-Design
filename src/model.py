import os
import re
import random
import shutil
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector


class RNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 keep_prob = 0.8,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"):
        """
        Construct a RNN model using LSTM cell.

        Hyperparameters:
            stock_count: number of stocks used in the dataset
            lstm_size: number of cell units in one LSTM layer
            num_layers: number of stacked LSTM layers
            num_steps: number of steps data grouped into one training input
            input_size: size of each training data point
            keep_prob: percent of units to keep in dropout
            embed_size: length of embedding vector, only used when stock_count > 1.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.stock_count = stock_count
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        def build_lstm_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        # inputs.shape = (number of examples, number of input, shape of each input)
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        # self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # Stock symbols are mapped to integers
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")
        self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([build_lstm_cell() for i in range(self.num_layers)], state_is_tuple=True)
        else:
            cell = build_lstm_cell()

        if self.embed_size > 0 and self.stock_count > 1:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )

            # embeds.shape = (batch_size, embedding_size)
            stacked_labels = tf.tile(self.symbols, [1, self.num_steps], name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_labels)

            # After concat, inputs.shape = (batch_size, num_steps, input_size + embed_size)
            self.embedded_inputs = tf.concat([self.inputs, stacked_embeds], axis=2, name="embedded_inputs")
            self.embed_matrix_sum = tf.summary.histogram("embed_matrix", self.embed_matrix)

        else:
            self.embedded_inputs = tf.identity(self.inputs)
            self.embed_matrix_sum = None

        print("Shape:", self.inputs.shape)
        print("Embed shape:", self.embedded_inputs.shape)

        # Run dynamic RNN
        outputs, state_ = tf.nn.dynamic_rnn(cell, self.embedded_inputs, dtype=tf.float32, scope="dynamic_rnn")

        # outputs.get_shape() = (num_steps, batch_size, lstm_size)
        outputs = tf.transpose(outputs, [1, 0, 2])

        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        weighted_sum = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="lstm_state")
        self.pred = tf.matmul(last, weighted_sum) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", weighted_sum)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_sum = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optimizer")

        # Separated from train loss
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.trainables = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self, stock_dataset, config):
        assert len(stock_dataset) > 0
        self.merged_sum = tf.summary.merge_all()

        # Create logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        if self.use_embed:
            # For embedding visualization
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            projector_config = projector.ProjectorConfig()

            # Add embedding
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            # Link this tensor to its metadata file (e.g. labels).
            shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                            os.path.join(self.model_logs_dir, "metadata.tsv"))
            added_embed.metadata_path = "metadata.tsv"

            # Write projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, projector_config)

        tf.global_variables_initializer().run()

        # Merged different stocks data
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label, data in enumerate(stock_dataset):
            merged_test_X += list(data.test_X)
            merged_test_y += list(data.test_y)
            merged_test_labels += [[label]] * len(data.test_X)

        merged_test_X = np.array(merged_test_X)
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print("len(merged_test_X) =", len(merged_test_X))
        print("len(merged_test_y) =", len(merged_test_y))
        print("len(merged_test_labels) =", len(merged_test_labels))

        test_data_feed = {
            self.learning_rate: 0.0,
            # self.keep_prob: 1.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 0

        num_batches = sum(len(data.train_X) for data in stock_dataset) // config.batch_size
        random.seed(time.time())

        # Select samples for plotting
        sample_labels = range(min(config.sample_size, len(stock_dataset)))
        sample_indices = {}
        for l in sample_labels:
            symbol = stock_dataset[l].stock_symbol
            target_indices = np.array([
                i for i, symbol_label in enumerate(merged_test_labels)
                if symbol_label[0] == l])
            sample_indices[symbol] = target_indices
        print(sample_indices)

        print("Start training for stocks:", [data.stock_symbol for data in stock_dataset])
        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label, data in enumerate(stock_dataset):
                for batch_X, batch_y in data.generate_one_epoch(config.batch_size):
                    global_step += 1
                    epoch_step += 1
                    batch_labels = np.array([[label]] * len(batch_X))
                    train_data_feed = {
                        self.learning_rate: learning_rate,
                        # self.keep_prob: config.keep_prob,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                    }
                    train_loss, _, train_merged_sum = self.sess.run(
                        [self.loss, self.optimizer, self.merged_sum], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)

                    if np.mod(global_step, len(stock_dataset) * 200 / config.input_size) == 1:
                        test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                        print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                            global_step, epoch, learning_rate, train_loss, test_loss))

                        # Plot samples
                        for sample_symbol, indices in sample_indices.items():
                            image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                                sample_symbol, epoch, epoch_step))
                            sample_preds = test_pred[indices]
                            sample_truth = merged_test_y[indices]
                            self.plot_graphs(sample_preds, sample_truth, image_path, stock_symbol=sample_symbol)

                        self.save(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save(global_step)
        # print(final_pred, final_loss)
        return final_pred

    @property
    def model_name(self):
        name = "model_lstm%d_input%d_step%d" % (
            self.lstm_size, self.input_size, self.num_steps)

        if self.embed_size > 0:
            name += "_embed%d" % self.embed_size

        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    def save(self, step):
        name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, name),
            global_step=step
        )

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot_graphs(self, predicted, targets, title, stock_symbol=None, multiplier=5):
        def flatten(sequence):
            return np.array([x for y in sequence for x in y])

        actual = flatten(targets)[-200:]
        predicted = (flatten(predicted) * multiplier)[-200:]
        days = range(len(actual))[-200:]

        # Calculate profits using predicted values
        profit = 0
        correct_days = 0
        k = 200 if len(days) > 200 else len(days)
        for i in range(k):
            if(actual[i] > 0 and predicted[i] > 0) or (actual[i] < 0 and predicted[i] < 0):
                profit = profit + abs(actual[i])
                correct_days = correct_days + 1
            else:
                profit = profit - abs(actual[i])
        print("profit = {}, correct days = {}".format(profit, correct_days))

        plt.figure(figsize=(12, 6))
        plt.plot(days, actual, label='Actual values')
        plt.plot(days, predicted, label='Predicted values')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("Day")
        plt.ylabel("Normalized price")
        plt.ylim((min(actual), max(actual)))
        plt.grid(ls='--')

        if stock_symbol:
            plt.title(stock_symbol + " | Last %d days in test" % len(actual))

        plt.savefig(title, format='png', bbox_inches='tight', transparent=True)
        plt.close()


    def predict(self, dataset_list, max_epoch, config):
        merged_test_X, merged_test_y, merged_test_labels = [], [], []
        for label_, d_ in enumerate(dataset_list):
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        test_X = np.array(merged_test_X)
        test_y = np.array(merged_test_y)

        status, counter = self.load()
        if status:
            graph = tf.get_default_graph()
            test_data_feed = {
                self.learning_rate: 0.0,
                self.inputs: test_X,
                self.targets: test_y
            }
            #prediction = graph.get_tensor_by_name('output_layer/add:0')
            #loss = graph.get_tensor_by_name('train/loss_mse:0')

            # Select samples for plotting.
            sample_labels = range(min(config.sample_size, len(dataset_list)))
            sample_indices = {}
            for l in sample_labels:
                sym = dataset_list[l].stock_symbol
                target_indices = np.array([
                    i for i, sym_label in enumerate(merged_test_labels)
                    if sym_label[0] == l])
                sample_indices[sym] = target_indices


            test_prediction, test_loss = self.sess.run([self.pred, self.loss], test_data_feed)

            for sample_sym, indices in sample_indices.items():
                test_pred = test_prediction[indices]

        return test_pred, test_loss
