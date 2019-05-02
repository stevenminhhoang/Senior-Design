import os
import pprint
import pickle

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from stock_dataset import StockDataset
from model import RNN

flags = tf.app.flags
flags.DEFINE_integer("input_size", 1, "Size of each training data point")
flags.DEFINE_integer("num_steps", 30, "Number of steps data grouped into one training input")
flags.DEFINE_integer("num_layers", 1, "Number of stacked LSTM layers")
flags.DEFINE_integer("lstm_size", 128, "Number of cell units in one LSTM layer")
flags.DEFINE_integer("batch_size", 64, "Number of data points used in one batch")
flags.DEFINE_float("keep_prob", 0.8, "Percent of units to keep in dropout")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate")
flags.DEFINE_integer("init_epoch", 5, "Number of initial epochs")
flags.DEFINE_integer("max_epoch", 50, "Number of total epochs used in training")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size")
flags.DEFINE_string("stock_symbol", None, "Stock symbol to train and make prediction")
flags.DEFINE_integer("stock_count", 100, "Number of stocks used in the dataset")
flags.DEFINE_integer("sample_size", 3, "Number of stocks to plot during training")
flags.DEFINE_boolean("train", False, "True for training, False for testing")
flags.DEFINE_boolean("write", False, "True for writing contents to the file of the same name")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_data(input_size, num_steps, k=None, stock_symbol=None, test_ratio=0.05):
    if stock_symbol is not None:
        return [
            StockDataset(
                stock_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

    # Load metadata of s & p 500 stocks
    sp500_data = pd.read_csv("data/constituents-financials.csv")
    sp500_data = sp500_data.rename(columns={col: col.lower().replace(' ', '_') for col in sp500_data.columns})
    sp500_data['file_exists'] = sp500_data['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print(sp500_data['file_exists'].value_counts().to_dict())

    sp500_data = sp500_data[sp500_data['file_exists'] == True].reset_index(drop=True)
    sp500_data = sp500_data.sort_values(by='market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        sp500_data = sp500_data.head(k)

    print("Head of S&P 500 information:\n", sp500_data.head())

    # Generate embedding meta file
    sp500_data[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataset(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05)
        for _, row in sp500_data.iterrows()]


def main(_):
    pp.pprint(FLAGS.flag_values_dict())
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = RNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            keep_prob=FLAGS.keep_prob,
            embed_size=FLAGS.embed_size
        )

        model_summary()

        stock_dataset = load_data(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            stock_symbol=FLAGS.stock_symbol,
        )

        if FLAGS.train:
            model.train(stock_dataset, FLAGS)
        else:
            test_prediction, test_loss = model.predict(stock_dataset, 50, FLAGS)
            if FLAGS.write:
                with open('api_log/'+FLAGS.stock_symbol+".pkl", 'wb') as f:
                    pickle.dump(test_prediction, f)

            if not model.load()[0]:
                raise Exception("Train a model first before running test mode!")


if __name__ == '__main__':
    tf.app.run()
