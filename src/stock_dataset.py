import os
import random
import time

import numpy as np
import pandas as pd

random.seed(time.time())

class StockDataset(object):
    def __init__(self,
                 stock_symbol,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1):
        self.stock_symbol = stock_symbol
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio

        # Read csv file
        df = pd.read_csv(os.path.join("data", "%s.csv" % stock_symbol))

        self.stock_list = df['Close'].tolist()
        self.stock_list = np.array(self.stock_list)
        self.train_X, self.train_y, self.test_X, self.test_y = self.prepare_data(self.stock_list)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_symbol, len(self.train_X), len(self.test_y))

    def prepare_data(self, seq):
        # split test data into sets of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        seq = [seq[0] / seq[0][0] - 1.0] + [current / seq[i][-1] - 1.0 for i, current in enumerate(seq[1:])]

        # split test data into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_len = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_len], X[train_len:]
        train_y, test_y = y[:train_len], y[train_len:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batches = list(range(num_batches))
        random.shuffle(batches)
        for j in batches:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
