import pickle
import subprocess
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api
from json import dumps


# x = tf.placeholder("float", [None, 784])
# sess = tf.Session()
#
# # restore trained data
# with tf.variable_scope("regression"):
#     y1, variables = model.regression(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/regression.ckpt")
#
#
# with tf.variable_scope("convolutional"):
#     keep_prob = tf.placeholder("float")
#     y2, variables = model.convolutional(x, keep_prob)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/convolutional.ckpt")


# webapp
app = Flask(__name__, static_url_path='/static')
# api = Api(app)


# @app.route('/api/mnist', methods=['POST'])
# def mnist():
#     input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
#     output1 = regression(input)
#     output2 = convolutional(input)
#     return jsonify(results=[output1, output2])

# class Predict(Resource):
#     def get(self, sym):
#         command = 'python3 main.py stock_symbol=%s --write' % sym
#         subprocess.call(command.split())
#         with open('api_log/'+sym+'.pkl', 'rb') as f:
#             prediction = str(pickle.load(f))
#             return jsonify(prediction)
#
# api.add_resource(Predict, '/predict/<sym>') # Route_3


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    stock = request.form['stock'].upper()





if __name__ == '__main__':
    app.run(debug=True)
