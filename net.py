import time
import cv2
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops

def toFP16(x):
    return np.float16(x)

def loadInputData_keras_data(datum):
    input_data = np.asarray(datum)
    input_data = (input_data - np.average(input_data)) / np.std(input_data)
    return input_data

# One Hot Encoding
def loadCorrectData_keras_data(datum, t_out):
    correct = np.asarray(datum)
    correct_data = np.zeros((len(correct), t_out))
    for i in range(len(correct)):
        correct_data[i, correct[i]] = 1
    return correct_data

def loadInputData(datum):
    input_data = np.asarray(datum.data)
    input_data = (input_data - np.average(input_data)) / np.std(input_data)
    return input_data

# One Hot Encoding
def loadCorrectData(datum, t_out):
    correct = np.asarray(datum.target)
    correct_data = np.zeros((len(correct), t_out))
    for i in range(len(correct)):
        correct_data[i, correct[i]] = 1
    return correct_data
    
def splitData(input_data, correct_data):
    x_train, x_test, t_train, t_test = train_test_split(input_data, correct_data)
    return [x_train, x_test, t_train, t_test]
    
class NeuralNetwork:
    def __init__(self, layers, datum ,lr, epochs, batch_size):
        self.eta = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.interval = 1
        self.x_train = toFP16(datum[0])
        self.x_test = toFP16(datum[1])
        self.t_train = toFP16(datum[2])
        self.t_test = toFP16(datum[3])
        
        self.n_batch = len(self.x_train) // self.batch_size
        self.error_record_train = []
        self.error_record_test = []
        
        for i in range(self.epochs):
            index_random = np.arange(len(self.x_train))
            np.random.shuffle(index_random)
            
            for j in range(self.n_batch):
                mb_index = index_random[j * self.batch_size : (j+1) * self.batch_size]
                x_mb = self.x_train[mb_index, :]
                # print(self.t_train[mb_index, :])
                t_mb = self.t_train[mb_index, :]
                self.forward_propagation(x_mb)
                self.back_propagation(t_mb)
                self.update_params()
            
            error_train = self.get_error(self.x_train, self.t_train)
            self.error_record_train.append(error_train)
            error_test = self.get_error(self.x_test, self.t_test)
            self.error_record_test.append(error_test)
            
            if i%self.interval == 0:
                num = str(i+1)
                print("Epochs: " + num + "/" + str(self.epochs), 
                      "  Error Train: " + str(error_train), 
                      "  Error Test : " + str(error_test)
                )
        
        print()
        plt.figure(figsize = (15, 4))
        plt.plot(range(1, len(self.error_record_train) + 1), self.error_record_train, label = "Train")
        plt.plot(range(1, len(self.error_record_test) + 1), self.error_record_test, label = "Test")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()
        
        acc_train = self.get_accuracy(self.x_train, self.t_train)
        acc_test = self.get_accuracy(self.x_test, self.t_test)
        
        print()
        print("TRAIN ACCURACY: ", acc_train)
        print("TEST ACCURACY: ", acc_test)
        print()
    
    def LoadModel(self):
        return self.layers
    
    def UseModel(self, x):
        y = self.forward_propagation(x)
        arr = np.argmax(y, axis = 1)
        return arr
    
    def forward_propagation(self, x):
        for layer in self.layers:
            layer.forward(x)
            x = layer.y
            # assert (layer.y.dtype == np.float16)
        return x
    
    def back_propagation(self, t):
        grad_y = t
        for layer in reversed(self.layers):
            layer.backward(grad_y)
            grad_y = layer.grad_x
            # print(grad_y)
        return grad_y
    
    def update_params(self):
        for layer in self.layers:
            layer.update(self.eta)
    
    def get_error(self, x, t):
        y = toFP16(self.forward_propagation(x))
        return toFP16(-np.sum(t * np.log(y + 1e-7)) / len(y))

    def get_accuracy(self, x, t):
        y = self.forward_propagation(x)
        count = np.sum(np.argmax(y, axis = 1) == np.argmax(t, axis = 1))
        return toFP16(count / len(y))
        
class BaseLayer:
    def __init__(self):
        pass
    
    def update(self, learning_rate):
        self.w -= learning_rate * self.grad_w
        self.b -= learning_rate * self.grad_b
    
class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = toFP16(np.random.randn(n_upper, n) * toFP16(np.sqrt(2/n_upper)))
        self.b = toFP16(np.zeros(n))
    
    def forward(self, x):
        self.x = x
        self.u = np.dot(self.x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)
        
    def backward(self, grad_y):
        delta = toFP16(grad_y * np.where(self.u <= 0, 0, 1))
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis =0)
        self.grad_x = np.dot(delta, self.w.T)

class OutputLayer(BaseLayer):
    def __init__(self, n_upper, n, batch_size):
        self.w = toFP16(np.random.randn(n_upper, n) / toFP16(np.sqrt(n_upper)))
        self.b = toFP16(np.zeros(n))
        self.batch_size = batch_size
        
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = self.softmax(u)
    
    def softmax(self, x, axis=-1):
        e = math_ops.exp(x - math_ops.reduce_max(x, axis=axis, keepdims = True))
        s = math_ops.reduce_sum(e, axis=axis, keepdims = True)
        output = e / s
        return output
    
    def backward(self, t):
        delta = toFP16(self.y - t) / toFP16(self.batch_size)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis = 0)
        self.grad_x = np.dot(delta, self.w.T)


def resize(dataset, input_size):
    newarr = []
    for img in dataset:
        newarr.append(cv2.resize(img, (input_size, input_size)))
    return np.array(newarr)


input_size = 8
lr = toFP16(0.001)
epochs = 50
batch_size = 128

(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()
x_train = resize(x_train, input_size)
x_test = resize(x_test, input_size)
x_train = x_train.reshape(60000, input_size ** 2) / 255
x_test = x_test.reshape(10000, input_size ** 2) / 255
x_train = loadInputData_keras_data(x_train)
t_train = loadCorrectData_keras_data(t_train, 10)
x_test = loadInputData_keras_data(x_test)
t_test = loadCorrectData_keras_data(t_test, 10)

datum = [x_train, x_test, t_train, t_test]
start = time.time()

# Layers
layers = [
    MiddleLayer(input_size ** 2, 128),
    MiddleLayer(128, 128),
    MiddleLayer(128, 128),
    OutputLayer(128, 10, batch_size)
]

NeuralNetwork(layers, datum, lr, epochs, batch_size)
print(time.time() - start)
