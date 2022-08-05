import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import os
import numpy as np
from quantization_util import input_max_value, input_min_value


class Relu7Model:
    def __init__(self):

        self.weights = []
        self.biases = []
        self.clip_by_0 = True

    def push_layer(self,w,b):
        
        self.weights.append(w)
        self.biases.append(b)

    def _layer(self,x,W,b):

        h1 = np.matmul(x,W) + b
        return np.clip(h1,0,7)

    def summary(self):
        for i in range(len(self.weights)):
            print('Layer {} W[{}], b[{}]'.format(i,str(self.weights[i].shape),str(self.biases[i].shape)))

    def forward(self,input_data):
        
        h1 = input_data
        for i in range(len(self.weights)):
            h1 = self._layer(h1,self.weights[i],self.biases[i])

        return h1

    def create_debug_mlp_0(self):
        w = np.zeros([2,2],dtype=np.float32)
        b = np.zeros([2],dtype=np.float32)
        w[0,0] = 1.0
        w[1,1] = -1.0

        self.weights.append(w)
        self.biases.append(b)

    def create_debug_mlp_1(self):
        w = np.zeros([2,2],dtype=np.float32)
        b = np.zeros([2],dtype=np.float32)
        w[0,0] = 1.0
        w[1,0] = -0.25
        w[1,0] = 0.25
        w[1,1] = -1.0

        self.weights.append(w)
        self.biases.append(b)

    def create_debug_mlp_2(self):
        w = np.zeros([2,2],dtype=np.float32)
        b = np.zeros([2],dtype=np.float32)
        w[0,0] = 1.0
        w[1,0] = -0.25
        w[1,0] = 0.25
        w[1,1] = -1.0
        b[0] = 0.125
        b[1] = 0.5

        self.weights.append(w)
        self.biases.append(b)

    def create_debug_mlp_3(self):
        w = np.zeros([2,2],dtype=np.float32)
        b = np.zeros([2],dtype=np.float32)
        w[0,0] = 1.0
        w[1,1] = -1.0

        self.weights.append(w)
        self.biases.append(b)

        w = np.zeros([2,2],dtype=np.float32)
        b = np.zeros([2],dtype=np.float32)
        w[0,0] = 1.0
        w[0,1] = 0.25
        w[1,0] = 0.75

        self.weights.append(w)
        self.biases.append(b)

    def save(self,path):
        np.savez(
            path, 
            w=self.weights,
            b=self.biases,
        )

    def get_input_size(self):
        return self.weights[0].shape[0]
    def get_output_size(self):
        return self.weights[-1].shape[1]
    def count_layers(self):
        return len(self.weights)
    
    def load(self,path):
        npzfile = np.load(path,encoding="latin1",allow_pickle=True)
        self.weights = npzfile['w']
        self.biases = npzfile['b']
