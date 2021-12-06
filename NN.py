"""Implementation of the building blocks for a neural network"""

import pandas as pd
import numpy as np
from sklearn import datasets

class Network(object):
    def __init__(self, sizes, input, output, lr):
        self.sizes = sizes
        self.input = input
        self.output = output
        self.weights = [np.random.rand(self.sizes[s + 1], self.sizes[s]) 
        for s in range(len(self.sizes) - 1)]
        self.biases = [np.random.rand(self.sizes[s + 1], 1)
         for s in range(len(self.sizes) - 1)]
        self.lr = lr

  
    #Forward pass
    def forward(self, x_in):
        x = x_in
        activations = [x] ###check
        for (w,b) in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w,x) + b)
            activations.append(x)
        return activations[-1], activations

    #Cost function
    def mse(self, y_pred, y):
        return (np.sum((y_pred - y)**2)/ len(y))

    def mse_prime(self, y_pred, y):
        return 2 * (y_pred - y) / len(y)


    #Activation function

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self,x):
        return x * (1 - x)

    
    def backprop(self, activations, y):
        sigmas = []
        nabla_W = []
        nabla_B = []

        #Compute output layer sigma
        y_pred = activations[-1]
        delta_C = self.mse_prime(y_pred, y)
        sigmas.append(np.multiply(delta_C, self.sigmoid_prime(y_pred)))

        #Compute the rest of the sigma vector
        for i in range(1, len(self.sizes)):
            sigma = np.multiply(np.dot(self.weights[-i].T, sigmas[-1]), self.sigmoid_prime(activations[-i-1]))
            sigmas.append(sigma)
        sigmas = sigmas[::-1] #se invierte
        #compute the nabla for w and b
        nabla_W = [sigmas[-i] * activations[-i-1].T for i in range(1,len(self.sizes))][::-1]
        nabla_B = [s for s in sigmas[1:]]

        #Update weights and biases
        self.weights = [self.weights[i] - self.lr * nabla_W[i] for i in range(len(self.weights))]
        self.biases = [self.biases[i] - self.lr * nabla_B[i] for i in range(len(self.biases))]


    
    def run(self):
        epochs = 500000
        for e in range(epochs):
            sum_error = 0
            for x, y in list(zip(self.input, self.output)):
                    y_pred, activations = self.forward(x)
                    error = self.mse(y_pred, y)
                    sum_error += error
                    self.backprop(activations, y)
            if e % 10000 == 0 :
                print("Error epoch {}----------: {}".format(e, sum_error))


""" Correccion:
- El update de weights se hace una vez recorrido todo el dataset. En cada
example se genera un nabla. Finalmente se deben promediar todos
- Se suman todos los nablas por cada example.Backprop en realidad
tiene que return nablaw y nablab (conteniendo las sumas)
- Agregar un update de weights y biases: w- 1/len(dataset)*nablaw
- """


iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target


def vectorize(x, y):
    #Vectorize x
    new_x = []
    for row in x:
        new_x.append(np.reshape(row, (len(row),1)))
    y_vector = []
    for i in y:
        y_v = np.zeros((3,1))
        y_v[i]= 1
        y_vector.append(y_v)
    return  np.array(new_x), np.array(y_vector)

    

X, y_output = vectorize(X, y)


size = [4, 10,  3]
#X = X[:1]
#y_output = y_output[:1]
nn = Network(sizes = size, input = X,  output = y_output,lr =  0.5)
#y_out, activations = nn.forward()
