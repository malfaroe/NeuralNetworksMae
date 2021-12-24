##GRADIENT CHECKING MODULE
"""Utility for checking if the backpropagation self-made unit is working properly
The module compares the numerically computed gradient with the gradient calculated with the network backprop
checking the criteria: norm(backgrad- num_grad) /(norm(backgrad) + norm(num_grad)) < epsilon
Parameters:
net: instantiation of the network
Ex: net2 = Dense(sizes = [4,10,3], activations = [Sigmoid(), Softmax()], Loss = CategoricalCrossEntropyLoss(),
            epochs = EPOCHS, metric = Accuracy(), learning_rate = 0.05)
            For execution use method gradient_check(X, y, epsilon)"""


import numpy as np


class GradientChecking():
    def __init__(self, net):
        self.net = net
    
    def get_params(self):
        """Returns an unique array containing the value of all weights together (rolled)"""
        return np.concatenate([w.ravel() for w in self.net.weights])

    def set_weights(self, weight_vector):
            """Reshapes weight_vector with the original weights shape"""
            passed = 0
            resized_vector = []
            for i in range(len(self.net.weights)):
                vector = np.array(weight_vector[passed: passed+ self.net.weights[i].size])
                passed += self.net.weights[i].size
                resized_vector.append(vector.reshape(self.net.weights[i].shape))
            return resized_vector
            
    def forward_for_checking(self, inputs, weights,biases):
        #"""Performs a forward pass using the parameters of interest"""
            x = inputs
            activated_layers = [x]
            for w,b, act in zip(weights, biases, self.net.activations):
                activation = act.activate(np.dot(x, w.T) + b)
                activated_layers.append(activation)
                x = activation
            return activated_layers


    def numerical_gradient(self, X, y, epsilon):
        """Returns a vector of unravel gradients of each wij
        [dw11,dw12,....,dwij...] obtained via the
        formula (J(x+e) - J(x-e)/(2*e) for each and everyone of the weights wij"""
        params = self.get_params()#vector of the gradients for each wij
        num_grad = np.zeros(params.shape)
        perturb = np.zeros(params.shape)
        for i in range(len(params)): #for each wij
            perturb[i] = epsilon
            w_plus = self.set_weights(params + perturb)
            y_pred_plus = self.forward_for_checking(X, w_plus, self.net.biases)[-1]
            loss_plus = self.net.Loss.forward_loss(y_pred_plus,y).mean()
            w_minus = self.set_weights(params - perturb)
            y_pred_minus = self.forward_for_checking(X, w_minus, self.net.biases)[-1]
            loss_minus = self.net.Loss.forward_loss(y_pred_minus,y).mean()
            #Assigns the gradient of wij
            num_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            perturb[i] = 0
        return num_grad


    def gradient_check(self, X, y, epsilon):
        """Executes the gradient checking according to the criteria"""
        num_gradient = self.numerical_gradient(X, y, epsilon)
        #Computing the gradient via backprop...
        y_pred = self.net.forward(X)
        self.net.backpropagate(y)
        backprop_gradients = self.net.gradients
        backprop_gradients = np.concatenate([g.ravel() for g in backprop_gradients])

        #Calculate and evaluate
        numerator = np.linalg.norm(backprop_gradients - num_gradient)
        denominator = np.linalg.norm(backprop_gradients) + np.linalg.norm(num_gradient)
        difference = numerator / denominator

        if difference < epsilon:
            print('The gradient is correct')
        else:
            print("El gradiente es pútrido!!! Puajj!!")
        
        print("Fórmula alternativa:", np.linalg.norm(backprop_gradients - num_gradient)/np.linalg.norm(backprop_gradients  + num_gradient))

        return difference