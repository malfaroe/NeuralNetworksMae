import pandas as pd
import numpy as np



class Dense():

    def __init__(self, sizes, activations, Loss, epochs, metric, learning_rate):
        self.weights =[np.random.randn(sizes[i],sizes[i-1]) for i in range(1, len(sizes))]
        self.biases =  [np.zeros((1, sizes[i])) for i in range(1, len(sizes))]
        self.activations = activations
        self.Loss = Loss
        self.epochs = epochs
        self.metric = metric
        self.learning_rate = learning_rate
        
    def forward(self, inputs):
        x = inputs
        self.activated_layers = [x]
        for w,b, act in zip(self.weights, self.biases, self.activations):
            activation = act.activate(np.dot(x, w.T) + b)
            self.activated_layers.append(activation)
            x = activation
            
            
    def backpropagate(self, y):
        #Input: activated_layers
        #output: Container Gradient dE/dw 
        #Initialize
        sigmas_box = []
        sigma_prime_box = []
    
       #Backprop the error ()
        #1. Compute output layer sigma
        loss_grad = self.Loss.loss_gradient(self.activated_layers[-1], y)
        output_sigma = self.activations[-1].output_layer_sigma(loss_grad, self.activated_layers[-1])
        sigmas_box = [output_sigma]
        #Sigmas of the rest of layers...
        for w,a,o_layer in zip(self.weights[::-1], self.activations[:-1][::-1],self.activated_layers[:-1][::-1]) :
            sigmas_box.append(np.dot(sigmas_box[-1], w) * a.sigma_prime(o_layer))
       
        #Reverse sigma_box    
        sigmas_box.reverse()
        #Biases update
        self.grad_biases = sigmas_box
        #Gradient (dE/dw):
        self.gradients = []
        for a, s in zip(self.activated_layers[:-1], sigmas_box):
            self.gradients.append(np.dot(s.T, a))
        #Nota: al hacerse la multiplicacion de todos los inputs a la vez
        #Igual se mantiene el shape de cada weight pero mientras más
        #Inputs más grandes salen los valores de cada componente de la matriz
        #Por eso después se divide cada weight por el total de inputs (mean)
        # print("Gradients shapes:")
        # for g in gradients:
        #     print(g.shape)
       

    def weight_update(self):
        for w,gw, b, gb in zip(self.weights, self.gradients, self.biases, self.grad_biases):
            w -= (self.learning_rate / len(X))* gw
            b -= (self.learning_rate / len(gb))* np.sum(gb, axis= 0)

        return self.weights, self.biases
        
    
    def train(self, X,y):
        print("Training......")
        for e in range(self.epochs):
            #Dar un forward pass y testear
            self.forward(X)
            #print("Testing output layers:", self.activated_layers[-1][:5])
            #Compute the error
            error = np.mean(self.Loss.forward_loss(self.activated_layers[-1], y))
            if e % 1 == 0:
                print("Error epoch {0}/{1} : {2}---Accuracy: {3}".format(e,self.epochs,
                                                                         error, self.metric.get_accuracy(self.activated_layers[-1], y)))
            #Backprop the error ()
            self.backpropagate(y)
            self.weights, self.biases = self.weight_update()
        print("Training done!")


            
    def SGD(self, X,y,x_test, y_test, minibatch_size):
        """Vectorized version"""
        print("SGD Training......")
        for e in range(1, self.epochs + 1):
            #tomar dataset y generar minibatches box
            minibatches = self.minibatch_generator(X,y, minibatch_size)
            Losses = []
            Accuracies = []
            for mb in minibatches:
                nabla_w, nabla_b = [], [] #box para ir guardando los dC/dw y dC/db de cada ejemplo
                input = mb[0]
                y_true = np.array(mb[1]).astype(int)
                #Dar un forward pass 
                #print("Minibatch input shape:", input.shape)
                #print("Minibatch y_true shape:", y_true.shape)
                #print("Bias", self.biases[0])

                
                self.forward(input)
                #Calcular el error
                error = np.mean(self.Loss.forward_loss(self.activated_layers[-1], y_true))
                #Guardar el error y accuracy
                Losses.append(error)
                Accuracies.append(self.metric.get_accuracy(self.activated_layers[-1], y_true))


                #Obtener los dC/dw y dC/db del minibatch usando backprop
                self.backpropagate(y_true)
                delta_nw = self.gradients #dC/dw
                delta_nb = self.grad_biases #dC/db
                self.weights = [w - (self.learning_rate/ len(mb)) * dw for w,dw in zip(self.weights, delta_nw)]
                self.biases = [b - (self.learning_rate/ len(mb)) * np.sum(db, axis = 0) for b, db in zip(self.biases, delta_nb)]
            
            #Reporte de error epoch...
            if (e % 10 == 0 ) or (e == self.epochs):
                print("Average Error epoch {0}: {1}---Average Accuracy: {2}".format(e, np.mean(Losses), np.mean(Accuracies)))
                print("Accuracy in test set:", self.evaluate_test(x_test, y_test))
                   
        print("Training complete!")

                

    def minibatch_generator(self, X,y, batch_size):
        dataset = list(zip(X,np.array(y)))
        np.random.shuffle(dataset)
        minibatches = [(X[i:i+batch_size,:], y[i:i+batch_size]) for
                        i in range(0, len(y), batch_size)]

        #si minibatch final es mas chico que el batch size se le mete desde
        #atras inputs hasta completar el tamaño batch size
        if len(minibatches[-1][0]) < batch_size:
            #print("Len minibatches -1:", len(minibatches[-1][0]))
            minibatches[-1] = (X[-batch_size:,:], y[-batch_size:])
            
        return minibatches
    
    def evaluate_test(self, x_test, y_test):
        """Evaluates the model on the test set
        input: x_test, y_test
        output: accuracy"""
        #Forward pass---obtain prediction y_pred
        self.forward(x_test)
        #Evaluate prediction with accuracy
        acc_test = self.metric.get_accuracy(self.activated_layers[-1], y_test)
        #Return accuracy
        return acc_test

        
 
class Relu():
    def activate(self, x):
        self.output = np.maximum(0,x)
        return self.output
    
    def sigma_prime(self, x):
        return 1. * (x > 0)


class Sigmoid():
    def activate(self, x):
        #np.exp - (x - np.max(x, axis = 1, keepdims= True))
        x = np.clip(x, 1e-7, 1 - 1e-7)
        self.output = 1 / (1+ np.exp (- (x - np.max(x, axis = 1, keepdims= True))))
        #self.output = 1 / (1+ np.exp(-x))
        return self.output
    
    def output_layer_sigma(self, loss_gradients, x):
        """en realidad calcula todo el sigma de una vez como dC/da * sigma_prime
        dC/da = loss_gradient"""
        
        self.output_sigma = loss_gradients * self.sigma_prime(x)
        return self.output_sigma
    
    def sigma_prime(self, x):
        return x * (1-x)

class Softmax():
    def activate(self, x):
        #Get unnormalized probs
        exp_values = np.exp(x - np.max(x, axis = 1, keepdims= True))
        #Get normalized probs
        self.output = exp_values / np.sum(exp_values, axis= 1, keepdims= True)
        return self.output
    
    def output_layer_sigma(self, loss_gradients, out_activations):
        """Dado que es complejo multplicar el jacobiano de cada input por
        #su loss_gradient por que el jac es una matrix, se hace aca todo directo y se saca 
        #el output layer sigma = dE/dsigma.dsigma/dz"""
        #Se crea un contenedor donde irá el output_sigma de cada input
        #del tamaño del loss_gradient (dinputs)
        self.output_sigma = np.empty_like(loss_gradients)

        #Tomo uno a uno los Loss_gradientes de cada input y cada
        #softmax activation de la output layer para hacer uno a uno los
        #output_sigmas...
        for index, (single_act, single_loss_grad) in enumerate(zip(out_activations, loss_gradients)):
            single_act = single_act.reshape(-1,1)
            #Calculate jacobian matrix (sigma_prime of softmax)
            jacobian_matrix = np.diagflat(single_act) - np.dot(single_act, single_act.T)
            self.output_sigma[index] = np.dot(jacobian_matrix, single_loss_grad)
        return self.output_sigma

         

    
##Loss Units
class MSE():
    
    #Forward
    def forward_loss(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        labels = len(y_pred[0])                  
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        return  np.sum((y_pred- y_true)**2, axis=1) / len(y_pred)
        
    #Derivative
    def loss_gradient(self, y_pred, y_true): #dE/dact
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        labels = len(y_pred[0])                  
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = (2/len(y_pred)) * (y_pred - y_true)
        return self.dinputs
    
    
class CategoricalCrossEntropyLoss():
    def forward_loss(self, y_pred, y_true):
         #entrega el vector de negative losses de cada sample
         y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) #recorta para evitar logs mulas
         if len(y_true.shape) == 1: #si el y_true viene en un solo vector de escalares
             #extraigo el valor que tiene el indice indicado en el y_true
             #correspondiente
             correct_confidences = y_pred[range(len(y_pred)), y_true]
        
         if len(y_true.shape) == 2: #matrix
             #lo mismo pero multiplique y sume para obtener el valor
             #que tiene el indice indicado por el y_true (el resto se hace zero
             #al multiplicar)
             correct_confidences = np.sum( y_pred * y_true, axis = 1)
        
         negative_loss_likehoods = -np.log(correct_confidences)

         return negative_loss_likehoods
    
    def loss_gradient(self, dvalues, y_true): #dE/dact
        # Number of samples
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs



class Accuracy():
#gets the accuracy of the training stage
    def get_accuracy(self, y_pred, y_true):
        #saca el indice donde esta el valor mas grande
        predictions = np.argmax(y_pred, axis= 1)

        #y_true en formato escalares
        if len(y_true.shape) == 1:
            accuracy = np.mean(predictions == y_true)
        #matrix
        elif len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis= 1)
            accuracy = np.mean(predictions == y_true) #promedia coincidencias de valor de indice

        return accuracy



###TESTING MODULE
# from sklearn import datasets

# iris = datasets.load_iris()
# X = iris.data 
# y = iris.target

# X, y = datasets.make_classification(n_samples = 50000, n_features = 8,n_redundant=0,n_informative= 5,
#                                       n_classes = 3)
# from sklearn import preprocessing

# scaler = preprocessing.StandardScaler().fit(X)

# X_scaled = scaler.transform(X)
# # if __name__ == '__main__':
#     sizes = [8, 10, 3]
#     EPOCHS = 50
#     #sizes = [4, 5,  3]
#     net = Dense(sizes, activations = [Relu(), Softmax()], Loss = CategoricalCrossEntropyLoss(),
#                 epochs = EPOCHS, metric = Accuracy(), learning_rate = 0.015)
#     # net.train(X_scaled,y)
#     net.SGD(X_scaled,y, minibatch_size = 10)

   

##NEXT....................................................
#REVIEW AND UNDERSTAND THE BACKPROP AND PARAMETER UPDATING MECHANICS
#BIAS UPDATE ESTA MUY CUTRE
#REARRANGE, EDIT DEAD CODE AND RESTACK THE CODE
#hacer un codigo explicativo de aprendizaje y generar un codigo terminado para produccion
#NEXT LEVEL: SGD
#test with train, test split
#Benchmarking with sklearn