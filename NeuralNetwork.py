import numpy as np
import copy

class NeuralNetwork():
    def __init__(self):
        pass

    def initialize_parameters_deep(self, layer_dims):
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
            
        return parameters
    
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        
        cache = (A, W, b)
        
        return Z, cache
    
    def sigmoid(self, Z):
        return 1 / 1 + np.exp(-Z), Z
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        return A, Z
    
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "relu":
            A, activation_cache = self.relu(Z)
        elif activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "softmax":
            A, activation_cache = self.softmax(Z)
            
        cache = (linear_cache, activation_cache)
        return A, cache

    def L_model_forward(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2
        
        for l in range(1, L):
            A_prev = A 

            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
            caches.append(cache)
            
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
        caches.append(cache)
        
        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        cost = np.squeeze(cost)
        return cost
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def relu_backward(self, dA, activation_cache):
        A = activation_cache
        dZ = np.array(dA, copy=True)

        dZ[A <= 0] = 0

        return dZ
    
    def sigmoid_backward(self, dA, activation_cache):
        A = activation_cache
        
        dZ = dA * A * (1 - A)

        return dZ
    
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dZL = AL - Y
        current_cache = caches[L-1]
        linear_cache, _ = current_cache
        dA_prev_temp, dW_temp, db_temp = self.linear_backward(dZL, linear_cache)
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    def update_parameters(self, params, grads, learning_rate):
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

        return parameters

    def train_model(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, print_cost_every_iterations=100):
        np.random.seed(1)
        costs = []
        
        parameters = self.initialize_parameters_deep(layers_dims)
        
        # Gradient Descent
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X, parameters)
            
            cost = self.compute_cost(AL, Y)
            
            grads = self.L_model_backward(AL, Y, caches)
            
            parameters = self.update_parameters(parameters, grads, learning_rate)
            
            if print_cost and i % print_cost_every_iterations == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
        
        return parameters, costs
    
    def predict(self, X, parameters):
        # Perform forward propagation
        AL, _ = self.L_model_forward(X, parameters)
        
        # Get the index of the row with the highest value for each example
        predictions = np.argmax(AL, axis=0)
        confidences = np.max(AL, axis=0)
        
        return predictions, confidences