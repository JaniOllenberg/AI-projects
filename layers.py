import numpy as np
np.random.seed(0)

X = [[1, 2 , 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8],
     [1.5, 2.0, 0.4, -1.3]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

print(np.zeros((1,5)))

layer1 = Layer_Dense(4,10)
layer2 = Layer_Dense(10,2)
# layer1_output = layer1.forward(X)
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)