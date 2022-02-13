import numpy as np
np.random.seed(2)

X = [[1,2,3,4,5],
     [2,3,4,5,6],
     [3,4,5,6,7],
     [4,5,6,7,8],
     [5,6,7,8,9]]

class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.1*np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
    def forward(self, input):
        self.output = np.dot(input, self.weights)+self.biases

layer1 = Layer_Dense(5,10)
layer2 = Layer_Dense(10,200)
layer3 = Layer_Dense(200,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
layer3.forward(layer2.output)
print(layer3.output)