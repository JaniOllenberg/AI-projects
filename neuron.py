inputs = [1, 2, 3 , 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights =  [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

print(output)

# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     print(neuron_weights)
#     print(neuron_bias)
#     neuron_output = 0
#     for n_weight, input in zip(neuron_weights, inputs):
#         neuron_output += input * n_weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
# print(layer_outputs)

layer_outputs = []
for neuron_weights, bias in zip(weights,biases):
    neuron_output = 0
    for input, n_weight in zip(inputs, neuron_weights):
        neuron_output += input * n_weight
    neuron_output += bias
    layer_outputs.append(neuron_output)
print(layer_outputs)