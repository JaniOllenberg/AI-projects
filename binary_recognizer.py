import random
import sys

sys.setrecursionlimit(10000)

class Neuron:
    def __init__(self, number):
        self.__number = number
        self.__connections = []

    def add_connection(self, neuron, strength):
        self.__connections.append((neuron, strength))
    
    def get_connections(self):
        return self.__connections

    def print_connections(self):
        print(self.__connections)
    
    def get_number(self):
        return self.__number

def network_output(all_neurons):
    find_path(1, 101, all_neurons)

backtracking = 0
def find_path(input, output, all_neurons):
    global backtracking
    for neuron in all_neurons:
        connections = neuron.get_connections()
        for connection in connections:
            if(connection[0] == output):
                print(neuron.get_number(), connection,"ding")
                if neuron.get_number() == input:
                    print("connection to 1 found")
                    return
                backtracking += 1
                print("backtracking", backtracking)
                find_path(input, neuron.get_number(), all_neurons)

all_neurons = []
for number in range(2,99):
    neuron = Neuron(number)
    for i in range(1,100):
        random_number = random.randint(0,101)
        while random_number == number:
            random_number = random.randint(0,101)
        neuron.add_connection(random_number, random.random())
    all_neurons.append(neuron)

for neuron in all_neurons:
    print(neuron.get_number())
    neuron.print_connections()

neuron0 = Neuron(0)
neuron1 = Neuron(1)
# neuron2.add_connection(1, 0.7)
# neuron1.add_connection(2, 0.15)
# neuron1.add_connection(2, 0.95)
neuron0.print_connections()
neuron1.print_connections()

input = 1
output = network_output(all_neurons)