import numpy as np
import nnfs
from nnfs.datasets import spiral_data
x = [
        [1.1,1.2,1.3,1.4],
        [2.1,2.2,2.3,2.4],
        [3.1,3.2,3.3,3.4]
    ]

nnfs.init()
x,y = spiral_data(100,3)
class layer_dense:
    def __init__(self,n_input,n_neuron):
        self.weights = .1 * np.random.randn(n_input,n_neuron)
        self.bias = np.zeros((1,n_neuron))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias
class activation_relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1 = layer_dense(2,5)
activate1 = activation_relu()

layer1.forward(x)
activate1.forward(layer1.output)
print(activate1.output)