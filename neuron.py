import numpy as np
x = [
        [1.1,1.2,1.3],
        [2.1,2.2,2.3],
        [3.1,3.2,3.3]
    ]

class layer_dense:
    def __init__(self,n_input,n_neuron):
        self.weights = .1 * np.random.randn(n_input,n_neuron)
        self.bias = np.zeros((1,n_neuron))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias