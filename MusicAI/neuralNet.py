import numpy as np

class MLP:
    def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs = 2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        activations = inputs
        for w in self.weights:
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    mlp = MLP()
    inputs = np.random.rand(mlp.num_inputs)
    outputs = mlp.forward_propagate(inputs)
    print(inputs)
    print(outputs)
