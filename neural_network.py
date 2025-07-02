import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x)) 
        return e_x / np.sum(e_x)

ACTIVATIONS = {
    "sigmoid": ActivationFunction.sigmoid,
    "relu": ActivationFunction.relu,
    "softmax": ActivationFunction.softmax
}


class CostFunction:
    @staticmethod
    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def cross_entropy(y_pred, y_true, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  
        return -np.sum(y_true * np.log(y_pred))

COSTS = {
    "mse": CostFunction.mse,
    "cross_entropy": CostFunction.cross_entropy
}

class Agent:
    def __init__(self, network):
        class NeuralNetwork:
            def __init__(self, layers_sizes, activations, loss_name):
                self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
                self.biases = [np.random.randn(y) for y in layer_sizes[1:]]

                self.activations = [ACTIVATIONS[name] for name in activation_names]

                self.loss_function = COSTS[loss_name]

            def propagate(self, x):
                for w, b, act in zip(self.weights, self.biases, self.activations):
                    x = act(np.dot(w, x) + b)
                return x
            def mean_error(self, data):
                return np.mean([
                    self.loss_function(self.propagate(x), y)
                    for x, y in data
                ])


nn = NeuralNetwork(
    layer_sizes=[7, 10, 3],
    activation_names=["relu", "softmax"],
    loss_name="cross_entropy"
)

x = np.random.randn(7)
print(x)
output = nn.propagate(x)
print(output)  # np. [0.2, 0.6, 0.2]
