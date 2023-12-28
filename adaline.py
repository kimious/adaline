"""
Adaptive Linear Neuron to do binary classification using 2 labels (0 and 1).
(see https://en.wikipedia.org/wiki/ADALINE)
"""

import numpy as np

class Adaline:
    """Simple ADALINE implementation"""
    def __init__(self, learning_rate=0.1, epochs=500, seed=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self._weights = []
        self._bias = 0.0
        self.losses = []
        self._rgen = np.random.default_rng(self.seed)

    def fit(self, inputs, labels):
        """Calibrates perceptron's weights and bias unit."""
        self._weights = self._rgen.normal(loc=0.0, scale=0.01, size=inputs.shape[1])
        self._bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            outputs = self._activation(self._net_input(inputs))
            errors = labels - outputs
            self._weights += 2.0 * self.learning_rate * np.dot(inputs.T, errors) / inputs.shape[0]
            self._bias += 2.0 * self.learning_rate * errors.mean()
            self.losses.append((errors ** 2).mean())

        return self

    def predict(self, features):
        """Returns the output using current weights, bias unit and step unit function."""
        return np.where(self._activation(self._net_input(features)) >= 0.5, 1, 0)

    def _net_input(self, features):
        return np.dot(features, self._weights) + self._bias

    def _activation(self, features):
        return features
