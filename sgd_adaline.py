"""
Adaptive Linear Neuron (with SGD) to do binary classification using 2 labels (0 and 1).
Uses Stochastic Gradient Descent to calibrate weights and bias unit.
(see https://en.wikipedia.org/wiki/ADALINE)
"""

import numpy as np

class SGDAdaline:
    """Simple ADALINE implementation (with SGD)"""
    def __init__(self, learning_rate=0.1, epochs=500, seed=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self._weights = []
        self._bias = 0.0
        self.losses = []
        self._rgen = np.random.default_rng(self.seed)

    def fit(self, inputs, labels):
        """Calibrates neuron's weights and bias unit."""
        self._weights = self._rgen.normal(loc=0.0, scale=0.01, size=inputs.shape[1])
        self._bias = 0.0
        self.losses = []

        shuffled_inputs, shuffled_labels = self._shuffle(inputs, labels)

        for _ in range(self.epochs):
            losses = []
            for features, label in zip(shuffled_inputs, shuffled_labels):
                output = self._activation(self._net_input(features))
                error = label - output
                self._weights += 2.0 * self.learning_rate * features * error
                self._bias += 2.0 * self.learning_rate * error
                losses.append(error ** 2)
            self.losses.append(np.mean(losses))

        return self

    def predict(self, features):
        """Returns the output using current weights, bias unit and step unit function."""
        return np.where(self._activation(self._net_input(features)) >= 0.5, 1, 0)

    def _net_input(self, features):
        return np.dot(features, self._weights) + self._bias

    def _activation(self, features):
        return features

    def _shuffle(self, inputs, labels):
        permutation = self._rgen.permutation(len(labels))
        return inputs[permutation], labels[permutation]
