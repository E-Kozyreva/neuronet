import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса:", *synaptic_weights, sep="\n")

for _ in range(100_000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs
    adjustments = np.dot(input_layer.T, error * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Веса после обучения:", *synaptic_weights, sep="\n")
print("Результат после обучения:", *outputs, sep="\n")

new_inputs = np.array([1, 1, 0])
output = sigmoid(np.dot(new_inputs, synaptic_weights))

print("Новая ситуация:", output, sep="\n")
