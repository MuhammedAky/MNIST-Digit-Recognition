import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
            hidden_outputs = self.sigmoid(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
            final_outputs = self.sigmoid(final_inputs)

            output_error = targets - final_outputs

            output_delta = output_error * self.sigmoid_derivative(final_outputs)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_outputs)

            self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def predict(self, inputs):
        return self.forward(inputs)


input_size = GRID_SIZE * GRID_SIZE
hidden_size = 64
output_size = 10

nn = NeuralNetwork(input_size, hidden_size, output_size)