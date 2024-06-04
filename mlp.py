import numpy as np

class MLP:
    def __init__(self, inputs, outputs, hidden, learning_rate=0.1):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.learning_rate = learning_rate

        self.hidden_weights = np.random.uniform(0, 1, (inputs, hidden))
        self.hidden_output_weights = np.random.uniform(0, 1, (hidden, outputs))
        self.hidden_bias = np.random.uniform(0, 1, (1, hidden))
        self.output_bias = np.random.uniform(0, 1, (1, outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Corrected to use the output of the sigmoid directly

    def predict(self, inputs):
        hidden = self.sigmoid(np.dot(inputs, self.hidden_weights) + self.hidden_bias)
        output = self.sigmoid(np.dot(hidden, self.hidden_output_weights) + self.output_bias)
        return output

    def train(self, inputs, targets):
        for _ in range(10000):
            for x, y in zip(inputs, targets):
                output = self.predict(x)
                output_error = y - output
                output_delta = output_error * self.sigmoid_derivative(output)

                hidden = self.sigmoid(np.dot(x, self.hidden_weights) + self.hidden_bias)
                hidden_error = output_delta.dot(self.hidden_output_weights.T)
                hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

                self.hidden_output_weights += np.outer(hidden, output_delta) * self.learning_rate
                self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
                self.hidden_weights += np.outer(x, hidden_delta) * self.learning_rate
                self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

if __name__ == "__main__":
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    network = MLP(2, 1, 2, 0.1)  # Note: adjusted outputs from 2 to 1
    network.train(data, targets)
    for d in data:
        print(f"Input: {d} -> Output: {network.predict(d)}")
