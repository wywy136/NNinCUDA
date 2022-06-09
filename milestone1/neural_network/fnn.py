import numpy as np
import math


class FNNForMNIST:
    def __init__(self, num_dense, num_unit, learning_rate):
        self.weights = []  # will have length L-1
        self.bias = []  # will have length L-1
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.hidden_dim = num_unit
        self.hidden_layer_num = num_dense
        self.lr = learning_rate
        
        # Create weight matrix and bias vector for each layer
        for i in range(num_dense + 1):
            # Input layer
            if i == 0:
                self.weights.append(np.eye(28 * 28, self.hidden_dim))
                self.bias.append(np.ones(self.hidden_dim))
            # Output layer
            elif i == num_dense:
                self.weights.append(np.eye(self.hidden_dim, self.output_dim))
                self.bias.append(np.ones(self.output_dim))
            # Other intermediate layer
            else:
                self.weights.append(np.eye(self.hidden_dim, self.hidden_dim))
                self.bias.append(np.ones(self.hidden_dim))

        self.sigmoid_matrix = np.vectorize(self.sigmoid)
        self.sigmoid_derivative_matrix = np.vectorize(self.sigmoid_derivative)
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        return sigmoid * (1 - sigmoid)
        
    # Perform forward - backward - gradient descent
    def step(self, images: np.ndarray, labels: np.ndarray, batch_size: int):
        # images: [batch_num, 28 * 28]
        # labels: [batch_num, output_dim]
        # Forward
        self.activation = [images]  # activation[i]: activation for layer i+1 
        self.z = []  # z[i]: weighted input for layer i+1
        self.error = []  # error[i]: error for layer L - i
        # Input layer
        hidden_state = np.dot(images, self.weights[0])  # -> [batch_num, hidden_dim]
        for i in range(batch_size):
            hidden_state[i] = np.add(hidden_state[i], self.bias[0])
        self.z.append(hidden_state)
        # self.sigmoid(hidden_state)
        hidden_state = self.sigmoid_matrix(hidden_state)
        self.activation.append(hidden_state)

        # Intermediate layers
        for i in range(self.hidden_layer_num - 1):
            hidden_state = np.dot(hidden_state, self.weights[i + 1])  # -> [batch_num, hidden_dim]
            for j in range(batch_size):
                hidden_state[j] = np.add(hidden_state[j], self.bias[i + 1])
            self.z.append(hidden_state)
            hidden_state = self.sigmoid_matrix(hidden_state)
            self.activation.append(hidden_state)
        
        # Output layer
        hidden_state = np.dot(hidden_state, self.weights[-1])  # -> [batch_num, output_dim]
        for i in range(batch_size):
            hidden_state[i] = np.add(hidden_state[i], self.bias[-1])
        output = self.sigmoid_matrix(hidden_state)

        # Compute error at the output layer
        # Element-wise multiplication
        # print(labels - output)
        # print(np.linalg.norm(labels - output, axis=1))
        loss = np.sum(np.multiply(np.linalg.norm(labels - output), np.linalg.norm(labels - output)), dtype=np.float32) / batch_size
        # loss = np.sum(np.abs(labels - output), dtype=np.float32) / batch_size
        self.error.append(np.multiply(np.abs(labels - output), self.sigmoid_derivative_matrix(hidden_state)))  # -> [batch_num, output_dim]

        # Backward (L-2 steps)
        for i in range(self.hidden_layer_num-1, -1, -1):
            # self.weights[i+1]: weight at layer (i+2)
            # self.z[i]: z at layer (i+1)
            # error at layer (i+2)
            self.error.append(
                np.multiply(
                    np.dot(self.error[-1], np.transpose(self.weights[i + 1])), 
                    self.sigmoid_derivative_matrix(self.z[i])
                )
            )  # -> [batch_num, hidden_dim]
        
        # Gradient descent (L-1 steps)
        for i in range(self.hidden_layer_num, -1, -1):
            # self.error[hidden_layer_num - i]: error at layer (i+1)
            # self.activation[i]: activation at layer i+1
            # self.weights[i]: weight at layer i+1
            grad_w = np.dot(np.transpose(self.activation[i]), self.error[self.hidden_layer_num - i])
            self.weights[i] = self.weights[i] - self.lr / float(batch_size) * grad_w
            self.bias[i] = self.bias[i] - self.lr / float(batch_size) * np.sum(self.error[self.hidden_layer_num - i], axis=0)
        
        return loss
    
    # Perform only forward compute
    def predict(self, images: np.ndarray, batch_size):
        # self.activation = [images]  # activation[i]: activation for layer i+1 
        # self.z = []  # z[i]: weighted input for layer i+1
        # Input layer
        hidden_state = np.dot(images, self.weights[0])  # -> [batch_num, hidden_dim]
        for i in range(batch_size):
            hidden_state[i] = np.add(hidden_state[i], self.bias[0])
        # self.z.append(hidden_state)
        # self.sigmoid(hidden_state)
        hidden_state = self.sigmoid_matrix(hidden_state)
        # self.activation.append(hidden_state)

        # Intermediate layers
        for i in range(self.hidden_layer_num - 1):
            hidden_state = np.dot(hidden_state, self.weights[i + 1])  # -> [batch_num, hidden_dim]
            for j in range(batch_size):
                hidden_state[j] = np.add(hidden_state[j], self.bias[i + 1])
            # self.z.append(hidden_state)
            hidden_state = self.sigmoid_matrix(hidden_state)
            # self.activation.append(hidden_state)
        
        # Output layer
        hidden_state = np.dot(hidden_state, self.weights[-1])  # -> [batch_num, output_dim]
        for i in range(batch_size):
            hidden_state[i] = np.add(hidden_state[i], self.bias[-1])
        output = self.sigmoid_matrix(hidden_state)  # [batch_num, output_dim

        return np.argmax(output, axis=1)
