import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary

class Layer:
    def __init__(self, input_dim, output_dim, activation='tanh'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.z = None
        self.a = None
        self.input = None

    def activate(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Invalid activation function")

    def activate_derivative(self, x):
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError("Invalid activation function")

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.activate(self.z)
        return self.a

    def backward(self, delta):
        d_z = delta * self.activate_derivative(self.z)
        d_W = np.dot(self.input.T, d_z)
        d_b = np.sum(d_z, axis=0, keepdims=True)
        d_input = np.dot(d_z, self.W.T)
        return d_W, d_b, d_input

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, nn_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        super().__init__(nn_dims[0], nn_dims[-1], nn_dims[-1], actFun_type, reg_lambda, seed)
        self.nn_dims = nn_dims
        self.layers = []
        for i in range(len(nn_dims) - 1):
            self.layers.append(Layer(nn_dims[i], nn_dims[i + 1], actFun_type))

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        exp_scores = np.exp(X)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        num_examples = len(X)
        probs = self.feedforward(X)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        data_loss += self.reg_lambda / 2 * sum(np.sum(np.square(layer.W)) for layer in self.layers)
        return (1. / num_examples) * data_loss

    def backprop(self, X, y):
        num_examples = len(X)
        probs = self.feedforward(X)

        # Gradient of the output layer
        delta = probs
        delta[range(num_examples), y] -= 1
        delta /= num_examples

        gradients = []
        for i in reversed(range(len(self.layers))):
            d_W, d_b, delta = self.layers[i].backward(delta)
            gradients.append((d_W, d_b))

        return list(reversed(gradients))

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            gradients = self.backprop(X, y)

            for j, layer in enumerate(self.layers):
                d_W, d_b = gradients[j]
                layer.W += -epsilon * (d_W + self.reg_lambda * layer.W)
                layer.b += -epsilon * d_b

            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")

    def predict(self, X):
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)

def generate_data(dataset_func=datasets.make_moons, **kwargs):
    np.random.seed(0)
    X, y = dataset_func(**kwargs)
    return X, y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def main():
    # Generate and visualize Make-Moons dataset
    X, y = generate_data(datasets.make_moons, n_samples=200, noise=0.20)
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Make-Moons Dataset")
    plt.show()

    # Test different network configurations
    configs = [
        ([2, 5, 2], 'sigmoid'),
        ([2, 10, 5, 2], 'tanh'),
        ([2, 20, 10, 5, 2], 'relu'),
    ]

    for nn_dims, activation in configs:
        model = DeepNeuralNetwork(nn_dims, actFun_type=activation)
        model.fit_model(X, y)

        plt.figure(figsize=(10, 5))
        plot_decision_boundary(model, X, y)
        plt.title(f"{len(nn_dims) - 1}-layer NN, {activation} activation")
        plt.show()

    # Test on a different dataset (Make-Circles)
    configs = [
        ([2, 5, 2], 'sigmoid'),
        ([2, 10, 5, 2], 'tanh'),
        ([2, 10, 5, 4, 2], 'relu'),
    ]

    X, y = generate_data(datasets.make_circles, n_samples=200, noise=0.05, factor=0.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Make-Circles Dataset")
    plt.show()

    for nn_dims, activation in configs:
        model = DeepNeuralNetwork(nn_dims, actFun_type=activation)
        model.fit_model(X, y)

        plt.figure(figsize=(10, 5))
        plot_decision_boundary(model, X, y)
        plt.title(f"{len(nn_dims) - 1}-layer NN, {activation} activation")
        plt.show()

if __name__ == "__main__":
    main()