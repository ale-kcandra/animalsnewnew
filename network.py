import numpy as np
import scipy.special

class MNIST_Neural_Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1, epochs=5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.weights_input_hidden = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        
        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self, inputs_list):
        inputs = np.array(inputs_list).reshape(self.input_nodes, 1)
        
        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.weights_hidden_output, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output