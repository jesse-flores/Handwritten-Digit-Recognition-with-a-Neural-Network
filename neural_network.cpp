// Author: Jesse Flores

#include "neural_network.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>
#include <fstream>
using namespace std;

/**
 * Name: Neuron
 * Purpose: Represents a single neuron in the neural network.
 * Inputs:
 * - input_size: The number of inputs to the neuron.
 * - gen: A random number generator for weight initialization.
 * Returns: A Neuron object with initialized weights and bias.
 * Effects: Initializes weights using He initialization for ReLU activation and sets a small positive bias.
 * Note: The weights are initialized to random values based on the input size, and the bias is set to a small positive value to ensure the ReLU activation function fires initially.
 */
Neuron::Neuron(int input_size, mt19937& gen) {
    if (input_size == 0) return;
    // He weight initialization for ReLU activation
    normal_distribution<> d(0.0, sqrt(2.0 / input_size));
    bias = 0.01; // Small positive bias to ensure ReLU fires initially
    weights.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
        weights[i] = d(gen);
    }
}

/**
 * Name: Layer
 * Purpose: Represents a layer of neurons in the neural network.
 * Inputs:
 * - num_neurons: The number of neurons in the layer.
 * - input_size: The number of inputs to each neuron in the layer.
 * - gen: A random number generator for weight initialization.
 * Returns: A Layer object containing the specified number of neurons.
 * Effects: Initializes each neuron in the layer with He initialization for weights and a small positive bias.
 */
Layer::Layer(int num_neurons, int input_size, mt19937& gen) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(input_size, gen);
    }
}
/**
 * Name: NeuralNetwork
 * Purpose: Constructs a neural network with the specified layer sizes.
 * Inputs:
 * - layer_sizes: A vector of integers representing the number of neurons in each layer.
 * Returns: A NeuralNetwork object with initialized layers.
 * Effects: Initializes the neural network with layers based on the provided sizes, using a random number generator for weight initialization.
 * Note: The first element in layer_sizes is the input layer size, and the subsequent elements
 */
NeuralNetwork::NeuralNetwork(const vector<int>& layer_sizes) {
    random_device rd;
    gen = mt19937(rd());

    // Create layers
    // input layer is conceptual, we start from the first hidden layer
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i - 1], gen);
    }
}

/**
 * Name: relu
 * Purpose: Applies the ReLU activation function to a given value.
 * Inputs:
 * - x: The input value to the ReLU function.
 * Returns: The activated value, which is max(0, x).
 * Effects: Returns the input value if it is positive, otherwise returns 0.
 * Note: This function is used in the feed-forward process of the neural network to introduce non-linearity.
 */
double NeuralNetwork::relu(double x) {
    return max(0.0, x);
}

/**
 * Name: relu_derivative
 * Purpose: Computes the derivative of the ReLU activation function.
 * Inputs:
 * - activated_value: The activated value of the neuron.
 * Returns: The derivative of the ReLU function at the given activated value.
 * Effects: Returns 1 if the activated value is greater than 0, otherwise returns 0.
 */
double NeuralNetwork::relu_derivative(double activated_value) {
    return activated_value > 0 ? 1.0 : 0.0;
}

/**
 * Name: softmax
 * Purpose: Computes the softmax probabilities from a vector of logits.
 * Inputs:
 * - logits: A vector of raw output values (logits) from the neural network.
 * Returns: A vector of probabilities corresponding to each logit, normalized to sum to 1.
 * Effects: Applies the softmax function to convert logits into probabilities.
 * Note: This function is typically used in the output layer of a classification network.
 */
vector<double> NeuralNetwork::softmax(const vector<double>& logits) {
    vector<double> probabilities;
    if (logits.empty()) return probabilities;
    probabilities.reserve(logits.size());

    double max_logit = *max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;

    for (double logit : logits) {
        double exp_val = exp(logit - max_logit);
        probabilities.push_back(exp_val);
        sum_exp += exp_val;
    }

    if (sum_exp > 1e-9) {
        // Avoid division by zero
        for (double& p : probabilities) {
            p /= sum_exp;
        }
    }
    return probabilities;
}

/**
 * Name: feed_forward
 * Purpose: Computes the output of the neural network for a given input.
 * Inputs:
 * - inputs: A vector of input values to the neural network.
 * Returns: None (the output is stored in the neurons of the last layer).
 * Effects: Propagates the input through each layer, applying the activation function to compute the activated values.
 * Note: The output layer uses a linear activation function (identity) for regression tasks or logits for classification.
 */
void NeuralNetwork::feed_forward(const vector<double>& inputs) {
    vector<double> current_inputs = inputs;

    for (size_t i = 0; i < layers.size(); ++i) {
        vector<double> next_inputs;
        next_inputs.reserve(layers[i].neurons.size());
        for (Neuron& neuron : layers[i].neurons) {
            double z = neuron.bias;
            for (size_t j = 0; j < neuron.weights.size(); ++j) {
                z += neuron.weights[j] * current_inputs[j];
            }
            neuron.z_value = z;

            // Apply ReLU for hidden layers, identity for the output layer
            if (i < layers.size() - 1) {
                neuron.value = relu(neuron.z_value);
            } else {
                neuron.value = neuron.z_value;
                // Output is raw logit
            }
            next_inputs.push_back(neuron.value);
        }
        current_inputs = move(next_inputs);
    }
}

/**
 * Name: apply_gradients
 * Purpose: Applies the computed gradients to update the weights and biases of the network.
 * Inputs:
 * - weight_gradients: A 3D vector containing the gradients for each neuron's weights.
 * - bias_gradients: A 2D vector containing the gradients for each neuron's bias.
 * - learning_rate: The learning rate for the gradient descent update.
 * - batch_size: The size of the current batch used for training.
 * Returns: None (the weights and biases are updated in place).
 * Effects: Updates the weights and biases of each neuron in the network based on the computed gradients.
 */
void NeuralNetwork::apply_gradients(
    const vector<vector<vector<double>>>& weight_gradients,
    const vector<vector<double>>& bias_gradients,
    double learning_rate, int batch_size) {

    double lr_per_batch = learning_rate / batch_size;
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].neurons.size(); ++j) {
            layers[i].neurons[j].bias -= lr_per_batch * bias_gradients[i][j];
            for (size_t k = 0; k < layers[i].neurons[j].weights.size(); ++k) {
                layers[i].neurons[j].weights[k] -= lr_per_batch * weight_gradients[i][j][k];
            }
        }
    }
}

/**
 * Name: train
 * Purpose: Trains the neural network using the provided training data and labels.
 * Inputs:
 * - training_data: A vector of vectors, where each inner vector represents an input sample.
 * - training_labels: A vector of vectors, where each inner vector represents the expected output for the corresponding input sample.
 * - epochs: The number of training epochs to run.
 * - learning_rate: The learning rate for the gradient descent update.
 * - batch_size: The size of the mini-batch used for training.
 * Returns: None (the network is trained in place).
 * Effects: Updates the weights and biases of the network based on the computed gradients from the training data.
 */
void NeuralNetwork::train(const vector<vector<double>>& training_data,
                        const vector<vector<double>>& training_labels,
                        int epochs, double learning_rate, size_t batch_size) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        vector<int> indices(training_data.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), gen);

        double total_loss = 0.0;

        for (size_t i = 0; i < training_data.size(); i += batch_size) {
            // Initialize gradients for the batch to zero
            vector<vector<vector<double>>> weight_gradients(layers.size());
            vector<vector<double>> bias_gradients(layers.size());
            for(size_t l=0; l<layers.size(); ++l) {
                weight_gradients[l].resize(layers[l].neurons.size());
                bias_gradients[l].assign(layers[l].neurons.size(), 0.0);
                for(size_t n=0; n<layers[l].neurons.size(); ++n) {
                    weight_gradients[l][n].assign(layers[l].neurons[n].weights.size(), 0.0);
                }
            }

            int current_batch_size = 0;
            for (size_t b = 0; b < batch_size && i + b < training_data.size(); ++b) {
                current_batch_size++;
                int data_index = indices[i + b];
                const auto& inputs = training_data[data_index];
                const auto& expected_outputs = training_labels[data_index];
                
                feed_forward(inputs);

                vector<double> logits;
                logits.reserve(layers.back().neurons.size());
                for(const auto& neuron : layers.back().neurons) logits.push_back(neuron.value);
                vector<double> probabilities = softmax(logits);
                for(size_t j=0; j<probabilities.size(); ++j) {
                    if (expected_outputs[j] == 1.0) {
                        total_loss -= log(probabilities[j] + 1e-9);
                    }
                }

                vector<double> deltas;
                deltas.reserve(probabilities.size());
                for (size_t j = 0; j < probabilities.size(); ++j) {
                    deltas.push_back(probabilities[j] - expected_outputs[j]);
                }

                for (int l = layers.size() - 1; l >= 0; --l) {
                    vector<double> prev_layer_outputs;
                    if (l > 0) {
                        prev_layer_outputs.reserve(layers[l-1].neurons.size());
                        for(const auto& n : layers[l-1].neurons) prev_layer_outputs.push_back(n.value);
                    } else {
                        prev_layer_outputs = inputs;
                    }
                    
                    vector<double> new_deltas;
                    if (l > 0) new_deltas.assign(layers[l-1].neurons.size(), 0.0);
                    
                    for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                        double delta = deltas[j];
                        bias_gradients[l][j] += delta;
                        for (size_t k = 0; k < prev_layer_outputs.size(); ++k) {
                            weight_gradients[l][j][k] += delta * prev_layer_outputs[k];
                        }
                        if (l > 0) {
                             for (size_t k = 0; k < layers[l-1].neurons.size(); ++k) {
                                new_deltas[k] += layers[l].neurons[j].weights[k] * delta;
                            }
                        }
                    }
                    
                    if (l > 0) {
                        vector<double> propagated_deltas;
                        propagated_deltas.reserve(layers[l-1].neurons.size());
                         for (size_t k = 0; k < layers[l-1].neurons.size(); ++k) {
                            propagated_deltas.push_back(new_deltas[k] * relu_derivative(layers[l-1].neurons[k].value));
                        }
                        deltas = move(propagated_deltas);
                    }
                }
            }
            
            if(current_batch_size > 0) {
                apply_gradients(weight_gradients, bias_gradients, learning_rate, current_batch_size);
            }
        }
        cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << total_loss / training_data.size() << endl;
    }
}

/**
 * Name: predict
 * Purpose: Predicts the output for given inputs using the neural network.
 * Inputs:
 * - inputs: A vector of doubles representing the input features.
 * Returns: A vector of doubles representing the predicted output probabilities for each class.
 * Effects: Performs a feed-forward operation through the network and applies softmax to the output layer.
 */
vector<double> NeuralNetwork::predict(const vector<double>& inputs) {
    feed_forward(inputs);
    vector<double> logits;
    const Layer& output_layer = layers.back();
    logits.reserve(output_layer.neurons.size());
    for (const auto& neuron : output_layer.neurons) {
        logits.push_back(neuron.value);
    }
    return softmax(logits);
}


/**
 * Name: save_weights
 * Purpose: Saves the weights and biases of the neural network to a binary file.
 * Inputs:
 * - filename: The name of the file where the weights will be saved.
 * Returns: True if the weights were successfully saved, false otherwise.
 * Effects: Writes the weights and biases of each neuron in each layer to the specified file.
 * Note: This function is useful for persisting the trained model for later use.
 */
bool NeuralNetwork::save_weights(const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file){
        cout << "Model Couldn't Be Saved!" << endl;
        return false;
    } 

    for (const auto& layer : layers) {
        for (const auto& neuron : layer.neurons) {
            file.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(neuron.bias));
            file.write(reinterpret_cast<const char*>(neuron.weights.data()), neuron.weights.size() * sizeof(double));
        }
    }
    cout << "Model Saved!\n";
    return true;
}

/**
 * Name: load_weights
 * Purpose: Loads the weights and biases of the neural network from a binary file.
 * Inputs:
 * - filename: The name of the file from which the weights will be loaded.
 * Returns: True if the weights were successfully loaded, false otherwise.
 * Effects: Reads the weights and biases of each neuron in each layer from the specified file.
 * Note: This function is useful for restoring a previously trained model.
 */
bool NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) return false;

    for (auto& layer : layers) {
        for (auto& neuron : layer.neurons) {
            file.read(reinterpret_cast<char*>(&neuron.bias), sizeof(neuron.bias));
            file.read(reinterpret_cast<char*>(neuron.weights.data()), neuron.weights.size() * sizeof(double));
        }
    }
    return true;
}