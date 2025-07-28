// Author: Jesse Flores

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <random>
#include <string>

// Represents a single neuron in the network
struct Neuron {
    double value = 0.0;    // Activated value: activation(z_value)
    double z_value = 0.0;  // Pre-activation value: weights * inputs + bias
    double bias = 0.0;
    std::vector<double> weights;

    // Constructor for He weight initialization
    Neuron(int input_size, std::mt19937& gen);
};

// Represents a layer of neurons
struct Layer {
    std::vector<Neuron> neurons;
    Layer(int num_neurons, int input_size, std::mt19937& gen);
};

// The main Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes);

    std::vector<double> predict(const std::vector<double>& inputs);

    void train(const std::vector<std::vector<double>>& training_data,
               const std::vector<std::vector<double>>& training_labels,
               int epochs, double learning_rate, size_t batch_size);
               // For model saving
    bool save_weights(const std::string& filename);
    bool load_weights(const std::string& filename);


private:
    std::vector<Layer> layers;
    std::mt19937 gen; // Mersenne Twister for random numbers

    void feed_forward(const std::vector<double>& inputs);
    
    // Gradient application function for mini-batching
    void apply_gradients(
        const std::vector<std::vector<std::vector<double>>>& weight_gradients,
        const std::vector<std::vector<double>>& bias_gradients,
        double learning_rate, int batch_size);

    // Activation Functions
    static double relu(double x);
    static double relu_derivative(double activated_value);
    static std::vector<double> softmax(const std::vector<double>& logits);
};

#endif