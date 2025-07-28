// Author: Jesse Flores

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "neural_network.h"
#include "mnist_reader.h"
using namespace std;

/**
 * Name: to_one_hot
 * Purpose: Converts a vector of integer labels into a one-hot encoded format.
 * Inputs:
 * - labels: A vector of integer labels (e.g., class indices).
 * - num_classes: The total number of classes for one-hot encoding.
 * Returns: A vector of vectors, where each inner vector is a one-hot encoded representation of the corresponding label.
 * Effects: Each label is converted to a vector of size num_classes, with a 1 at the index of the label and 0s elsewhere.
 */
vector<vector<double>> to_one_hot(const vector<int>& labels, int num_classes) {
    vector<vector<double>> one_hot_labels(labels.size(), vector<double>(num_classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < num_classes) {
            one_hot_labels[i][labels[i]] = 1.0;
        }
    }
    return one_hot_labels;
}

int main() {
    // Load Data
    cout << "Loading MNIST dataset..." << endl;
    string data_path = "../data/"; 
    
    auto training_images = load_mnist_images(data_path + "train-images.idx3-ubyte");
    auto training_labels_int = load_mnist_labels(data_path + "train-labels.idx1-ubyte");
    auto test_images = load_mnist_images(data_path + "t10k-images.idx3-ubyte");
    auto test_labels_int = load_mnist_labels(data_path + "t10k-labels.idx1-ubyte");

    if (training_images.empty() || training_labels_int.empty() || test_images.empty() || test_labels_int.empty()) {
        cerr << "Failed to load dataset. Make sure the data files are in the ./data/ directory." << endl;
        return 1;
    }
    
    int num_classes = 10;
    auto training_labels = to_one_hot(training_labels_int, num_classes);

    // Initialize Network
    cout << "Initializing neural network..." << endl;
    // A slightly deeper network for better performance
    vector<int> layer_sizes = {784, 128, 64, 10}; 
    NeuralNetwork nn(layer_sizes);

    // Train Network
    cout << "Starting training..." << endl;
    int epochs = 10;
    // Update based on training data
    double learning_rate = 0.05;
    size_t batch_size = 32;
    nn.train(training_images, training_labels, epochs, learning_rate, batch_size);
    cout << "Training complete." << endl;

    // Evaluate Network
    cout << "Evaluating network on test data..." << endl;
    int correct_predictions = 0;
    int sure_predictions = 0;
    int sure_correct_predictions = 0;
    const double confidence_threshold = 0.90;

    for (size_t i = 0; i < test_images.size(); ++i) {
        vector<double> prediction_probs = nn.predict(test_images[i]);
        
        auto max_it = max_element(prediction_probs.begin(), prediction_probs.end());
        int predicted_label = distance(prediction_probs.begin(), max_it);
        double confidence = *max_it;

        if (predicted_label == test_labels_int[i]) {
            correct_predictions++;
        }

        // Check for predictions the model is 90% or greater sure about
        if (confidence >= confidence_threshold) {
            sure_predictions++;
            if (predicted_label == test_labels_int[i]) {
                sure_correct_predictions++;
            }
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / test_images.size() * 100.0;
    double sure_accuracy = (sure_predictions == 0) ? 0.0 : static_cast<double>(sure_correct_predictions) / sure_predictions * 100.0;

    cout << "----------------------------------------" << endl;
    cout << "Overall Test Accuracy: " << accuracy << "%" << endl;
    cout << "Correctly predicted " << correct_predictions << " out of " << test_images.size() << " test images." << endl;
    cout << "----------------------------------------" << endl;
    cout << "Confidence Threshold: " << confidence_threshold * 100 << "%" << endl;
    cout << "Number of 'sure' predictions: " << sure_predictions << " (" << (static_cast<double>(sure_predictions) / test_images.size() * 100.0) << "%)" << endl;
    cout << "Accuracy on 'sure' predictions: " << sure_accuracy << "%" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Training complete." << endl;
    nn.save_weights("mnist_model.dat");

    return 0;
}
