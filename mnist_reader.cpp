// Author: Jesse Flores

#include "mnist_reader.h"
#include <iostream>
using namespace std;

/**
 * Name: reverse_int
 * Purpose: Reverses the byte order of an integer.
 * Inputs:
 * - i: The integer to reverse.
 * Returns: The integer with its byte order reversed.
 * Effects: This is useful for reading binary data in a platform-independent way.
 */
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * Name: load_mnist_images
 * Purpose: Loads MNIST images from a binary file.
 * Inputs:
 * - path: The path to the MNIST images file.
 * Returns:
 * - Vector of vectors, where each inner vector represents an image as a flattened array of pixel values.
 * Effects: Reads the binary file, extracts image data, and normalizes pixel values to the range [0.0, 1.0].
 * Note: The images are expected to be in grayscale format, where each pixel's brightness is represented by a single channel (0-255).
 */
vector<vector<double>> load_mnist_images(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << path << endl;
        return {};
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    vector<vector<double>> images(number_of_images, vector<double>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                // Normalize pixel values to be between 0.0 and 1.0
                images[i][(n_rows * r) + c] = (double)temp / 255.0;
            }
        }
    }
    return images;
}

/**
 * Name: load_mnist_labels
 * Purpose: Loads MNIST labels from a binary file.
 * Inputs:
 * - path: The path to the MNIST labels file.
 * Returns:
 * - Vector of integers, where each integer represents the label for the corresponding image.
 * Effects: Reads the binary file and extracts label data.
 */
vector<int> load_mnist_labels(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << path << endl;
        return {};
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    vector<int> labels(number_of_labels);
    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }
    return labels;
}
