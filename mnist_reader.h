// Author: Jesse Flores

#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <vector>
#include <string>
#include <fstream>
using namespace std;

// A structure to hold the MNIST dataset
struct MNIST_Dataset {
    vector<vector<double>> images;
    vector<int> labels;
};

// Function to load the MNIST images
vector<vector<double>> load_mnist_images(const string& path);

// Function to load the MNIST labels
vector<int> load_mnist_labels(const string& path);

#endif