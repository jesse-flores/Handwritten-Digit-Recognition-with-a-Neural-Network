// Author: Jesse Flores

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

// SFML 3.0.1 Headers
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include "neural_network.h"
using namespace std;

// Forward declarations for helper functions
vector<double> preprocess_image(const sf::Image& image);
void drawThickLine(sf::RenderTexture& canvas, const sf::Vector2f& p1, const sf::Vector2f& p2, float thickness, const sf::Color& color);


int main() {
    cout << "Application started. Press Enter to continue...";
    cin.get();

    // Load the Neural Network
    vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn(layer_sizes);
    if (!nn.load_weights("mnist_model.dat")) {
        cerr << "Error: Could not load 'mnist_model.dat'." << endl;
        return 1;
    }
    cout << "Neural network model loaded successfully." << endl;
    
    // Setup the GUI Window and Drawing Canvas
    const unsigned int WINDOW_SIZE = 448;
    const float BRUSH_THICKNESS = 40.f;
    sf::RenderWindow window(sf::VideoMode({WINDOW_SIZE, WINDOW_SIZE}), "Handwriting Recognition - SFML 3");
    window.setFramerateLimit(120);

    sf::RenderTexture canvas({WINDOW_SIZE, WINDOW_SIZE});
    canvas.clear(sf::Color::Black);

    // Main Application Loop
    sf::Vector2f last_mouse_pos;
    bool is_drawing = false;

    cout << "\n--- Controls ---\n"
              << "Draw with the mouse.\n"
              << "Press [Space] to predict.\n"
              << "Press [C] to clear.\n"
              << "Press [Escape] to close.\n" << endl;

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }

            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->scancode == sf::Keyboard::Scan::Escape) {
                    window.close();
                }
                if (keyPressed->scancode == sf::Keyboard::Scan::C) {
                    canvas.clear(sf::Color::Black);
                    cout << "Canvas cleared." << endl;
                }
                if (keyPressed->scancode == sf::Keyboard::Scan::Space) {
                    canvas.display();
                    sf::Image screenshot = canvas.getTexture().copyToImage();
                    
                    vector<double> input_vector = preprocess_image(screenshot);
                    vector<double> prediction = nn.predict(input_vector);

                    auto max_it = max_element(prediction.begin(), prediction.end());
                    int predicted_label = distance(prediction.begin(), max_it);
                    double confidence = *max_it;

                    cout << "------------------------\n"
                              << "Predicted Digit: " << predicted_label << "\n"
                              << "Confidence: " << confidence * 100.0 << "%\n"
                              << "------------------------" << endl;
                }
            }

            if (const auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mousePressed->button == sf::Mouse::Button::Left) {
                    is_drawing = true;
                    last_mouse_pos = sf::Vector2f(mousePressed->position);
                    sf::CircleShape brush(BRUSH_THICKNESS / 2.f);
                    brush.setOrigin({BRUSH_THICKNESS / 2.f, BRUSH_THICKNESS / 2.f});
                    brush.setFillColor(sf::Color::White);
                    brush.setPosition(last_mouse_pos);
                    canvas.draw(brush);
                }
            }
            
            if (const auto* mouseReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseReleased->button == sf::Mouse::Button::Left) {
                    is_drawing = false;
                }
            }

            if (const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()) {
                if (is_drawing) {
                    sf::Vector2f new_pos(mouseMoved->position);
                    drawThickLine(canvas, last_mouse_pos, new_pos, BRUSH_THICKNESS, sf::Color::White);
                    last_mouse_pos = new_pos;
                }
            }
        }

        window.clear();
        canvas.display();
        window.draw(sf::Sprite(canvas.getTexture()));
        window.display();
    }

    return 0;
}


/**
 * Name: preprocess_image
 * Purpouse: Converts an SFML image to a vector of doubles suitable for input to the neural network.
 * Inputs:
 * - image: The SFML image to preprocess.
 * Returns: A vector of doubles representing the normalized pixel values of the image.
 * Effects: Resizes the image to 28x28 pixels, normalizes pixel values to the range [0, 1], and flattens it into a vector.
 * Note: The image is expected to be in grayscale format, where each pixel's brightness is represented by a single channel (0-255).
 */
vector<double> preprocess_image(const sf::Image& image) {
    vector<double> processed_data(28 * 28, 0.0);
    sf::Vector2u image_size = image.getSize();
    float scale_x = static_cast<float>(image_size.x) / 28.0f;
    float scale_y = static_cast<float>(image_size.y) / 28.0f;

    for (unsigned int y = 0; y < 28; ++y) {
        for (unsigned int x = 0; x < 28; ++x) {
            float total_brightness = 0.0f;
            int sample_count = 0;
            for (unsigned int sy = 0; sy < static_cast<unsigned int>(scale_y); ++sy) {
                for (unsigned int sx = 0; sx < static_cast<unsigned int>(scale_x); ++sx) {
                    unsigned int original_x = static_cast<unsigned int>(x * scale_x) + sx;
                    unsigned int original_y = static_cast<unsigned int>(y * scale_y) + sy;
                    if (original_x < image_size.x && original_y < image_size.y) {
                        total_brightness += image.getPixel({original_x, original_y}).r;
                        sample_count++;
                    }
                }
            }
            processed_data[y * 28 + x] = (sample_count > 0) ? (total_brightness / sample_count / 255.0) : 0.0;
        }
    }
    return processed_data;
}

/**
 * Name: drawThickLine
 * Purpouse: Draws a thick line between two points on a render texture.
 * Inputs:
 * - canvas: The render texture to draw on.
 * - p1: The starting point of the line.
 * - p2: The ending point of the line.
 * - thickness: The thickness of the line.
 * - color: The color of the line.
 * Effects: Draws a thick line on the canvas and adds caps at the endpoints.
 */
void drawThickLine(sf::RenderTexture& canvas, const sf::Vector2f& p1, const sf::Vector2f& p2, float thickness, const sf::Color& color) {
    sf::Vector2f direction = p2 - p1;
    float length = sqrt(direction.x * direction.x + direction.y * direction.y);
    if (length == 0) return;

    sf::Vector2f unitDirection = direction / length;
    sf::Vector2f unitPerpendicular(-unitDirection.y, unitDirection.x);
    sf::Vector2f offset = (thickness / 2.f) * unitPerpendicular;

    sf::Vertex vertices[] = {
        {.position = p1 + offset, .color = color},
        {.position = p1 - offset, .color = color},
        {.position = p2 + offset, .color = color},
        {.position = p2 - offset, .color = color}
    };
    canvas.draw(vertices, 4, sf::PrimitiveType::TriangleStrip);

    sf::CircleShape cap(thickness / 2.f);
    cap.setOrigin({thickness / 2.f, thickness / 2.f});
    cap.setFillColor(color);
    cap.setPosition(p2);
    canvas.draw(cap);
}