//
// Created by Andrew Xia on 5/12/23.
//

#include "Model.h"


Model::Model(vector<int> l) {
    this->layers = move(l);
    weights = vector<Matrix<float>>();
    biases = vector<Matrix<float>>();
    for (int i = 1; i < layers.size(); i++) {
        biases.push_back(rand_matrix(layers[i], 1));
        weights.push_back(rand_matrix(layers[i], layers[i - 1]));
    }
}

Matrix<float> Model::forward(const Matrix<float>& input) {
    auto activation = input;
    for (int i = 0; i < weights.size(); i++) {
        activation =  (weights[i].dot(activation) + biases[i]);
    }
    return activation;
}

float r(float f) {
    return static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / 4) - 2.0;
}


Matrix<float> Model::rand_matrix(int height, int width) {
    Matrix<float> m(height, width);
    m.apply(r);
    return m;
}

void Model::print() {
    for (int i = 0; i < weights.size(); i++) {
        std::cout << "Weights #" << i + 1 << ":\n";
        weights[i].print();
        std::cout << "Biases #" << i + 1 << ":\n";
        biases[i].print();
    }

}

void Model::backprop(const Matrix<float>& input) {

}
