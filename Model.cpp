//
// Created by Andrew Xia on 5/12/23.
//

#include "Model.h"

std::mt19937 gen(42);
std::normal_distribution<float> d(0, 1);

Model::Model(vector<int> l) {
    this->layers = move(l);
    weights = vector<Matrix<float>>();
    biases = vector<Matrix<float>>();
    for (int i = 1; i < layers.size(); i++) {
        biases.push_back(rand_matrix(layers[i], 1));
        weights.push_back(rand_matrix(layers[i], layers[i - 1]));
    }
}

float sigmoid(float f) {
    return 1.0f / (1.0f + exp(-f));
}

float sigmoid_prime(float f) {
    return sigmoid(f) * (1-sigmoid(f));
}

Matrix<float> Model::forward(const Matrix<float>& input) {
    auto activation = input;
    for (int i = 0; i < weights.size(); i++) {
        auto z = weights[i].dot(activation) + biases[i];
        activation = z.apply(sigmoid);
    }
    return activation;
}

float r(float f) {
    return d(gen);
}


Matrix<float> Model::rand_matrix(int height, int width) {
    Matrix<float> m(height, width);
    return m.apply(r);
}

void Model::print() {
    for (int i = 0; i < weights.size(); i++) {
        std::cout << "Weights #" << i + 1 << ":\n";
        weights[i].print_f();
        std::cout << "Biases #" << i + 1 << ":\n";
        biases[i].print_f();
    }

}

template <class T>
T& index(vector<T>& a, int i) {
    if (i < 0) return a[a.size() + i];
    else return a[i];
}

CostGradient Model::backprop(const Matrix<float>& input, const Matrix<float>& output) {
    vector<Matrix<float>> nabla_b(biases);
    vector<Matrix<float>> nabla_w(weights);

    auto activation = input;
    vector<Matrix<float>> activations;
    activations.push_back(activation);
    vector<Matrix<float>> zs;
    for (int i = 0; i < weights.size(); i++) {
        auto z = weights[i].dot(activation) + biases[i];
        zs.push_back(z);
        activation = z.apply(sigmoid);
        activations.push_back(activation);
    }

    auto delta = (index(activations, -1) - output) * index(zs, -1).apply(sigmoid_prime);
    index(nabla_b, -1) = delta;
    index(nabla_w, -1) = delta.dot(index(activations, -2).transpose());

    for (int l = 2; l < layers.size(); l++) {
        auto z = index(zs, -l);
        auto sp = z.apply(sigmoid_prime);
        delta = index(weights, -l + 1).transpose().dot(delta) * sp;
        index(nabla_b, -l) = delta;
        index(nabla_w, -l) = delta.dot(index(activations, -l-1).transpose());
    }
    return CostGradient { nabla_w, nabla_b };
}

CostGradient Model::batch_gradient(const vector<Matrix<float>>& inputs, const vector<Matrix<float>>& outputs) {
    assert(inputs.size() == outputs.size() && !inputs.empty());
    CostGradient nabla_c = backprop(inputs[0], outputs[0]);
    for (int i = 1; i < inputs.size(); i++) {
        auto local_c = backprop(inputs[i], outputs[i]);
        for (int l = 0; l < nabla_c.nabla_b.size(); l++) {
            nabla_c.nabla_b[l] = nabla_c.nabla_b[l] + local_c.nabla_b[l];
            nabla_c.nabla_w[l] = nabla_c.nabla_w[l] + local_c.nabla_w[l];
        }
    }
    return nabla_c;
}

void Model::update(const CostGradient& nabla_c, float learning_rate, int batch_size) {
    assert(nabla_c.nabla_w.size() == weights.size());
    for (int l = 0; l < weights.size(); l++) {
        weights[l] = weights[l] - (nabla_c.nabla_w[l] * (learning_rate/(float) batch_size));
        biases[l] = biases[l] - (nabla_c.nabla_b[l] * (learning_rate/(float) batch_size));
    }
}

