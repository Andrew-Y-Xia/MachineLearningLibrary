//
// Created by Andrew Xia on 5/12/23.
//

#ifndef MACHINELEARNINGLIBRARY_MODEL_H
#define MACHINELEARNINGLIBRARY_MODEL_H

#include <vector>
#include "Matrix.h"
#include <cstdlib>
#include <cmath>
#include <random>

using std::vector;


struct CostGradient {
    vector<Matrix<float>> nabla_w;
    vector<Matrix<float>> nabla_b;
};

class Model {
private:
    vector<int> layers;
    vector<Matrix<float>> weights;
    vector<Matrix<float>> biases;

public:
    Model(vector<int> layers);

    Matrix<float> forward(const Matrix<float>& input);

    void print();

    CostGradient backprop(const Matrix<float>& input, const Matrix<float>& output);

    static Matrix<float> rand_matrix(int height, int width);

    CostGradient batch_gradient(const vector<Matrix<float>>& inputs, const vector<Matrix<float>>& outputs);

    void update(const CostGradient& nabla_c, float learning_rate, int batch_size);
};


#endif //MACHINELEARNINGLIBRARY_MODEL_H
