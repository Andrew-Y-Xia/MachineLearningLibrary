//
// Created by Andrew Xia on 5/12/23.
//

#ifndef MACHINELEARNINGLIBRARY_MODEL_H
#define MACHINELEARNINGLIBRARY_MODEL_H

#include <vector>
#include "Matrix.h"
#include <cstdlib>

using std::vector;

class Model {
private:
    vector<int> layers;
    vector<Matrix<float>> weights;
    vector<Matrix<float>> biases;

public:
    Model(vector<int> layers);

    Matrix<float> forward(const Matrix<float>& input);

    void print();

    void backprop(const Matrix<float>& input);

    static Matrix<float> rand_matrix(int height, int width);
};


#endif //MACHINELEARNINGLIBRARY_MODEL_H
