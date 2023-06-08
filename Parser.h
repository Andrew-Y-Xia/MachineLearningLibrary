//
// Created by Andy on 6/7/23.
//

#ifndef MACHINELEARNINGLIBRARY_PARSER_H
#define MACHINELEARNINGLIBRARY_PARSER_H
#include <iostream>
#include <vector>
#include <fstream>
#include "Matrix.h"

using namespace std;

void read_mnist(const string& images_filepath, const string& labels_filepath, int num_images, vector<Matrix<float>>& images, vector<Matrix<float>>& labels);


#endif //MACHINELEARNINGLIBRARY_PARSER_H
