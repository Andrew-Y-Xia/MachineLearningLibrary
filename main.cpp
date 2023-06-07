#include <iostream>
#include "Matrix.h"
#include "Model.h"

int main() {
    std::srand(42);

    vector<int> layers = {748,100,100,100,10};
    Model m(layers);
    m.forward(Model::rand_matrix(748, 1)).print();

    /*
    Matrix<float> a = Model::rand_matrix(3, 3);
    Matrix<float> b = Model::rand_matrix(3, 1);
    a.print(); b.print();
    a.dot(b).print();
     */

    return 0;
}
