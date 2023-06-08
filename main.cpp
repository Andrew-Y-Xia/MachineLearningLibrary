#include <iostream>
#include "Matrix.h"
#include "Model.h"
#include "Parser.h"

std::mt19937 i_gen(42);
std::uniform_int_distribution<int> int_dist(0, 10000);

int main() {

    vector<Matrix<float>> images;
    vector<Matrix<float>> labels;
    read_mnist("/Users/andy/Downloads/train-images.idx3-ubyte", "/Users/andy/Downloads/train-labels.idx1-ubyte", 50000, images, labels);

    vector<Matrix<float>> test_images;
    vector<Matrix<float>> test_labels;
    read_mnist("/Users/andy/Downloads/t10k-images.idx3-ubyte", "/Users/andy/Downloads/t10k-labels.idx1-ubyte", 50000, test_images, test_labels);


    vector<int> layers = {784,30,10};
    Model m(layers);
    m.print();
    int BATCH_SIZE = 10;

    m.forward(images[0].flatten()).print_f();


    for (int epoch = 0; epoch < 20; epoch++) {
        std::mt19937 eng1(int_dist(i_gen));
        auto eng2 = eng1;

        std::shuffle(begin(images), end(images), eng1);
        std::shuffle(begin(labels), end(labels), eng2);
        for (int batch = 0; batch < 50000 / BATCH_SIZE; batch++) {
            vector<Matrix<float>> inputs, outputs;
            for (int i = 0; i < BATCH_SIZE; i++) {
                inputs.push_back(images[batch * BATCH_SIZE + i].flatten());
                outputs.push_back(labels[batch * BATCH_SIZE + i]);
            }
            m.update(m.batch_gradient(inputs, outputs), 0.1, BATCH_SIZE);
            if (batch % 100 == 0) std::cout << "+" << std::flush;
        }
        std::cout << "\nEpoch: " << epoch << std::endl;

        int correct = 0;
        for (int i = 0; i < 100; i++) {
            auto input = test_images[i].flatten();
            auto t = m.forward(input).max().y;
            if (test_labels[i](t, 0) > 0.9) correct++;
        }
        std::cout << "Correct: " << correct << '\n';
    }

    m.forward(images[1].flatten()).print_f();


    int correct = 0;
    for (int i = 0; i < 1000; i++) {
        auto input = test_images[i].flatten();
        auto t = m.forward(input).max().y;
        if (test_labels[i](t, 0) > 0.9) correct++;
    }
    std::cout << "Correct: " << correct;
    return 0;
}
