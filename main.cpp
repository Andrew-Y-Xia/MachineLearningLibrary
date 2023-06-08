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


    vector<int> layers = {784,300,200,10};
    Model m(layers);

    test_images[0].print_i();
    m.forward(test_images[0].flatten()).print_f();

    int BATCH_SIZE = 20;
    float learning_rate = 0.15;

    for (int epoch = 0; epoch < 25; epoch++) {

        // Shuffle training set
        std::mt19937 eng1(int_dist(i_gen));
        auto eng2 = eng1;
        std::shuffle(begin(images), end(images), eng1);
        std::shuffle(begin(labels), end(labels), eng2);

        // Process training data in batches
        for (int batch = 0; batch < 50000 / BATCH_SIZE; batch++) {

            // Process batch
            vector<Matrix<float>> inputs, outputs;
            for (int i = 0; i < BATCH_SIZE; i++) {
                inputs.push_back(images[batch * BATCH_SIZE + i].flatten());
                outputs.push_back(labels[batch * BATCH_SIZE + i]);
            }

            // Calculate gradient for each batch
            auto c = m.batch_gradient(inputs, outputs);
            m.update(c, learning_rate, BATCH_SIZE);
            if (batch % 100 == 0) std::cout << "+" << std::flush;
        }
        // Decrease learning rate
        learning_rate *= 0.9;

        std::cout << "\nEpoch: " << epoch << std::endl;

        // Score for accuracy
        int correct = 0;
        for (int i = 0; i < 1000; i++) {
            auto input = test_images[i].flatten();
            auto t = m.forward(input).max().y;
            if (test_labels[i](t, 0) > 0.9) correct++;
        }
        std::cout << "Correct: " << correct << "/1000\n";
    }

    int correct = 0;
    for (int i = 0; i < 10000; i++) {
        auto input = test_images[i].flatten();
        auto t = m.forward(input).max().y;
        if (test_labels[i](t, 0) > 0.9) correct++;
    }
    std::cout << "Correct: " << correct << "/10000\n";

    for (int i = 0; i < 10; i++) {
        test_images[i].print_i();
        m.forward(test_images[i].flatten()).print_f();
    }

    return 0;
}
