//
// Created by Andy on 6/7/23.
//

#include "Parser.h"


// TAKEN FROM https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

// TAKEN FROM https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
void ReadMNIST_internal(const string& filepath, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file (filepath,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}

typedef unsigned char uchar;


uchar** read_mnist_images(string full_path, int& number_of_images, int image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}



uchar* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void read_mnist(const string& images_filepath, const string& labels_filepath, int num_images, vector<Matrix<float>>& images, vector<Matrix<float>>& labels) {
    /*
    vector<vector<double>> arr;
    ReadMNIST_internal(filepath, num_images,784,arr);
    vector<Matrix<float>> image_ret;
    for (int y = 0; y < arr.size(); y++) {
        vector<float> v;
        for (int x = 0; x < arr[y].size(); y++) {
            v.push_back((float) arr[y][x]);
        }
        Matrix<float> m(&*v.begin(), 28, 28);
        image_ret.push_back(m);
    }
     */

    vector<vector<double>> arr;
    uchar** image_pointer = read_mnist_images(images_filepath, num_images, 784);
    vector<Matrix<float>> image_ret;
    for (int y = 0; y < num_images; y++) {
        vector<float> v;
        for (int x = 0; x < 784; x++) {
            v.push_back((float) image_pointer[y][x]);
        }
        Matrix<float> m(&*v.begin(), 28, 28);
        image_ret.push_back(m);
        delete[] image_pointer[y];
    }
    delete[] image_pointer;

    vector<Matrix<float>> label_ret;
    uchar* label_pointer = read_mnist_labels(labels_filepath, num_images);
    for (int i = 0; i < num_images; i++) {
        Matrix<float> output(10, 1);
        output(label_pointer[i], 0) = 1.0;
        label_ret.push_back(output);
    }
    delete[] label_pointer;

    images = move(image_ret);
    labels = move(label_ret);
}
