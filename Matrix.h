//
// Created by Andrew Xia on 5/12/23.
//

#ifndef MACHINELEARNINGLIBRARY_MATRIX_H
#define MACHINELEARNINGLIBRARY_MATRIX_H

#include <iostream>

template<typename T, int WIDTH, int HEIGHT>
class Matrix {
private:
    T* a;

public:

    Matrix() {
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = 0;
            }
        }
    }

    Matrix(T* i_a) {
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = *i_a;
                i_a++;
            }
        }
    }

    Matrix(const Matrix<T, WIDTH, HEIGHT>& other) {
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = other(j, i);
            }
        }
    }

    Matrix<T, WIDTH, HEIGHT>& operator=(const Matrix<T, WIDTH, HEIGHT>& other) {
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = other(j, i);
            }
        }
        return *this;
    }


    T& operator()(int y, int x) {
        return a[y * WIDTH + x];
    }

    Matrix<T, WIDTH, HEIGHT> operator+(Matrix<T, WIDTH, HEIGHT> other) {
        // Coefficient-wise add
        Matrix<T, WIDTH, HEIGHT> m;
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) + other(j, i);
            }
        }
        return m;
    }

    Matrix<T, WIDTH, HEIGHT> operator-(Matrix<T, WIDTH, HEIGHT> other) {
        // Coefficient-wise subtract
        Matrix<T, WIDTH, HEIGHT> m;
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) - other(j, i);
            }
        }
        return m;
    }

    Matrix<T, WIDTH, HEIGHT> coeff_mult(Matrix<T, WIDTH, HEIGHT> other) {
        // Coefficient-wise mult
        Matrix<T, WIDTH, HEIGHT> m;
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) * other(j, i);
            }
        }
        return m;
    }

    template<int w2>
    Matrix<T, w2, HEIGHT> operator*(Matrix<T, w2, WIDTH> other) {
        // Matrix multiplication
        Matrix<T, w2, HEIGHT> m;
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < w2; i++) {
                for (int k = 0; k < WIDTH; k++) {
                    m(j, i) += (*this)(j, k) * other(k, i);
                }
            }
        }
        return m;
    }

    void apply(T func(T)) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = func((*this)(j, i));
            }
        }
    }

    void print() {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                std::cout << (*this)(j, i) << ' ';
            }
            std::cout << '\n';
        }
    }

    struct Dimensions {
        int y, x;
    };

    Dimensions get_dimensions() {
        return Dimensions{HEIGHT, WIDTH};
    }
};


#endif //MACHINELEARNINGLIBRARY_MATRIX_H
