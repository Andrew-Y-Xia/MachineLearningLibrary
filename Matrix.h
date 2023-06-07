//
// Created by Andrew Xia on 5/12/23.
//

#ifndef MACHINELEARNINGLIBRARY_MATRIX_H
#define MACHINELEARNINGLIBRARY_MATRIX_H

#include <iostream>
#include <string>
#include <cstdlib>

template<typename T>
class Matrix {
private:
    T* a;

    int HEIGHT;
    int WIDTH;

public:

    Matrix(int height, int width) {
        HEIGHT = height;
        WIDTH = width;
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = 0;
            }
        }
    }

    Matrix(T* i_a, int height, int width) {
        HEIGHT = height;
        WIDTH = width;
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = *i_a;
                i_a++;
            }
        }
    }

    Matrix(const Matrix<float>& other) {
        Dimensions d = other.get_dimensions();
        HEIGHT = d.y;
        WIDTH = d.x;
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = other(j, i);
            }
        }
    }

    static void swap(Matrix& a, Matrix& b) {
        std::swap(a.a, b.a);
        std::swap(a.WIDTH, b.WIDTH);
        std::swap(a.HEIGHT, b.HEIGHT);
    }

    Matrix<T>& operator=(Matrix<T> other) {
        swap(*this, other);
        return *this;
    }

    int get_height() const { return HEIGHT;}
    int get_width() const { return WIDTH;}


    T& operator()(int y, int x) {
        if (y >= HEIGHT || x >= WIDTH) throw std::string("MATRIX OUT OF BOUNDS");
        return a[y * WIDTH + x];
    }

    const T& operator()(int y, int x) const {
        if (y >= HEIGHT || x >= WIDTH) throw std::string("MATRIX OUT OF BOUNDS");
        return a[y * WIDTH + x];
    }

    Matrix<T> operator+(Matrix<T> other) {
        // Coefficient-wise add
        assert(this->get_height() == other.get_height() && this->get_width() == other.get_width());
        Matrix<T> m(other.get_height(), other.get_width());
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) + other(j, i);
            }
        }
        return m;
    }

    Matrix<T> operator-(Matrix<T> other) {
        // Coefficient-wise subtract
        assert(this->get_height() == other.get_height() && this->get_width() == other.get_width());
        Matrix<T> m(HEIGHT, WIDTH);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) - other(j, i);
            }
        }
        return m;
    }

    Matrix<T> operator*(Matrix<T> other) {
        // Coefficient-wise mult
        assert(this->get_height() == other.get_height() && this->get_width() == other.get_width());
        Matrix<T> m(HEIGHT, WIDTH);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) * other(j, i);
            }
        }
        return m;
    }

    Matrix<T> dot(Matrix<T> other) {
        // Matrix multiplication
        assert(this->get_width() == other.get_height());
        int w2 = other.get_width();
        Matrix<T> m(HEIGHT, w2);
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
        std::cout << "Dimensions (Height, Width): " << HEIGHT << ' ' << WIDTH << "\n";
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

    Dimensions get_dimensions() const {
        return Dimensions{HEIGHT, WIDTH};
    }
};


#endif //MACHINELEARNINGLIBRARY_MATRIX_H
