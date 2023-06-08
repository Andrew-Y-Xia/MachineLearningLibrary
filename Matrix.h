//
// Created by Andrew Xia on 5/12/23.
//

#ifndef MACHINELEARNINGLIBRARY_MATRIX_H
#define MACHINELEARNINGLIBRARY_MATRIX_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <math.h>

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

    explicit Matrix(const std::vector<std::vector<T>>& vec) {
        HEIGHT = vec.size();
        WIDTH = vec[0].size();
        a = new T[HEIGHT * WIDTH];
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                (*this)(j, i) = vec[j][i];
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

    Matrix(Matrix&& other) noexcept : a(nullptr), HEIGHT(0), WIDTH(0) {
        a = other.a;
        HEIGHT = other.HEIGHT;
        WIDTH = other.WIDTH;
        other.a = nullptr;
    }

    ~Matrix() {
        delete[] a;
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

    int get_height() const { return HEIGHT; }

    int get_width() const { return WIDTH; }


    T& operator()(int y, int x) {
        if (y >= HEIGHT || x >= WIDTH) throw std::string("MATRIX OUT OF BOUNDS");
        return a[y * WIDTH + x];
    }

    const T& operator()(int y, int x) const {
        if (y >= HEIGHT || x >= WIDTH) throw std::string("MATRIX OUT OF BOUNDS");
        return a[y * WIDTH + x];
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
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

    Matrix<T> operator+(float f) const {
        // Coefficient-wise mult
        Matrix<T> m(HEIGHT, WIDTH);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) + f;
            }
        }
        return m;
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
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

    Matrix<T> operator-(float f) const {
        // Coefficient-wise mult
        Matrix<T> m(HEIGHT, WIDTH);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) - f;
            }
        }
        return m;
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
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

    Matrix<T> operator*(float f) const {
        // Coefficient-wise mult
        Matrix<T> m(HEIGHT, WIDTH);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = (*this)(j, i) * f;
            }
        }
        return m;
    }

    Matrix<T> dot(const Matrix<T>& other) const {
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

    Matrix<T> apply(T func(T)) const {
        Matrix m(*this);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                m(j, i) = func(m(j, i));
            }
        }
        return m;
    }

    Matrix<T> transpose() {
        Matrix m(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                m(x, y) = (*this)(y, x);
            }
        }
        return m;
    }

    Matrix<T> flatten() {
        Matrix m(a, WIDTH * HEIGHT, 1);
        return m;
    }

    struct Dimensions {
        int y, x;
    };

    Dimensions max() {
        T n = 0.0;
        Dimensions ret;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if ((*this)(y, x) > n) {
                    n = (*this)(y, x);
                    ret = Dimensions {y, x};
                }
            }
        }
        return ret;
    }

    void print_i() {
        std::cout << "Dimensions (Height, Width): " << HEIGHT << ' ' << WIDTH << "\n";
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                printf("%3d ", (int) round((*this)(j, i)));
            }
            std::cout << '\n';
        }
    }

    void print_f() {
        std::cout << "Dimensions (Height, Width): " << HEIGHT << ' ' << WIDTH << "\n";
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                std::cout << (*this)(j, i);
            }
            std::cout << '\n';
        }
    }

    Dimensions get_dimensions() const {
        return Dimensions{HEIGHT, WIDTH};
    }
};



#endif //MACHINELEARNINGLIBRARY_MATRIX_H
