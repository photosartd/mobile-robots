#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "C++ to python example";
    m.def("add", &add, "A function which adds 2 numbers");
}