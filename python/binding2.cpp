#include <iostream>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;


int add(int i, int j)
{
	return i + j;
}


PYBIND11_MODULE(im_enhance, mod) {

	mod.doc() = "im_enhance is a python module to image contrast enhancement";

	mod.def("add", &add, "Add two numbers");
}
