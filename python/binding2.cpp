#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

#include "im_enhance.h"
#include "cv_conversion.h"

using namespace std;
namespace py = pybind11;


void module_init()
{
	import_array();
}


int add(int i, int j)
{
	return i + j;
}

int add_vect(const std::vector<int> &vs)
{
	int s=0;

	for (auto v: vs) {
		s+=v;
	}
	return s;
}

int test_mat(cv::Mat &M)
{
	return M.rows;
}

void test_kp(cv::KeyPoint &k)
{
	std::cout << k.pt.x << ", " << k.pt.y << std::endl;
}


PYBIND11_MODULE(im_enhance, mod) {

	module_init();

	mod.doc() = "im_enhance is a python module to image contrast enhancement";

	mod.def("add", &add, "Add two numbers");

	mod.def("add_vect", &add_vect, "add a list of integers");

	mod.def("test_mat", &test_mat, "returns number of rows");

	mod.def("test_kp", &test_kp, "prints position of the keypoint");

	mod.def("autoAdjustGammaRGB", &ice::autoAdjustGammaRGB, "Automatic gamma adjusment");
	mod.def("exposureFusion", &ice::exposureFusion, "Exposure Fusion");
}
