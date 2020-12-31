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
	_import_array();
}


cv::Mat _iceWrapAutoGammaRGB(const cv::Mat &src, const cv::Mat &mask=cv::Mat())
{ return ice::autoAdjustGammaRGB(src, mask); }


PYBIND11_MODULE(im_enhance, mod) {

	module_init();

	mod.doc() = "im_enhance is a python module to image contrast enhancement";

	mod.def("autoAdjustGammaRGB", &_iceWrapAutoGammaRGB,
		"Automatic gamma adjusment",
		py::arg("source"),
		py::arg("mask")=cv::Mat());

	mod.def("exposureFusion", &ice::exposureFusion, "Exposure Fusion");
}
