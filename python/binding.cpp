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


PYBIND11_MODULE(im_enhance, mod) {

	module_init();

	mod.doc() = "im_enhance is a python module to image contrast enhancement";

	mod.def("autoAdjustGammaRGB", &ice::autoAdjustGammaRGB, "Automatic gamma adjusment");
	mod.def("exposureFusion", &ice::exposureFusion, "Exposure Fusion");
}
