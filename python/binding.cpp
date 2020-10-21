/*
 * mod.cpp
 *
 *  Created on: Jun 24, 2020
 *      Author: sujiwo
 */

#include <Python.h>
#include <iostream>
#include "conversion.h"
#include "im_enhance.h"


// https://realpython.com/build-python-c-extension-module/

void im_test()
{
	std::cout << "Hello python\n";
}


static PyObject *method_im_test(PyObject *self, PyObject *args)
{
	std::cout << "ICE printout\n";
	Py_RETURN_NONE;
}


static PyObject *method_autoAdjustGammaRGB(PyObject *self, PyObject *args)
{
	NDArrayConverter cvt;
	PyObject *img_o;
	PyArg_ParseTuple(args, "O", &img_o);
	cv::Mat img_in, img_out;
	img_in = cvt.toMat(img_o);
	img_out = ice::autoAdjustGammaRGB(img_in);
	PyObject *obj_np = cvt.toNDArray(img_out);
	return obj_np;
}


static PyObject *method_multiScaleRetinexCP(PyObject *self, PyObject *args)
{
	NDArrayConverter cvt;
	PyObject *img_o;
	double sigma1=ice::msrcp_sigma1,
			sigma2=ice::msrcp_sigma2,
			sigma3=ice::msrcp_sigma3,
			lowClip=ice::msrcp_lowClip,
			highClip=ice::msrcp_highClip;

	PyArg_ParseTuple(args, "O|ddddd", &img_o, &sigma1, &sigma2, &sigma3, &lowClip, &highClip);
	cv::Mat img_in, img_out;
	img_in = cvt.toMat(img_o);
	img_out = ice::multiScaleRetinexCP(img_in, sigma1, sigma2, sigma3, lowClip, highClip);
	PyObject *obj_np = cvt.toNDArray(img_out);
	return obj_np;
}


static PyObject *method_dynamicHistogramEqualization(PyObject *self, PyObject *args)
{
	NDArrayConverter cvt;
	PyObject *img_o;
	PyArg_ParseTuple(args, "O", &img_o);
	cv::Mat img_in, img_out;
	img_in = cvt.toMat(img_o);
	img_out = ice::autoAdjustGammaRGB(img_in);
	PyObject *obj_np = cvt.toNDArray(img_out);
	return obj_np;
}


static PyObject *method_exposureFusion(PyObject *self, PyObject *args)
{
	NDArrayConverter cvt;
	PyObject *img_o;
	PyArg_ParseTuple(args, "O", &img_o);
	cv::Mat img_in, img_out;
	img_in = cvt.toMat(img_o);
	img_out = ice::exposureFusion(img_in);
	PyObject *obj_np = cvt.toNDArray(img_out);
	return obj_np;
}


static PyMethodDef im_enhanceMethods[] = {
    {"im_test", method_im_test, METH_NOARGS, "Test Method"},
	{"autoAdjustGammaRGB", method_autoAdjustGammaRGB, METH_VARARGS, "Automatic gamma adjusment"},
	{"multiScaleRetinexCP", method_multiScaleRetinexCP, METH_VARARGS, "Multi-scale Retinex with Color Preservation"},
	{"dynamicHistogramEqualization", method_dynamicHistogramEqualization, METH_VARARGS, "Dynamic Histogram Equalization"},
	{"exposureFusion", method_exposureFusion, METH_VARARGS, "Exposure Fusion"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initim_enhance(void)
{
	(void) Py_InitModule("im_enhance", im_enhanceMethods);
}
