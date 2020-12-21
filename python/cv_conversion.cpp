/*
 * cv_conversion.cpp
 *
 *  Created on: Dec 21, 2020
 *      Author: sujiwo
 */

#include <exception>
#include <string>
#include "cv_conversion.h"

namespace pybind11 { namespace detail {

bool
type_caster<cv::Mat>::load(handle obj, bool)
{
	auto b = reinterpret_borrow<array>(obj);
	buffer_info info = b.request();

	int rows = 1;
	int cols = 1;
	int channels = 1;
	int ndims = info.ndim;
	if(ndims == 2){
		rows = info.shape[0];
		cols = info.shape[1];
	} else if(ndims == 3){
		rows = info.shape[0];
		cols = info.shape[1];
		channels = info.shape[2];
	}else{
		char msg[64];
		std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
		throw std::logic_error(msg);
		return false;
	}

	int dtype;
	if(info.format == format_descriptor<unsigned char>::format()){
		dtype = CV_8UC(channels);
	}else if (info.format == format_descriptor<int>::format()){
		dtype = CV_32SC(channels);
	}else if (info.format == format_descriptor<float>::format()){
		dtype = CV_32FC(channels);
	}else if (info.format == format_descriptor<double>::format()){
		dtype = CV_64FC(channels);
	}else{
		throw std::logic_error("Unsupported type, only support uchar, int32, float");
		return false;
	}

	value = cv::Mat(rows, cols, dtype, info.ptr);
	value.addref();
	return true;
}


handle
type_caster<cv::Mat>::cast(const cv::Mat& mat, return_value_policy, handle defval)
{
	std::string format = format_descriptor<unsigned char>::format();
	size_t elemsize = sizeof(unsigned char);
	int cols = mat.cols;
	int rows = mat.rows;
	int channels = mat.channels();
	int depth = mat.depth();
	int type = mat.type();
	int dim = (depth == type)? 2 : 3;

	if(depth == CV_8U){
		format = format_descriptor<unsigned char>::format();
		elemsize = sizeof(unsigned char);
	}else if(depth == CV_32S){
		format = format_descriptor<int>::format();
		elemsize = sizeof(int);
	}else if(depth == CV_32F){
		format = format_descriptor<float>::format();
		elemsize = sizeof(float);
	}else if(depth == CV_64F){
		format = format_descriptor<double>::format();
		elemsize = sizeof(double);
	}else{
		throw std::logic_error("Unsupport type, only support uchar, int32, float");
	}

	std::vector<size_t> bufferdim;
	std::vector<size_t> strides;
	if (dim == 2) {
		bufferdim = {(size_t) rows, (size_t) cols};
		strides = {elemsize * (size_t) cols, elemsize};
	} else if (dim == 3) {
		bufferdim = {(size_t) rows, (size_t) cols, (size_t) channels};
		strides = {(size_t) elemsize * cols * channels, (size_t) elemsize * channels, (size_t) elemsize};
	}
	return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
}

}}
