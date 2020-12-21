/*
 * cv_conversion.cpp
 *
 *  Created on: Dec 21, 2020
 *      Author: sujiwo
 */

#include <exception>
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
        dtype = CV_8UC(nc);
    }else if (info.format == format_descriptor<int>::format()){
        dtype = CV_32SC(nc);
    }else if (info.format == format_descriptor<float>::format()){
        dtype = CV_32FC(nc);
    }else{
        throw std::logic_error("Unsupported type, only support uchar, int32, float");
        return false;
    }

//    value = cv::Mat(rows, cols)

	return true;
}


handle
type_caster<cv::Mat>::cast(const cv::Mat& mat, return_value_policy, handle defval)
{

}

}}
