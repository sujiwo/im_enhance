/*
 * cv_conversion.h
 *
 *  Created on: Dec 21, 2020
 *      Author: sujiwo
 */

#ifndef IM_ENHANCE_CV_CONVERSION_H_
#define IM_ENHANCE_CV_CONVERSION_H_

#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <opencv2/core.hpp>


namespace pybind11 { namespace detail {

template<>
struct type_caster<cv::Mat> {

public:
	PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

};

}}


#endif /* IM_ENHANCE_CV_CONVERSION_H_ */
