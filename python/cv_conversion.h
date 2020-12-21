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
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>


namespace pybind11 { namespace detail {


template<>
struct type_caster<cv::Mat> {
public:
	PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

	//! 1. cast numpy.ndarray to cv::Mat
	bool load(handle obj, bool);

	//! 2. cast cv::Mat to numpy.ndarray
	static handle cast(const cv::Mat& mat, return_value_policy, handle defval);
};


template<>
struct type_caster<cv::KeyPoint> {
public:
	PYBIND11_TYPE_CASTER(cv::KeyPoint, _("cv2.KeyPoint"));
	bool load(handle obj, bool);
	static handle cast(const cv::KeyPoint& Kp, return_value_policy, handle defval);
};

}}


#endif /* IM_ENHANCE_CV_CONVERSION_H_ */
