#include <exception>
#include <algorithm>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>


namespace ice {





cv::Mat autoAdjustGammaRGB (const cv::Mat &rgbImg, cv::InputArray mask=cv::noArray());

cv::Mat toIlluminatiInvariant (const cv::Mat &imageRgb, const float alpha);

/*
 * Retinex Family
 *
 * Suggested values:
 * Sigmas = { 15, 80, 250 }
 * low clip = 0.01
 * high clip = 0.9999999
 */

/*
 * Default values for MSRCP
 */
const float
	msrcp_sigma1 = 15.0,
	msrcp_sigma2 = 80.0,
	msrcp_sigma3 = 250.0,
	msrcp_lowClip = 0.01,
	msrcp_highClip = 0.99;

cv::Mat multiScaleRetinexCP(const cv::Mat &rgbImage,
	const float sigma1=msrcp_sigma1,
	const float sigma2=msrcp_sigma2,
	const float sigma3=msrcp_sigma3,
	const float lowClip=msrcp_lowClip,
	const float highClip=msrcp_highClip);

/*
 * Dynamic Histogram Equalization (choice #3)
 */
const float dhe_alpha = 0.5;
cv::Mat dynamicHistogramEqualization(const cv::Mat &rgbImage, const float alpha=dhe_alpha);

/*
 * Exposure Fusion (choice #4)
 */
cv::Mat exposureFusion(const cv::Mat &rgbImage);


// Testing only, need to used only internally
cv::Mat matlab_covar(const cv::Mat &A, const cv::Mat &B);
cv::Mat corrcoeff (const cv::Mat &M1, const cv::Mat &M2);


}		// namespace ice



