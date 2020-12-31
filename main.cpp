/*
 * main.cpp
 *
 *  Created on: Jun 25, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/hdf.hpp>
#include <boost/filesystem.hpp>
#include "mpi.h"
#include "im_enhance.h"
#include "npy.hpp"
#include "src/matutils.h"
#include "src/timer.h"


using namespace std;
namespace fs=boost::filesystem;



int main(int argc, char *argv[])
{
//	ice::Matf3 I5=ice::Matf3::ones(5,5);
	cv::Mat I5 = cv::Mat::ones(5,5,CV_32FC(3));
//	I5 = I5 - cv::Vec3f(1,1,1);
	cout << I5 << endl;

	return 0;
}
