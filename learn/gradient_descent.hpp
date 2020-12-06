#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <opencv2/core.hpp>

cv::Mat gradient_descent(cv::Mat theta, cv::Mat X, cv::Mat y, double alpha, double lmda);

#endif 
