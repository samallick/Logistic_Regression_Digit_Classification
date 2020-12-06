#ifndef COST_H
#define COST_H

#include <opencv2/core.hpp>

double cost(cv::Mat theta, cv::Mat X, cv::Mat y, double lmda);

#endif
