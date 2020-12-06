#include "gradient_descent.hpp"
#include "sigmoid.hpp"
#include <iostream>

using namespace cv;

Mat gradient_descent(Mat theta, Mat X, Mat y, double alpha, double lmda) {
    double m = y.rows;
    Mat hyp = sigmoid(X * theta.t());

    // Don't apply regularization to theta[0,0].
    Mat temp_theta = theta.clone();
    temp_theta.at<double>(0,0) = 0;

    theta = theta.t() - (alpha/m) * X.t() * (hyp - y) + (lmda / m * temp_theta.t());

    theta = theta.t();


    return theta;
}
