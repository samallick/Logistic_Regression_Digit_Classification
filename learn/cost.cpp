#include "cost.hpp"
#include "sigmoid.hpp"
#include <iostream>
#include <typeinfo>

using namespace cv;

// Pseudo code of the equation for cost. Broken down in function.
// double J = (1/m) * ((-y.t() * log(hyp)) - ((1-y).t() * log(1-hyp)));
double cost(Mat theta, Mat X, Mat y, double lmda) {
    double m = y.rows;

    //Mat z = X * theta.t();
    Mat hyp = sigmoid(X * theta.t()); 

    // Calculate log(hyp) element-wise.
    Mat log_hyp;
    log(hyp, log_hyp);

    // Calculate log(1-hyp element-wise).
    Mat log_hyp_2;
    log(1 - hyp, log_hyp_2);

    // Calculate LHS and RHS of subtraction then
    // extract double from 1x1 Mat.
    Mat LHS = -y.t() * log_hyp;
    double LHS_d = LHS.at<double>(0,0);
    Mat RHS = (1-y).t() * log_hyp_2;
    double RHS_d = RHS.at<double>(0,0);


    // Calculate the cost.
    double J = (1/m) * (LHS_d - RHS_d);

    // Apply regularization (but not to theta[0,0]).
    Mat temp_theta = theta.clone();
    temp_theta.at<double>(0,0) = 0;
    double sum_theta = 0;
    for (int i=0; i<temp_theta.rows; i++) {
        for (int j=0; j<temp_theta.cols; j++) {
            double px = temp_theta.at<double>(i,j);
            sum_theta += px * px;
        }
    }
    double reg = (lmda/(2*m) + sum_theta);

    return J + reg;
}
