#include "sigmoid.hpp"
#include <iostream>

using namespace cv;

//------------------------------------------------------------------
// Computes the sigmoid element-wise.
//------------------------------------------------------------------
Mat sigmoid(Mat z) {
    for (int i=0; i<z.rows; i++) {
        for (int j=0; j<z.cols; j++) {
            double &p = z.at<double>(i,j);
            p = 1 / (1 + exp(-p));
        }
    }
    return z;
}
