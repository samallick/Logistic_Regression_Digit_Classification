#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "sigmoid.hpp"

using namespace cv;

void populate_X(Mat &X, int num_digits) {
    for (int i=0; i<num_digits; i++) {
        std::string folder = std::to_string(i);
        std::string url = "../../test_data/" + folder + "/" + folder + "_" + "0" + ".png"; 
        url = samples::findFile(url);
        Mat img = imread(url, IMREAD_GRAYSCALE);
        img.convertTo(img, CV_64FC1);
        img = img.reshape(1,1);
        X.push_back(img);
    }
}

int main() {
    int num_digits = 10;
    int num_images = 10;
    int num_rows_pixels = 100;
    int num_cols_pixels = 100;
    int num_pixels = num_rows_pixels * num_cols_pixels;

    Mat X = Mat(0, num_pixels, CV_64FC1);
    populate_X(X, num_digits);

    Mat theta;
    FileStorage fs("../../learn/build/classifier.txt", FileStorage::READ);
    fs["classifier"] >> theta;

    Mat prob_matrix;
    for (int i=0; i< num_images; i++) {
        int most_likely = 0;
        double most_likely_prob = 0;
        for (int j=0; j<num_digits; j++) {
            // It gives 1x1 Mat, convert to double.
            Mat prob = sigmoid(X.row(i) * theta.row(j).t());
            double prob_d = prob.at<double>(0,0);
            if (prob_d > most_likely_prob) {
                most_likely = j;
                most_likely_prob = prob_d;
            }
        }
        prob_matrix.push_back(most_likely);
    }

    for (int i=0; i< num_images; i++) {
        Mat img = X.row(i);
        img = img.reshape(1,100);
        //img.convertTo(img, CV_8UC1);
        int guess = prob_matrix.at<int>(i,0);
        std::string guess_str = std::to_string(guess);
        std::string window_text = "DRAWING = " + std::to_string(i) + "    -    GUESS = " + guess_str;
        putText(img, guess_str, Point(2, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 2);
        namedWindow(window_text, WINDOW_NORMAL);
        resizeWindow(window_text, 500, 500);
        imshow(window_text, img);
        waitKey(0);
    }

    return 0;
}
