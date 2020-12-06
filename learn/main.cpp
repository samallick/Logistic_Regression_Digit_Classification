#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "cost.hpp"
#include "gradient_descent.hpp"
#include "sigmoid.hpp"

using namespace cv;

//------------------------------------------------------------------
// NOTES
//------------------------------------------------------------------

// CV_8UC1 is the type of Matrix created import using GRAYSCALE.
// Means 8 bits, unsigned (0 - 255), one channel only for grayscale.

// We will convert all of our Matrixes to CV_64FC1 so that the 
// values are of type double and we are able to do math with them.
// Ex. can only get dot product of Mats with same type.

// To index into CV_64FC1 Mat use the following.
// unsigned char P = img.at<double>(0,0);


//------------------------------------------------------------------
// POPULATE TRAINING SET X WITH IMAGE MATRICES FLATTENED INTO ROWS.
// SIZE OF X IS 1000 ROWS (IMAGES), 10000 COLS (PIXELS).
//------------------------------------------------------------------
void populate_X(Mat &X, int num_versions, int num_digits) {
    for (int i=0; i<num_digits; i++) {
        for (int j=0; j<num_versions; j++) {
            // Craft url string for each image.
            std::string folder = std::to_string(i);
            std::string version = std::to_string(j);
            std::string url = "../../training_data/" + folder + "/" + folder + "_" + version + ".png";
            // Convert image to opencv Mat.
            url = samples::findFile(url);
            Mat img = imread(url, IMREAD_GRAYSCALE);
            // Convert CV_8UC1 to CV_64FC1 so we can do math.
            img.convertTo(img, CV_64FC1);
            // Flatten out image Mat.
            img = img.reshape(1,1);
            // Push image row onto training example set.
            X.push_back(img);
        }
    }
}


//------------------------------------------------------------------
// POPULATE TRAINING SET y WITH LABELS 0 OR 1.
// EACH ROW IS ASSOCIATED WITH AN IMAGE.
// EACH COLUMN IS ASSOCIATED WITH ONE DIGIT (EX COL 0 FOR DIGIT 0).
// EX. IMG ASSOCIATED WITH ROW 0 IS A DRAWING OF DIGIT 0,
//   ROW 0, COL 0 y=1
//   ROW 0, COL 1-9 y=0
// SO 1 == MATCH, 0 == NOT MATCH.
// IMAGES IN X ARE 100 rows of 0, 100 rows of 1, etc.
// INITIAL y IS ALL ZEROS, FILL IN THE ONES.
// 1000 ROWS (IMAGES), 10 COLUMNS (1 FOR EACH DIGIT 0-9).
//------------------------------------------------------------------
void populate_y(Mat &y, int num_versions, int num_digits) {
    for (int i=0; i<num_digits; i++) {
        for (int j=0; j<num_versions; j++) {
            int row_idx = i * num_versions + j;
            int col_idx = i;
            y.at<double>(row_idx, col_idx) = 1;
        }
    }
}


//------------------------------------------------------------------
// MAIN.
//------------------------------------------------------------------
int main() {
    // Define test data quantities so can be changed once here.
    int num_digits = 10;
    int num_images = 1000;
    int num_versions = num_images/10;
    int num_rows_pixels = 100;
    int num_cols_pixels = 100;
    int num_pixels = num_rows_pixels * num_cols_pixels;
    

    // Define learning rate alpha and regularization value lmda.
    double alpha = 0.00000001;
    double lmda = 3;

    // Create empty training set X then populate it with training images.
    Mat X = Mat(0, num_pixels, CV_64FC1);
    populate_X(X, num_versions, num_digits);

    // Create label set y of zeros then populate it with training labels.
    Mat y = Mat::zeros(num_images, num_digits, CV_64FC1);
    populate_y(y, num_versions, num_digits);


    // Create parameter set theta initialized with zeros.
    // One row for classifying each digit.
    // One column (theta_i) for each feature (pixel in an image).
    Mat theta = Mat::zeros(num_digits, num_pixels, CV_64FC1);



    // Train the classifiers (rows of theta).
    for (int i=0; i<num_digits; i++) {
        Mat theta_row = theta.row(i).clone();
        Mat y_col = y.col(i).clone();
        double J = 1;
        while (J > 0) {
            theta_row = gradient_descent(theta_row, X, y_col, alpha, lmda);
            J = cost(theta_row, X, y_col, lmda);
            std::cout << J << "   " << i << std::endl;
        }
        // Copy completed new theta set for an image to theta.
        for (int j=0; j<theta.cols; j++) {
            theta.at<double>(i,j) = theta_row.at<double>(0,j);
        }
    }

    // Save theta for use in making predictions.
    FileStorage fs("classifier.txt", FileStorage::WRITE);
    fs << "classifier" << theta;

    return 0;
}


