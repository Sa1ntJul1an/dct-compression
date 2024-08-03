#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;




vector<vector<double>> get_dct_cofficients(Mat image, int blocksize){
    int M = image.rows;
    int N = image.cols;

    double alpha_p;
    double alpha_q;

    vector<vector<double>> coefficients;

    for (int p = 0; p < M; p++){
        vector<double> column;
        for (int q = 0; q < N; q++){
            vector<int> index = {p, q, 0};          // bgr value at x and y index
            cout << image.at<Vec3b>(q, p);
        }
    }

    return coefficients;
}


int main(){
    
    string image_path = "lenna.png";
    Mat image = imread(image_path, IMREAD_COLOR);

    double max_rows = 1000;
    
    // downsize image to fit in screen
    double scale_factor = 1.0;
    if (image.rows > max_rows){
        scale_factor = max_rows / image.rows;
    }

    Size image_size(image.cols * scale_factor, image.rows * scale_factor);

    resize(image, image, image_size);

    cout << image.size << endl;

    vector<vector<double>> dct_coefficients = get_dct_cofficients(image, 8);

    imshow("Image", image);
    waitKey(0);

    return 0;
}
