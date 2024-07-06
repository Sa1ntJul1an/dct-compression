#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){

    string image_path = "../../engine_darkened.jpg";
    Mat image = imread(image_path, IMREAD_COLOR);
    imshow("Test", image);
    waitKey(0);

    return 0;
}
