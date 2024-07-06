#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    
    string image_path = "../../engine_darkened.jpg";
    Mat image = imread(image_path, IMREAD_COLOR);

    Size image_size(500, 500);

    resize(image, image, image_size);

    cout << image.size << endl;

    imshow("Image", image);
    waitKey(0);

    return 0;
}
