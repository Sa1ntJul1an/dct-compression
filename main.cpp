#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <vector>

using namespace std;
using namespace cv;


float alpha_p(int M, int p){
    if(p == 0){
        return 1 / sqrt(float(M));
    } else {
        return sqrt(2 / float(M));
    }
}


float alpha_q(int N, int q){
    if(q == 0){
        return 1 / sqrt(float(N));
    } else {
        return sqrt(2 / (float(N)));
    }
}


map<vector<int>, vector<vector<float>>> compute_basis_functions(int M, int N){
    float pi = M_PI;

    map<vector<int>, vector<vector<float>>> basis_function_map;

    // for each frequency in x and y
    for (int p = 0; p < M; p++){
        for (int q = 0; q < N; q++){

            // key to basis function
            vector<int> key = {p, q};

            vector<vector<float>> basis_function;
            // for each pixel in x and y
            for (int x = 0; x < M; x++){

                vector<float> column;

                for (int y = 0; y < N; y++){
                    float A = (pi * (2*x + 1) * p) / (2 * M);
                    float B = (pi * (2*y + 1) * q) / (2 * N);

                    float intensity = alpha_p(M, p) * alpha_q(N, q) * cos(A) * cos(B);

                    column.push_back(intensity);
                }
                basis_function.push_back(column);
            }

            // add basis function to map
            basis_function_map[key] = basis_function;
        }
    }

    return basis_function_map;
}


Mat create_basis_func_image(map<vector<int>, vector<vector<float>>> basis_func_map, int blocksize_x, int blocksize_y, Size image_size){
    // display basis functions
    Mat basis_functions_image = Mat::zeros(Size(pow(blocksize_y, 2) + blocksize_y - 1, pow(blocksize_x, 2) + blocksize_x - 1), CV_8UC1);

    float dct_min = 0;
    float dct_max = 0;

    int block_index_x = 0;
    for (int x = 0; x < blocksize_x; x++){
        int block_index_y = 0;
        for (int y = 0; y < blocksize_y; y++){
            vector<int> key = {x, y};
            vector<vector<float>> basis_func = basis_func_map[key];

            for (int col = 0; col < basis_func.size(); col++){
                for (int row = 0; row < basis_func.at(0).size(); row++){
                    float basis_func_val = basis_func.at(col).at(row);

                    if (basis_func_val < dct_min){
                        dct_min = basis_func_val;
                    } else if (basis_func_val > dct_max) {
                        dct_max = basis_func_val;
                    }
                }
            }
            block_index_y += blocksize_y + 1;
        }
        block_index_x += blocksize_x + 1;
    }

    float dct_func_range = dct_max - dct_min;
    float func_val_map_range = 255;

    block_index_x = 0;
    for (int x = 0; x < blocksize_x; x++){
        int block_index_y = 0;
        for (int y = 0; y < blocksize_y; y++){
            vector<int> key = {x, y};
            vector<vector<float>> basis_func = basis_func_map[key];

            for (int col = 0; col < basis_func.size(); col++){
                for (int row = 0; row < basis_func.at(0).size(); row++){
                    float basis_func_val = basis_func.at(col).at(row);

                    float grayscale_intensity = (basis_func_val - dct_min) * func_val_map_range / dct_func_range;

                    basis_functions_image.at<uchar>(block_index_x + col, block_index_y + row) = grayscale_intensity;
                }
            }
            block_index_y += blocksize_y + 1;
        }
        block_index_x += blocksize_x + 1;
    }

    resize(basis_functions_image, basis_functions_image, image_size);

    return basis_functions_image;
}


map<vector<int>, Mat> get_dct_cofficients(Mat& image, map<vector<int>, vector<vector<float>>> basis_func_map, int blocksize_x, int blocksize_y){
    int height = image.rows;
    int width = image.cols;

    map<vector<int>, Mat> dct_coefficients;

    int num_blocks_in_x = width / blocksize_x;
    int num_blocks_in_y = height / blocksize_y;

    int x_index = 0;
    int y_index = 0;
    for (int block_x = 0; block_x < num_blocks_in_x; block_x ++){
        for (int block_y = 0; block_y < num_blocks_in_y; block_y ++){

            Mat coefficient_block = Mat(blocksize_x, blocksize_x, CV_32F);

            Mat image_block = image(Rect(x_index, y_index, blocksize_x, blocksize_y));

            for (int p = 0; p < blocksize_x; p++) {
                for (int q = 0; q < blocksize_y; q++) {

                    vector<int> key = {p, q};
                    vector<vector<float>> basis_func = basis_func_map[key];

                    for (int col = 0; col < blocksize_x; col ++) {
                        for (int row = 0; row < blocksize_y; row++) {
                            coefficient_block.at<float>(col, row) += image_block.at<uchar>(col, row) * basis_func[col][row];
                        }
                    }
                }
            }
            // add dct block for this block of the image to the dct coeff map
            dct_coefficients[{block_x, block_y}] = coefficient_block;

            y_index += blocksize_y;
        }
        y_index = 0;
        x_index += blocksize_x;
    }

    return dct_coefficients;
}


Mat inverse_dct(map<vector<int>, Mat> coefficients_map, map<vector<int>, vector<vector<float>>> basis_function_map, int blocksize_x, int blocksize_y, int imagesize_x, int imagesize_y) {
    int image_height = imagesize_y;
    int image_width = imagesize_x;

    Mat image_out = Mat::zeros(Size(image_width, image_height), CV_8UC1);

    int blocks_in_x = image_width / blocksize_x;
    int blocks_in_y = image_height / blocksize_y;

    int image_block_count = 0;

    int x_index = 0;
    int y_index = 0;
    for (int block_x = 0; block_x < blocks_in_x; block_x++) {
        for (int block_y = 0; block_y < blocks_in_y; block_y++) {
            // for each image block in the final image

            // get the coefficients matrix at for this image block 
            Mat coefficient_block = coefficients_map[{block_x, block_y}];

            // image block constructed with linear combination of dct basis functions and coefficients matrix 
            Mat image_block = Mat::zeros(Size(blocksize_x, blocksize_y), CV_8UC1);

            for (int p = 0; p < blocksize_x; p++){
                for (int q = 0; q < blocksize_y; q++){
                    // for each basis function {p, q}

                    // get dct coefficient for each frequency
                    float dct_coefficient = coefficient_block.at<float>(p, q);

                    // get basis func
                    vector<vector<float>> basis_func = basis_function_map[{p, q}];

                    //matrix1.mul(dct_coefficient, matrix1, CV_MAT_MUL_INPLACE);

                    for (int image_block_index_y = 0; image_block_index_y < blocksize_y; image_block_index_y++){
                        for (int image_block_index_x = 0; image_block_index_x < blocksize_x; image_block_index_x++){
                            // for each pixel in image block 

                            // get value of basis func at pixel location
                            float basis_func_val = basis_func[image_block_index_x][image_block_index_y];

                            image_block.at<uchar>(image_block_index_x, image_block_index_y) += basis_func_val * dct_coefficient;
                        }
                    }
                }
            }
            image_block_count ++;
            image_block.copyTo(image_out(Rect(x_index, y_index, image_block.cols, image_block.rows)));

            y_index += blocksize_y;
        }

        imshow("Reconstructing image...", image_out);
        waitKey(1);

        y_index = 0;
        x_index += blocksize_x;
    }

    destroyAllWindows();

    return image_out;
}


int main(){

    string image_path = "lenna.png";
    Mat image = imread(image_path, IMREAD_COLOR);

    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    int dct_blocksize_x = 8;
    int dct_blocksize_y = 8;
    float max_rows = 1000;

    // downsize image to fit in screen
    float scale_factor = 1.0;
    if (image.rows > max_rows){
        scale_factor = max_rows / image.rows;
    }

    // create image sizes in x and y that are whole number multiples of blocksize
    int image_size_x = int(image.cols * scale_factor) - int(image.cols * scale_factor) % dct_blocksize_x;
    int image_size_y = int(image.rows * scale_factor) - int(image.rows * scale_factor) % dct_blocksize_y;

    Size image_size(image_size_x, image_size_y);

    resize(grayscale, grayscale, image_size);

    // get basis functions for 8x8 block size
    map<vector<int>, vector<vector<float>>> basis_function_map = compute_basis_functions(dct_blocksize_x, dct_blocksize_y);

    Mat basis_functions_image = create_basis_func_image(basis_function_map, dct_blocksize_x, dct_blocksize_y, image_size);

    map<vector<int>, Mat> coefficients_map = get_dct_cofficients(grayscale, basis_function_map, dct_blocksize_x, dct_blocksize_y);

    Mat reconstructed_image = inverse_dct(coefficients_map, basis_function_map, dct_blocksize_x, dct_blocksize_y, image_size_x, image_size_y);

    imshow("Original Image, Grayscale", grayscale);
    imshow("Basis Functions", basis_functions_image);
    imshow("Reconstructed Image", reconstructed_image);
    //imshow("Original", image);
    waitKey(0);

    return 0;
}
