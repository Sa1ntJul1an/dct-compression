#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <vector>

using namespace std;
using namespace cv;


double alpha_p(int M, int p){
    if(p == 0){
        return 1 / sqrt(M);
    } else{
        return sqrt(2 / M);
    }
}

double alpha_q(int N, int q){
    if(q == 0){
        return 1 / sqrt(N);
    } else{
        return sqrt(2 / N);
    }
}

map<vector<int>, vector<vector<double>>> compute_basis_functions(int M, int N){
    double pi = M_PI;

    map<vector<int>, vector<vector<double>>> basis_function_map;

    // for each frequency in x and y
    for (int p = 0; p < M; p++){
        for (int q = 0; q < N; q++){
            
            // key to basis function
            vector<int> key = {p, q};

            vector<vector<double>> basis_function;
            // for each pixel in x and y
            for (int x = 0; x < M; x++){

                vector<double> column;

                for (int y = 0; y < N; y++){
                    double A = (pi * (2*x + 1) * p) / 2 * M;
                    double B = (pi * (2*y + 1) * q) / 2 * N;

                    double intensity = alpha_p(M, p) * alpha_q(N, q) * cos(A) * cos(B);

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

vector<vector<double>> get_dct_cofficients(Mat& image, int blocksize){
    int M = image.rows;
    int N = image.cols;

    vector<vector<double>> coefficients;

    // initialize empty dct matrix of 0's
    for (int x = 0; x < M; x++){
        vector<double> column;
        for (int y = 0; y < N; y++){
            column.push_back(0.0);
        }
        coefficients.push_back(column);
    }

    // number of blocks in x and y
    int blocks_in_x = M / blocksize;
    int blocks_in_y = N / blocksize;

    // x and y locations in original image where block begins (top left corner)
    int block_index_x = 0;
    int block_index_y = 0;

    // loop over each block in image based on provided block size
    for(int block_x = 0; block_x < blocks_in_x; block_x++){
        for(int block_y = 0; block_y < blocks_in_y; block_y++){

                for (int p = 0; p < blocksize; p++){
                    for (int q = 0; q < blocksize; q++){
                        //within each block, loop over x and y pixels
                        for(int x = block_index_x; x < block_index_x + blocksize; x++){
                            for(int y = block_index_y; y < block_index_y + blocksize; y++){
                                // if we are out of range of the image, set to 0
                                int intensity = 0;
                                if (x < M - 1){
                                    intensity = int(image.at<uchar>(y, x));
                                }
                                
                            }
                        }
                    }
                }
                
            block_index_x += blocksize;
        }
        block_index_y += blocksize;
    }

    /*
    for (int p = 0; p < M; p++){
        vector<double> column;
        for (int q = 0; q < N; q++){
            vector<int> index = {p, q, 0};          // bgr value at x and y index

            // advance to beginning of next block
            block_index_x += blocksize;
        }

        block_index_y += blocksize;
    }
    */

    return coefficients;
}


int main(){
    
    string image_path = "lenna.png";
    Mat image = imread(image_path, IMREAD_COLOR);

    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    int dct_blocksize = 8;
    double max_rows = 1000;
    
    // downsize image to fit in screen
    double scale_factor = 1.0;
    if (image.rows > max_rows){
        scale_factor = max_rows / image.rows;
    }

    Size image_size(image.cols * scale_factor, image.rows * scale_factor);

    resize(grayscale, grayscale, image_size);

    // get basis functions for 8x8 block size
    map<vector<int>, vector<vector<double>>> basis_function_map = compute_basis_functions(dct_blocksize, dct_blocksize);

    for (int x = 0; x < dct_blocksize; x++){
        for (int y = 0; y < dct_blocksize; y++){
            vector<int> key = {x, y};
            vector<vector<double>> basis_func = basis_function_map[key];

            for (int row = 0; row < basis_func.size(); row++){
                for (int col = 0; col < basis_func.at(0).size(); col++){
                    cout << basis_func.at(row).at(col) << " ";
                }
                cout << endl;
            }
        }
    }


    //imshow("Original", image);
    //imshow("Grayscale", grayscale);
    //waitKey(0);

    return 0;
}
