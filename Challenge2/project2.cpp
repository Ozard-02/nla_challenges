#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstring>
#include <fstream>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

MatrixXd random_Noise_Generator(const MatrixXd& image) {
  MatrixXd noise= MatrixXd::Random(image.rows(), image.cols());
  MatrixXd noise_Added_Image = (image + (50.0) * noise).unaryExpr([](double val) { return std::clamp(val, 0.0, 255.0); });

  return noise_Added_Image;
}

int main(int argc, char* argv[]){

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    // Task 1 //
    const char* input_image_path = argv[1];
    int width, height, channels;
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);

    if(image_data == NULL) {
        cerr << "Error: Could not load image." << endl;
        return 1;
    }

    MatrixXd A(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j);
            A(i, j) = static_cast<double>(image_data[index]);
        }
    }

    stbi_image_free(image_data);

    MatrixXd symmetric_mat = A.transpose() * A;

    cout << "symmetric_mat is symmetric? " << (symmetric_mat.isApprox(symmetric_mat.transpose(), 1.e-9) ? "True" : "False") << endl;
    cout << "Symmetric matrix norm: " << symmetric_mat.norm() << endl;


    // Task 2 //
    JacobiSVD<MatrixXd> svd(symmetric_mat, ComputeThinU | ComputeThinV);
    VectorXd singular_values = svd.singularValues(); //The eigenvalues are already sorted in decreasing order

    for (int i = 0; i < 2; i++) {
        cout << "Singular value " << i << ": " << singular_values(i) << endl;
    }

    // Task 3 //

    //export "symmetric_mat" in the matrix market format 
    string matrix_market_path = "symmetric_mat.mtx";
    saveMarket(symmetric_mat, matrix_market_path.c_str());

    //Not finished


    // Task 5 //
    
    JacobiSVD<MatrixXd> svd2(A, ComputeThinU | ComputeThinV);
    VectorXd singular_values2 = svd2.singularValues();

    MatrixXd U = svd2.matrixU();
    MatrixXd V = svd2.matrixV();
    MatrixXd S = U.transpose() * A * V;

    for(int i = 0; i < singular_values2.size(); i++) {
        cout << "Singular value " << i << ": " << singular_values2(i)<< " | Matrix value:" << S(i,i) << endl;
    }

    cout << "U columns: " << U.cols() << " U rows: " << U.rows() << endl;
    cout << "U is orthogonal? " << ((U.transpose()*U).isApprox(MatrixXd::Identity(U.cols(), U.cols()), 1.e-9) ? "True" : "False") << endl;
    cout << "V is orthogonal? " << ((V.transpose()*V).isApprox(MatrixXd::Identity(V.cols(), V.cols()), 1.e-9) ? "True" : "False") << endl;
    cout << "S is diagonal? " << (S.isDiagonal() ? "True" : "False") << endl;
    cout << "Norm of S:" << S.norm() << endl;


    // Task 8 //

    MatrixXd checkerboard = MatrixXd::Zero(200, 200);

    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            if (!((((i/25) %2 == 0) && ((j/25) %2 == 0) )||( ((i/25) %2 == 1) && ((j/25) %2 == 1)))) {
                checkerboard(i, j) = 255.0;
            }
        }
    }

    cout << checkerboard.norm() << endl;

    // Task 9 //

    MatrixXd noisy_checkerboard = random_Noise_Generator(checkerboard);

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_checkerboard_image = noisy_checkerboard.unaryExpr([](double val) -> unsigned char {
         return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const string output_noisy_checkerboard_image_path = "noisy_checkerboard.png";
    if (stbi_write_png(output_noisy_checkerboard_image_path.c_str(), 200, 200, 1, noisy_checkerboard_image.data(), 200) == 0) {
        cerr << "Error: Could not save checkerboard image" << endl;
        return 1;
    }

    cout << "checkerboard image saved to " << output_noisy_checkerboard_image_path << endl;

    // Task 10 // 

    JacobiSVD<MatrixXd> svd3(noisy_checkerboard, ComputeThinU | ComputeThinV);
    VectorXd singular_values_checkerboard = svd3.singularValues(); //The eigenvalues are already sorted in decreasing order

    for (int i = 0; i < 2; i++) {
        cout << "Singular value " << i << ": " << singular_values_checkerboard(i) << endl;
    }
    return 0;
}