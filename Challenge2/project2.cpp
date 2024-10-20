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

    cout<<"Maximum eigenvalue from lis: 1.045818e+09. Is equal to the previous value? "<<((singular_values(0)-1.045818e+09)/((singular_values(0)+1.045818e+09)/2)<1e-8? "True" :"False")<<endl;

    // Task 4 //
    
    //Non trovo uno shift che accellera lol

    // Task 5 //
    
    JacobiSVD<MatrixXd> svd2(A, ComputeFullU | ComputeFullV);
    
    VectorXd singular_values2 = svd2.singularValues();

    MatrixXd U = svd2.matrixU();
    MatrixXd V = svd2.matrixV();
    MatrixXd S = U.transpose() * A * V;

    // for(int i = 0; i < singular_values2.size(); i++) {
    //     cout << "Singular value " << i << ": " << singular_values2(i)<< " | Matrix value:" << S(i,i) << endl;
    // }
    cout << "A rows: " << height << " | A columns: " << width << endl;
    cout << "U rows: " << U.rows() << " | U columns: " << U.cols() << endl;
    cout << "V rows: " << V.rows() << " | V columns: " << V.cols() << endl;
    cout << "S rows: " << S.rows() << " | S columns: " << S.cols() << endl;
    cout << "U is orthogonal? " << ((U.transpose()*U).isApprox(MatrixXd::Identity(U.cols(), U.cols()), 1.e-8) ? "True" : "False") << endl;
    cout << "V is orthogonal? " << ((V.transpose()*V).isApprox(MatrixXd::Identity(V.cols(), V.cols()), 1.e-8) ? "True" : "False") << endl;
    bool diagonal=true;
    int i=0;
    int j=0;
    while(diagonal&&i<S.rows()&&j<S.cols()){
        if(abs(S(i , j))>1.e-8&&i!=j) diagonal=false; // Print each element
        i++;j++;
    }
    cout << "S is diagonal? " << (diagonal ? "True" : "False") << endl;
    cout << "Norm of S: " << S.norm() << endl;

    // Task 6 //

    MatrixXd C_40 = U.block(0,0, U.rows(), 40);
    MatrixXd C_80 = U.block(0,0, U.rows(), 80);
    MatrixXd D_40(V.rows(), 40);
    MatrixXd D_80(V.rows(), 80);

    cout << "C_40 has " << C_40.rows() << "x"<<C_40.cols() << endl;
    cout << "C_80 has " << C_80.rows() << "x"<<C_80.cols() << endl;
    cout << "D_40 has " << D_40.rows() << "x"<<D_40.cols() << endl;
    cout << "D_80 has " << D_80.rows() << "x"<<D_80.cols() << endl;
    cout << "Number of singular values=" << singular_values2.size() << endl;

    for(int i=0; i<D_80.rows(); i++){
        for(int j=0; j<D_80.cols(); j++){
            // cout<<"i="<<i<<", j="<<j<<endl;
            // cout<<V(i,j)<<endl;
            // cout<<D_80(i,j)<<endl;
            // cout<<singular_values2(j)<<endl;
            if(j<D_40.cols()){
                // cout<<D_40(i,j)<<endl;
                D_40(i, j)=V(i,j)* singular_values2(j);
            }
            D_80(i, j)=V(i, j)*singular_values2(j);
        }
    }

    //compute non zero entries taking care of the machine precision
    int non_zero_entries_C_40 = 0;
    int non_zero_entries_C_80 = 0;
    int non_zero_entries_D_40 = 0;
    int non_zero_entries_D_80 = 0;
    for(int i = 0; i < C_40.rows(); i++) {
        for(int j = 0; j < C_40.cols(); j++) {
            if(C_40(i,j) > 1.e-9) {
                non_zero_entries_C_40++;
            }
        }
    }
    for(int i = 0; i < C_80.rows(); i++) {
        for(int j = 0; j < C_80.cols(); j++) {
            if(C_80(i,j) > 1.e-9) {
                non_zero_entries_C_80++;
            }
        }
    }
    for(int i = 0; i < D_40.rows(); i++) {
        for(int j = 0; j < D_40.cols(); j++) {
            if(D_40(i,j) > 1.e-9) {
                non_zero_entries_D_40++;
            }
        }
    }
    for(int i = 0; i < D_80.rows(); i++) {
        for(int j = 0; j < D_80.cols(); j++) {
            if(D_80(i,j) > 1.e-9) {
                non_zero_entries_D_80++;
            }
        }
    }
    
    //print results
    cout << "C_40 has "<<C_40.rows()<<"x"<<C_40.cols()<<" and has non zero entries in C_40: " << non_zero_entries_C_40 << endl;
    cout << "C_80 has "<<C_80.rows()<<"x"<<C_80.cols()<<" and has non zero entries in C_80: " << non_zero_entries_C_80 << endl;
    cout << "D_40 has "<<D_40.rows()<<"x"<<D_40.cols()<<" and has non zero entries in D_40: " << non_zero_entries_D_40 << endl;
    cout << "D_80 has "<<D_80.rows()<<"x"<<D_80.cols()<<" and has non zero entries in D_80: " << non_zero_entries_D_80 << endl;


    // Task 7 //

    MatrixXd A_tilde_40=C_40*D_40.transpose();
    MatrixXd A_tilde_80=C_80*D_80.transpose();

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> A_40_image = A_tilde_40.unaryExpr([](double val) -> unsigned char {
         return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const string output_A_40_image_path = "A_40_image.png";
    if (stbi_write_png(output_A_40_image_path.c_str(), A_tilde_40.cols(), A_tilde_40.rows(), 1, A_40_image.data(), A_tilde_40.cols()) == 0) {
        cerr << "Error: Could not save A_40 image" << endl;
        return 1;
    }

    cout << "A_40 image saved to " << output_A_40_image_path << endl;

     Matrix<unsigned char, Dynamic, Dynamic, RowMajor> A_80_image = A_tilde_80.unaryExpr([](double val) -> unsigned char {
         return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const string output_A_80_image_path = "A_80_image.png";
    if (stbi_write_png(output_A_80_image_path.c_str(), A_tilde_80.cols(), A_tilde_80.rows(), 1, A_80_image.data(), A_tilde_40.cols()) == A_tilde_80.cols()) {
        cerr << "Error: Could not save A_80 image" << endl;
        return 1;
    }

    cout << "A_80 image saved to " << output_A_80_image_path << endl;   

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