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
    cout << "Non zero entries in C_40: " << non_zero_entries_C_40 << endl;
    cout << "Non zero entries in C_80: " << non_zero_entries_C_80 << endl;
    cout << "Non zero entries in D_40: " << non_zero_entries_D_40 << endl;
    cout << "Non zero entries in D_80: " << non_zero_entries_D_80 << endl;


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

    cout << "Norm of checkerboard matrix: " <<checkerboard.norm() << endl;

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

    cout << "checkerboard noisy image saved to " << output_noisy_checkerboard_image_path << endl;

    // Task 10 // 

    JacobiSVD<MatrixXd> svd_checkboard(noisy_checkerboard, ComputeFullU | ComputeFullV);
    VectorXd singular_values_checkerboard = svd_checkboard.singularValues(); //The eigenvalues are already sorted in decreasing order

    for (int i = 0; i < 2; i++) {
        cout << "Singular value " << i << ": " << singular_values_checkerboard(i) << endl;
    }

    // Task 11 // 

    MatrixXd U_checkerboard = svd_checkboard.matrixU();
    MatrixXd V_checkerboard = svd_checkboard.matrixV();
    MatrixXd S_checkerboard = U_checkerboard.transpose() * noisy_checkerboard * V_checkerboard;


    // for(int i = 0; i < singular_values2.size(); i++) {
    //     cout << "Singular value " << i << ": " << singular_values2(i)<< " | Matrix value:" << S(i,i) << endl;
    // }
    cout << "Checkerboard rows: " << checkerboard.rows() << " | A columns: " << checkerboard.cols() << endl;
    cout << "U rows: " << U_checkerboard.rows() << " | U columns: " << U_checkerboard.cols() << endl;
    cout << "V rows: " << V_checkerboard.rows() << " | V columns: " << V_checkerboard.cols() << endl;
    cout << "S rows: " << S_checkerboard.rows() << " | S columns: " << S_checkerboard.cols() << endl;
    cout << "U is orthogonal? " << ((U_checkerboard.transpose()*U_checkerboard).isApprox(MatrixXd::Identity(U_checkerboard.cols(), U_checkerboard.cols()), 1.e-8) ? "True" : "False") << endl;
    cout << "V is orthogonal? " << ((V_checkerboard.transpose()*V_checkerboard).isApprox(MatrixXd::Identity(V_checkerboard.cols(), V_checkerboard.cols()), 1.e-8) ? "True" : "False") << endl;
    diagonal=true;
    i=0;
    j=0;
    while(diagonal&&i<S_checkerboard.rows()&&j<S_checkerboard.cols()){
        if(abs(S_checkerboard(i , j))>1.e-8&&i!=j) diagonal=false; // Print each element
        i++;j++;
    }
    cout << "S is diagonal? " << (diagonal ? "True" : "False") << endl;
    cout << "Norm of S: " << S_checkerboard.norm() << endl;

    MatrixXd C_checkerboard_5 = U_checkerboard.block(0,0, U_checkerboard.rows(), 5);
    MatrixXd C_checkerboard_10 = U_checkerboard.block(0,0, U_checkerboard.rows(), 10);
    MatrixXd D_checkerboard_5(V_checkerboard.rows(), 5);
    MatrixXd D_checkerboard_10(V_checkerboard.rows(), 10);

    cout << "C_checkerboard_5 has " << C_checkerboard_5.rows() << "x"<<C_checkerboard_5.cols() << endl;
    cout << "C_checkerboard_10 has " << C_checkerboard_10.rows() << "x"<<C_checkerboard_10.cols() << endl;
    cout << "D_checkerboard_5 has " << D_checkerboard_5.rows() << "x"<<D_checkerboard_5.cols() << endl;
    cout << "D_checkerboard_10 has " << D_checkerboard_10.rows() << "x"<<D_checkerboard_10.cols() << endl;
    cout << "Number of singular values=" << singular_values_checkerboard.size() << endl;

    for(int i=0; i<D_checkerboard_10.rows(); i++){
        for(int j=0; j<D_checkerboard_10.cols(); j++){
            // cout<<"i="<<i<<", j="<<j<<endl;
            // cout<<V(i,j)<<endl;
            // cout<<D_80(i,j)<<endl;
            // cout<<singular_values2(j)<<endl;
            if(j<D_checkerboard_5.cols()){
                // cout<<D_40(i,j)<<endl;
                D_checkerboard_5(i, j)=V_checkerboard(i,j)* singular_values_checkerboard(j);
            }
            D_checkerboard_10(i, j)=V_checkerboard(i, j)*singular_values_checkerboard(j);
        }
    }

    //compute non zero entries taking care of the machine precision
    int non_zero_entries_C_checkerboard_5 = 0;
    int non_zero_entries_C_checkerboard_10 = 0;
    int non_zero_entries_D_checkerboard_5 = 0;
    int non_zero_entries_D_checkerboard_10 = 0;
    for(int i = 0; i < C_checkerboard_5.rows(); i++) {
        for(int j = 0; j < C_checkerboard_5.cols(); j++) {
            if(C_checkerboard_5(i,j) > 1.e-9) {
                non_zero_entries_C_checkerboard_5++;
            }
        }
    }
    for(int i = 0; i < C_checkerboard_10.rows(); i++) {
        for(int j = 0; j < C_checkerboard_10.cols(); j++) {
            if(C_checkerboard_10(i,j) > 1.e-9) {
                non_zero_entries_C_checkerboard_10++;
            }
        }
    }
    for(int i = 0; i < D_checkerboard_5.rows(); i++) {
        for(int j = 0; j < D_checkerboard_5.cols(); j++) {
            if(D_checkerboard_5(i,j) > 1.e-9) {
                non_zero_entries_D_checkerboard_5++;
            }
        }
    }
    for(int i = 0; i < D_checkerboard_10.rows(); i++) {
        for(int j = 0; j < D_checkerboard_10.cols(); j++) {
            if(D_checkerboard_10(i,j) > 1.e-9) {
                non_zero_entries_D_checkerboard_10++;
            }
        }
    }

    //print results
    cout << "Non zero entries in C_checkerboard_5: " << non_zero_entries_C_checkerboard_5 << endl;
    cout << "Non zero entries in C_checkerboard_10: " << non_zero_entries_C_checkerboard_10 << endl;
    cout << "Non zero entries in D_checkerboard_5: " << non_zero_entries_D_checkerboard_5 << endl;
    cout << "Non zero entries in D_checkerboard_10: " << non_zero_entries_D_checkerboard_10 << endl;

    // Task 12 // 

    MatrixXd checkerboard_tilde_5=C_checkerboard_5*D_checkerboard_5.transpose();
    MatrixXd checkerboard_tilde_10=C_checkerboard_10*D_checkerboard_10.transpose();

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> checkerboard_tilde_5_image = checkerboard_tilde_5.unaryExpr([](double val) -> unsigned char {
         return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const string output_checkerboard_tilde_5_image_path = "checkerboard_tilde_5_image.png";
    if (stbi_write_png(output_checkerboard_tilde_5_image_path.c_str(), checkerboard_tilde_5_image.cols(), checkerboard_tilde_5_image.rows(), 1, checkerboard_tilde_5_image.data(), checkerboard_tilde_5_image.cols()) == 0) {
        cerr << "Error: Could not save checkerboard_tilde_5 image" << endl;
        return 1;
    }

    cout << "checkerboard_tilde_5 image saved to " << output_checkerboard_tilde_5_image_path << endl;

     Matrix<unsigned char, Dynamic, Dynamic, RowMajor> checkerboard_tilde_10_image = checkerboard_tilde_10.unaryExpr([](double val) -> unsigned char {
         return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const string output_checkerboard_tilde_10_image_path = "checkerboard_tilde_10_image.png";
    if (stbi_write_png(output_checkerboard_tilde_10_image_path.c_str(), checkerboard_tilde_10_image.cols(), checkerboard_tilde_10_image.rows(), 1, checkerboard_tilde_10_image.data(), checkerboard_tilde_10_image.cols()) == 0) {
        cerr << "Error: Could not save checkerboard_tilde_10 image" << endl;
        return 1;
    }

    cout << "checkerboard_tilde_10 image saved to " << output_checkerboard_tilde_10_image_path << endl;   

    return 0;
}