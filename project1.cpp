#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstring>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

MatrixXd random_Noise_Generator(const MatrixXd& image) {
  MatrixXd noise= MatrixXd::Random(image.rows(), image.cols());
  MatrixXd noise_Added_Image = image + (50.0/255.0)*noise;

  return noise_Added_Image;
}


int main(int argc, char* argv[]){
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <image_path>" << endl;
    return 1;
  }
  const char* input_image_path = argv[1];
  // Task 1 //
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height,&channels, 1);
  MatrixXd image(height, width);
  cout << "Image size: " << height << "x" << width << endl;
  cout << "Image size from stbi_load: " << height << " (height) x " << width << " (width)" << endl;
  cout << "The matrix has: " << image.rows() << " (height) x " << image.cols() << " (width)" << endl;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j);
      image(i, j) = static_cast<double>(image_data[index] / 255.0);
    }
  }
  stbi_image_free(image_data);
  cout<<"The matrix has: "<<image.rows()<<" rows and "<<image.cols()<<" columns"<<endl;

  // Task 2 //
  MatrixXd noisy_image = random_Noise_Generator(image);

  cout<<"The matrix has: "<<noisy_image.rows()<<" rows and "<<noisy_image.cols()<<" columns"<<endl;

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_final_image(height, width);
  
  cout<<"The matrix has: "<<noisy_final_image.rows()<<" rows and "<<noisy_final_image.cols()<<" columns"<<endl;

  noisy_final_image = noisy_image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  const string output_noisy_image_path = "output_noisy.png";

  if (stbi_write_png(output_noisy_image_path.c_str(), width, height, 1, noisy_final_image.data(), width) == 0) {
    cerr << "Error: Could not save noisy image" << endl;
    return 1;
  }

  cout << "Noisy image saved to " << output_noisy_image_path << endl;

  // Task 3 //
  // VectorXd v=Map<VectorXd>(image.transpose().data(), image.size());
  VectorXd v(height * width);
  for(int i = 0; i < height; i++){
    for(int j = 0; j< width; j++){
      int index = i * width + j;
      v(index) = image(i,j);
    }
  }
  cout << "Reshape the image matrix into a vector v of size " << v.size() << ". Is the size equal to mn? " << ((v.size()==height*width)?"True":"False") << endl;

  // VectorXd w=Map<VectorXd>(noisy_image.transpose().data(), noisy_image.size());
  VectorXd w(height * width);
  for(int i = 0; i < height; i++){
    for(int j = 0; j< width; j++){
      int index = i * width + j;
      w(index) = noisy_image(i,j);
    }
  }
  cout << "Reshape the noisy image matrix into a vector w of size " << w.size() << ". Is the size equal to mn? " << ((w.size()==height*width)?"True":"False") << endl;


  cout << "Norm of vector v, ||v||=" << v.norm() << endl;


  // Task 4 //
  vector<Triplet<double>> tripletList;
  tripletList.reserve(9 * height * width); // Reserve space for triplets
  SparseMatrix<double, RowMajor> A1(height * width, height * width);

  MatrixXd H_av2(3,3);
  H_av2<<1, 1, 1,
         1, 1, 1,
         1, 1, 1;
  H_av2/=9.0;

  for (int i = 0; i < height * width; i++) {
    //i=row*width+col
    int row = i / width; // Current row
    int col = i % width; // Current column

     // if(i<10) cout<<v(i)<<"||"<<image(row, col)<<endl;
      
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        // Calculate neighbor indices
        int neighborRow = row + j;
        int neighborCol = col + k;

        // Check if the neighbor indices are within bounds
        if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
          int neighborIndex = neighborRow * width + neighborCol; // Calculate the linear index
          tripletList.push_back(Triplet<double>(i, neighborIndex, H_av2(j+1, k+1)));
          }
        }
      }
  }

  A1.setFromTriplets(tripletList.begin(), tripletList.end());
  
  double sum = 0;
  for (int b = 0; b < 3; b++){
    for (int a = 0; a < height*width; a++){
      if(A1.coeff(b,a) != 0){
        cout << A1.coeff(b,a) << "in posizione: " << b << " " << a << endl;
        sum += A1.coeff(b,a);
      }
    }
    cout << "Somma totale riga " << b << " di A1: " << sum << endl;
    sum = 0;
  }

  cout << "Fino a qui tutto bene" << endl;

  VectorXd smoothed_vector=255*A1*v;


  cout << "Coefficiente di image: " << image(0,0) << endl;
  cout << "Coefficiente di v: " << v(0) << endl;
  cout << "Coefficiente di smoothed_image: " << smoothed_vector.coeff(0) << endl;

  cout << "Coefficiente di image: " << image(1,0) << endl;
  cout << "Coefficiente di v: " << v(3250) << endl;
  cout << "Coefficiente di smoothed_image: " << smoothed_vector.coeff(3250) << endl;

  cout << "number of elements smoothed_image" << smoothed_vector.size() << ". Is the size equal to mn? " << ((smoothed_vector.size()==height*width)?"True":"False") << endl;
  cout << "Smoothed image rows: " << smoothed_vector.rows() << " and cols: " << smoothed_vector.cols() << endl;
  cout << "A1 rows: " << A1.rows() << " and cols: " << A1.cols() << endl;

  cout << "Fino a qui tutto bene 2" << endl;

  MatrixXd smoothed_matrix(height, width);

  cout<<"The matrix has: "<<smoothed_matrix.rows()<<" rows and "<<smoothed_matrix.cols()<<" columns"<<endl;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;  // Row-major indexing
      smoothed_matrix(i, j) = smoothed_vector[index];
    }
  }

  cout << "Fino a qui tutto bene 3" << endl;
  cout << "Original image dimensions: " << image.rows() << "x" << image.cols() << endl;
  cout << "v vector size: " << v.size() << endl;
  cout << "smoothed_image vector size: " << smoothed_vector.size() << endl;
  cout << "Final smoothed image dimensions: " << smoothed_matrix.rows() << "x" << smoothed_matrix.cols() << endl;
  
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_final_image = 
  smoothed_matrix.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
  });

  cout<<"The matrix has: "<<smoothed_final_image.rows()<<" rows and "<<smoothed_final_image.cols()<<" columns"<<endl;
  const string output_smooth_image_path = "output_smooth.png";
  if (stbi_write_png(output_smooth_image_path.c_str(), width, height, 1, smoothed_final_image.data(), width) == 0) {
    cerr << "Error: Could not save noisy image" << endl;
    return 1;
    }
  cout << "smooth image saved to " << output_smooth_image_path << endl;

  // Task 5 //
  VectorXd smoothed_noisy_vector=255*A1*w;
  MatrixXd smoothed_noisy_matrix(height, width);

  cout<<"The matrix has: "<<smoothed_noisy_matrix.rows()<<" rows and "<<smoothed_noisy_matrix.cols()<<" columns"<<endl;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;  // Row-major indexing
      smoothed_noisy_matrix(i, j) = smoothed_noisy_vector[index];
    }
  }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_noisy_final_image = 
  smoothed_noisy_matrix.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
  });

  const string output_smooth_noisy_image_path = "output_smooth_noisy.png";
  if (stbi_write_png(output_smooth_noisy_image_path.c_str(), width, height, 1, smoothed_noisy_final_image.data(), width) == 0) {
    cerr << "Error: Could not save noisy image" << endl;
    return 1;
    }
  cout << "smooth image saved to " << output_smooth_noisy_image_path << endl;
  

  return 0;
}