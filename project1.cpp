#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

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
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_final_image(height, width);
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
  VectorXd v=Map<VectorXd>(image.data(), image.size());
  cout << "Reshape the image matrix into a vector v of size " << v.size() << ". Is the size equal to mn? " << ((v.size()==height*width)?"True":"False") << endl;
  VectorXd w=Map<VectorXd>(noisy_image.data(), noisy_image.size());
  cout << "Reshape the noisy image matrix into a vector w of size " << w.size() << ". Is the size equal to mn? " << ((w.size()==height*width)?"True":"False") << endl;
  cout << "Norm of vector v, ||v||=" << v.norm() << endl;

  // Task 4 //
  MatrixXd H_av2=(1.0/9.0)*MatrixXd::Ones(3,3);
  vector<Triplet<double>> tripletList;
  tripletList.reserve(9*height*width);
  SparseMatrix<double> A1(height*width, height*width);
  for(int i=0; i<height*width; i++){
    for( int j= -1; j<=1; j++){
      for ( int k= -1; k<=1; k++){
        if(i/width+j<0 || i/width+j>height-1 || i%width+k<0 || i%width+k>width-1){continue;}
        tripletList.push_back(Triplet<double>(i, j*width+k, 1.0/9.0));
      }
    }  
  }
  A1.setFromTriplets(tripletList.begin(), tripletList.end());
  std::cout<<A1.nonZeros()<<endl;
  std::cout<<A1<<endl;
  return 0;
}