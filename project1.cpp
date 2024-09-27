#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstring>

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
  VectorXd v=Map<VectorXd>(image.transpose().data(), image.size());
  cout << "Reshape the image matrix into a vector v of size " << v.size() << ". Is the size equal to mn? " << ((v.size()==height*width)?"True":"False") << endl;

  // int c=0;

  
  // for (int i = 0; i < 10 && i < v.size(); ++i) {
  //     std::cout << v(i) << "; "; // Print each value
  // }

  // for(int i=0; i<image.rows(); i++){
  //   for(int j=0; i<image.cols(); i++){
  //     if(image(i, j)!=v(j*image.cols()+i)){
  //       c++;
  //     }
  //   }
  // }

  // cout<<"Numero di valori diversi="<<c<<endl;

  VectorXd w=Map<VectorXd>(noisy_image.transpose().data(), noisy_image.size());
  cout << "Reshape the noisy image matrix into a vector w of size " << w.size() << ". Is the size equal to mn? " << ((w.size()==height*width)?"True":"False") << endl;
  cout << "Norm of vector v, ||v||=" << v.norm() << endl;

  // Task 4 //
  vector<Triplet<double>> tripletList;
  tripletList.reserve(9 * height * width); // Reserve space for triplets
  SparseMatrix<double, RowMajor> A1(height * width, height * width);

  for (int i = 0; i < height * width; i++) {

    //i=row*width+col
      int row = i % width; // Current row
      int col = i / width; // Current column

      // if(i<10) cout<<v(i)<<"||"<<image(row, col)<<endl;
      

      for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
              // Calculate neighbor indices
              int neighborRow = row + j;
              int neighborCol = col + k;

              // Check if the neighbor indices are within bounds
              if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                  int neighborIndex = neighborCol * width + neighborRow; // Calculate the linear index
                  tripletList.push_back(Triplet<double>(i, neighborIndex, 1.0 / 9.0));
              }
          }
      }
  }

  A1.setFromTriplets(tripletList.begin(), tripletList.end());

  cout<<"A1 creata"<<endl;

  MatrixXd prova(height, width);

  cout<<"Inizio convoluzione"<<endl;

  //prova convoluzione
  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      double somma=0;
      for(int k=-1; k<=1; k++){
        for(int l=-1; l<=1; l++){
          int neighborRow = i + k;
          int neighborCol = j + l;
          if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
            somma+=image(neighborRow,neighborCol)*1.0/9.0;
          }
        }
      }
      prova(i, j)=somma;
    }
  }

  cout<<endl<<"Convoluzione fatta"<<endl;

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_prova_image(height, width);
   cout<<"The matrix has: "<<prova.rows()<<" rows and "<<prova.cols()<<" columns"<<endl;
  smoothed_prova_image = prova.unaryExpr([](double val) -> unsigned char {
  return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
  });
  const string smoothed_prova_image_path = "output_porva_smooth.png";
  if (stbi_write_png(smoothed_prova_image_path.c_str(), width, height, 1, smoothed_prova_image_path.data(), width) == 0) {
    cerr << "Error: Could not save noisy image" << endl;
    return 1;
    }
  cout << "smooth image saved to " << smoothed_prova_image_path << endl;


  //cout << "Sparse Matrix A1:\n" << MatrixXd(A1) << endl;
  std::cout<< "numero di non zeri: "<<A1.nonZeros()<<endl;


  // Output the sparse matrix for verification
  // std::cout << MatrixXd(A1.block(3250, 0, 10, 10)) << std::endl;
  
  VectorXd smoothed_image=A1*v;
  //cout << "number of elements smoothed_image" << smoothed_image.size() << ". Is the size equal to mn? " << ((smoothed_image.size()==height*width)?"True":"False") << endl;
  

  //std::cout << "Smoothed image min: " << smoothed_image.minCoeff() << std::endl;
  //std::cout << "Smoothed image max: " << smoothed_image.maxCoeff() << std::endl;

  smoothed_image *= 255.0;

  MatrixXd smoothed(height, width);

  cout<<"The matrix has: "<<smoothed.rows()<<" rows and "<<smoothed.cols()<<" columns"<<endl;

    for (int i = 0; i < width; i++) {
      for (int j = 0; j <height ; j++) {
      int index = width*i+j;
      smoothed(i, j) = static_cast<double>(smoothed_image[index]);
    }
  }

  cout<<"-----------------------"<<endl;

  double somma=0;

  double vettore[6];
  int c=0;

  //

  cout<<"--------------------"<<endl;
  cout<<v(1)<<endl;
  for(int i=0; i<width*height; i++){
    if(A1.coeff(1, i)!=0) {
      cout <<v(i)<<""<<"="<< i <<", ("<< i/height <<", "<< i%height << "); ";
      somma+=A1.coeff(1, i);
      vettore[c]=v(i);
      c++;
    }
  }
  cout<<endl<<"--------------------"<<endl;
  cout<<image(1,0)<<", "<<somma*image(1,0)<<endl;

  cout<<"--------------------"<<endl;

  for(int i=0; i<6; i++){
    cout<<vettore[i]<<"; ";
  }

  cout<<endl<<"----------------"<<endl;

  std::cout << MatrixXd(image.block(0, 0, 5, 5)) << std::endl;
  cout<<"------------------"<<endl;
  std::cout << MatrixXd(smoothed.block(0, 0, 3, 3)) << std::endl;

   Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_final_image(width, height);
   cout<<"The matrix has: "<<smoothed_final_image.rows()<<" rows and "<<smoothed.cols()<<" columns"<<endl;
  smoothed_final_image = smoothed.unaryExpr([](double val) -> unsigned char {
  return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
  });
  const string output_smooth_image_path = "output_smooth.png";
  if (stbi_write_png(output_smooth_image_path.c_str(), width, height, 1, smoothed_final_image.data(), width) == 0) {
    cerr << "Error: Could not save noisy image" << endl;
    return 1;
    }
  cout << "smooth image saved to " << output_smooth_image_path << endl;


  return 0;
}