#include <Eigen/Dense>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

MatrixXd random_Noise_Generator(const MatrixXd& image) {
  MatrixXd noise= MatrixXd::Random(image.rows(), image.cols());
  MatrixXd noise_Added_Image = image + 50*noise;

  return noise_Added_Image;
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
    }
    const char* input_image_path = argv[1];
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height,&channels, 1);
    MatrixXd image(height, width);
    std::cout << "Image size: " << height << "x" << width << std::endl;
    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j);
      image(i, j) = static_cast<double>(image_data[index]);
        }
    }
    stbi_image_free(image_data);
    std::cout<<"the matrix has:"<<image.rows()<<"rows and"<<image.cols()<<"columns"<<std::endl;

    MatrixXd noisy_image = random_Noise_Generator(image);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisi_final_image(height, width);
    noisi_final_image = noisy_image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });





    return 0;
}