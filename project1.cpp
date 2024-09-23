#include <Eigen/Dense>
#include <iostream>

// from https://github.com/nothings/stb/tree/master
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
    int width, height;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height);  // Force load as RGB


    return 0;
}