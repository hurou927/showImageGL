
// hostメモリ上の画像を表示
// g++ -o main main.cpp -O2 -lGL -lglfw

#include <iostream>
#include "header/gl_host.hpp"
#include "header/my_pgm.h"

int main(int argc, char**argv){
    
    size_t Max,width,height,SamplePerPixel;
	//unsigned char * img = ReadPNM((char*)"lena_gray.pgm",NULL, &Max,&width,&height,&SamplePerPixel);
    unsigned char * img = ReadPNM("image/lena_color.ppm",NULL, &Max,&width,&height,&SamplePerPixel);

    colorType type = (SamplePerPixel == 1) ? GRAY : RGB;

    size_t windowWidth = 512;
    size_t windowHeight = 512;
    glImageHost mygl(width,height,type, windowWidth, windowHeight,"test window");

    std::cout << "End : ESCAPE key" << std::endl;
    while (!mygl.isClosed()) {
        for(size_t i=0;i<width*height*SamplePerPixel;i++) img[i]=img[i]-1;
        mygl.show(img);
    }

    free(img);
    return 0;
}