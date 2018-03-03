//
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
// CUDA arrays are opaque memory layouts optimized for texture fetching. 
// They are one dimensional, two dimensional, or three-dimensional and composed of elements, 
// each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 
// 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through 
// texture fetching as described in Texture Memory or surface reading and writing as 
// described in Surface Memory.
// 

// hostメモリ上の画像を表示
// g++ -o main main.cpp -O2 -lGL -lglfw

#include <iostream>
#include "header/gl_device_cuda.cuh"
#include "header/my_pgm.h"
// #define RGBALIGNMENT 3

template < typename T >
__global__ void minus1( T *in, unsigned int size){
    unsigned int index = threadIdx.x  + blockDim.x * blockIdx.x; 
    if( index < size )
        in[index] = in[index] - 1;
}


int main(int argc, char**argv){
    using namespace std;
    size_t Max,width,height,SamplePerPixel;
	//unsigned char * img = ReadPNM((char*)"lena_gray.pgm",NULL, &Max,&width,&height,&SamplePerPixel);
    unsigned char * img = ReadPNM("image/lena_color.ppm",NULL, &Max,&width,&height,&SamplePerPixel);

    colorType type = (SamplePerPixel == 1) ? GRAY : RGB;

    size_t windowWidth  = 512;
    size_t windowHeight = 512;
    glImageDevice glh(width,height,type, windowWidth, windowHeight,"test window");

    unsigned char *cuda_globalMemory;
    
    size_t imageSize = width * height *SamplePerPixel;
    size_t textureImageSize;
    if(SamplePerPixel == 1) textureImageSize = imageSize;
    else                    textureImageSize = width * height * 4;
    


    cudaMalloc((void **)&cuda_globalMemory, imageSize*sizeof(unsigned char));
    std::cout << "End : ESCAPE key" << std::endl;
    
    if(SamplePerPixel==1){       
       
        glh.setGlobalMemoryPtr((void *)cuda_globalMemory);
        cudaMemcpy(cuda_globalMemory,img,imageSize*sizeof(unsigned char),cudaMemcpyHostToDevice);
        while (!glh.isClosed()) {
            //++++++++ Host computation 
            // for(int i=0;i<width*height*SamplePerPixel;i++) img[i]=img[i]-1;
            // cudaMemcpy(cuda_globalMemory,img,imageSize*sizeof(unsigned char),cudaMemcpyHostToDevice);
            
            //++++++++ Device computation
            minus1 <unsigned char> <<< (textureImageSize+1023)/1024 , 1024 >>> ( cuda_globalMemory , textureImageSize );
            cudaDeviceSynchronize();
            
            glh.show();
        }


    }else if(SamplePerPixel==3){    
       
        unsigned char *cuda_globalMemory_RGBA;
        cudaMalloc((void **)&cuda_globalMemory_RGBA, textureImageSize*sizeof(unsigned char));
        glh.setGlobalMemoryPtr((void *)cuda_globalMemory_RGBA);

        cudaMemcpy(cuda_globalMemory,img,imageSize,cudaMemcpyHostToDevice);
        glh.RGB2RGBA(cuda_globalMemory_RGBA,cuda_globalMemory); //CUDA does not support RGBRGBRGB... (3 components/pixel)

        while (!glh.isClosed()) {

            //++++++++ Host computation 
            //  for(int i=0;i<width*height*SamplePerPixel;i++) img[i]=img[i]-1;
            //  cudaMemcpy(cuda_globalMemory,img,imageSize,cudaMemcpyHostToDevice);
            //  //glh.RGB2RGBA(cuda_globalMemory_RGBA,cuda_globalMemory); //CUDA does not support RGBRGBRGB... (3 components/pixel)
            //  glImageDevice::RGB2RGBA(cuda_globalMemory_RGBA,cuda_globalMemory,width,height); //CUDA does not support RGBRGBRGB... (3 components/pixel)

            //++++++++ Device computation
            minus1 <unsigned char> <<< (textureImageSize+1023)/1024 , 1024 >>> ( cuda_globalMemory_RGBA , textureImageSize ); // ignore Alpha values!
            cudaDeviceSynchronize();

            glh.show();
        }
        cudaFree(cuda_globalMemory_RGBA);


    }

    free(img);
    cudaFree(cuda_globalMemory);
    return 0;
}