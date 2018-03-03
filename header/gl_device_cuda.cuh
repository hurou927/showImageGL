
#ifndef __GLDEVICECUDACUH
#define __GLDEVICECUDACUH

// require : CUDA and NVIDIA's GPU connecting Display
//
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
// CUDA arrays are opaque memory layouts optimized for texture fetching. 
// They are one dimensional, two dimensional, or three-dimensional and composed of elements, 
// each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 
// 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through 
// texture fetching as described in Texture Memory or surface reading and writing as 
// described in Surface Memory.
// 
// ==== compile ====
// linux   : nvcc showHostmem.cpp -O2 -lglfw -lGL
// windows : nvcc  showHostmem.cpp -O2 -lglfw3 -lopengl32
//           ====> Install glfw3
// =================
//
// ==== example ====
// // Show GrayScale or RGBA image
// glImageDevice glh(imageWidth,imageHeight,type, windowWidth, windowHeight," windowName);
// glh.setGlobalMemoryPtr((void *)global_memory_ptr);
// while (!mygl.isClosed()) {
// //   =====GPU computing====== 
//      glh.show();
// }
// // SHOW RGB image (Convert RGB to RGBA)
// glImageDevice glh(imageWidth,imageHeight,type, windowWidth, windowHeight," windowName);
// glh.setGlobalMemoryPtr((void *)global_memory_ptr_RGBA);
// while (!mygl.isClosed()) {
// //   =====GPU computing====== 
//      glImageDevice::RGB2RGBA(cuda_globalMemory_RGBA,global_memory_ptr_RGB, imageWidth, imageHeight);
//      glh.show();
// }
//===================

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "gl_show_util.hpp"


 void checkCudaStatus(){
     cudaError_t err=cudaGetLastError();
     if(err){
         fprintf(stderr,"checkCudaStatus::%s(code:%d)\n",cudaGetErrorString(err),err);
         exit(1);
     }
 }


// RGBRGBRGB... ==> RGBARGBARGBA...
template <typename T, unsigned int NUMTHREADS>
__global__ void RGB2RGBA_device(T *out, T*in , unsigned int WIDTH, unsigned HEIGHT){
    
    __shared__ T s_buf[NUMTHREADS*4];
    
    s_buf[threadIdx.x*4+3] = 0;

    unsigned int startIndex;
    unsigned int offset;

    startIndex = 3 * NUMTHREADS * blockIdx.x;
    offset = threadIdx.x;
    for(int i=0; i<3; i++){
        if( startIndex + offset < WIDTH*HEIGHT*3 ){
            s_buf[ offset+(offset)/3 ] = in[ startIndex + offset ] ;
        }
        offset += NUMTHREADS;
        
    }
    __syncthreads();
    startIndex = 4 * NUMTHREADS * blockIdx.x;
    offset = threadIdx.x;
    for(int i=0; i<4; i++){
        if( startIndex + offset < WIDTH*HEIGHT*4 )
            out[ startIndex + offset ] = s_buf[ offset ] ;
        offset += NUMTHREADS;
        
    }



}






static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}



class glImageDevice{
public:
    glImageDevice(size_t imageWidth, size_t imageHeight, colorType format,size_t windowWidth, size_t windowHeight,const char* windowName);
    ~glImageDevice();
    void show();
    void setGlobalMemoryPtr(void *gmemptr);
    int isClosed() { return glfwWindowShouldClose(window); } ;
    void RGB2RGBA(unsigned char *out, unsigned char *in);
    static void  RGB2RGBA(unsigned char *out, unsigned char *in, size_t imageWidth, size_t imageHeight);   

    size_t imageWidth, imageHeight;
    size_t windowWidth, windowHeight;
    size_t bytesPerPixel;
    //colorType format;

private:
    
    void createWindow( const char * windowName );
    void createTexture();
    void init();

    
    GLFWwindow *window;

    GLuint Texture;
    
    GLenum type;
    void *GlobalMemoryPtr;
    struct cudaGraphicsResource *cuda_tex_result_resource;
    cudaArray *texture_ptr;
};

glImageDevice::glImageDevice(
    size_t imageWidth, 
    size_t imageHeight, 
    colorType format, 
    size_t windowWidth, 
    size_t windowHeight,
    const char* windowName)
    : imageWidth(imageWidth), imageHeight(imageHeight), windowWidth(windowWidth), windowHeight(windowHeight)
    {

    if( !glfwInit() ){
        ERROR("glfwInit error\n");
        exit (EXIT_FAILURE);
    }
    
    if(format == GRAY){
        type = GL_LUMINANCE;
        bytesPerPixel = 1;
    }else if(format == RGB){
        type = GL_RGB;
        bytesPerPixel = 4;
    }else {
        type = RGBA;
        bytesPerPixel = 4;
    }

    createWindow( windowName );
    createTexture();

}
glImageDevice::~glImageDevice(){
    cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
    glfwTerminate();
}





void glImageDevice::createWindow(const char *windowName){
    window = glfwCreateWindow(windowWidth, windowHeight, windowName, NULL, NULL);
    if (!window) {
        glfwTerminate();
        ERROR("open window error\n");
        exit (EXIT_FAILURE);
    }
    //glfwWindowHint(GLFW_ALPHA_BITS, 0);
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
}





void glImageDevice::createTexture(){

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,&Texture);

    glBindTexture(GL_TEXTURE_2D, Texture );
        
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( VERTEX_DIMENSION, GL_FLOAT, 0, __VERTEX_ARRAY );

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glTexCoordPointer(2, GL_FLOAT, 0, __TEXTURE_COORED_ARRAY);

    glTexImage2D(GL_TEXTURE_2D, 0, type , imageWidth, imageHeight, 0, type, GL_UNSIGNED_BYTE, NULL );

}





void glImageDevice::setGlobalMemoryPtr(void *gmemptr){

    if(gmemptr==NULL){
        fprintf(stderr,"Malloc globalMemory!!!\n");
        exit(1);
    }

    GlobalMemoryPtr = gmemptr;
    //register
    cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, Texture,GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    //map
    cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);
}





void glImageDevice::show(){
    
    cudaMemcpyToArray(texture_ptr, 0, 0, GlobalMemoryPtr, imageWidth*imageHeight*bytesPerPixel, cudaMemcpyDeviceToDevice);
    glBindTexture(GL_TEXTURE_2D, Texture );
    glDrawArrays(GL_TRIANGLE_STRIP, 0, NUM_VERTICES);
    glfwPollEvents();
    glfwSwapBuffers(window);
}





void glImageDevice::RGB2RGBA(unsigned char *devout, unsigned char *devin){
    if( type == GL_RGB ){
        size_t numBlocks = ( imageWidth * imageHeight * 3 + 511 ) / (512) ;
        RGB2RGBA_device <unsigned char, 512 > <<< numBlocks , 512 >>>(devout, devin , imageWidth, imageHeight);
        cudaDeviceSynchronize();
        checkCudaStatus();
    }
}





void glImageDevice::RGB2RGBA(unsigned char *devout, unsigned char *devin , size_t imageWidth, size_t imageHeight){

    size_t numBlocks = ( imageWidth * imageHeight * 3 + 511 ) / (512) ;
    RGB2RGBA_device <unsigned char, 512 > <<< numBlocks , 512 >>>(devout, devin , imageWidth, imageHeight);
    cudaDeviceSynchronize();
    checkCudaStatus();
}




#endif