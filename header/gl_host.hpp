
#ifndef __GLHOSTHPP
#define __GLHOSTHPP

// ==== compile ====
// linux   : g++ showHostmem.cpp -O2 -lglfw -lGL
// windows : cl  showHostmem.cpp -O2 -lglfw3 -lopengl32
//           ====> Install glfw3
// =================

// ===== show "img" ===
// glImageHost glh(ImageWidth,ImageHeight,type, windowWidth, windowHeight,windowName); //type -> GRAY or RGB or RGBA
// while (!glh.isClosed()){
//     glh.show(img);
// }
// ====================


#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <GLFW/glfw3.h>

#include "gl_show_util.hpp"


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

class glImageHost{
public:
    glImageHost(size_t imageWidth, size_t imageHeight, colorType format,size_t windowWidth, size_t windowHeight,const char* windowName);
    ~glImageHost();
    void show(void *image);
    int isClosed() { return glfwWindowShouldClose(window); } ;
    size_t imageWidth, imageHeight;
    size_t windowWidth, windowHeight;
    //const char * windowName;
    //colorType format;

private:
    
    void createWindow(const char *windowName);
    void createTexture();
    void init();
    GLFWwindow *window;

    GLuint Texture;
    
    GLenum type;
};

glImageHost::glImageHost(
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
    }else if(format == RGB){
        type = GL_RGB;
    }else {
        type = RGBA;
    }

    createWindow( windowName );
    createTexture();

}
glImageHost::~glImageHost(){
    glfwTerminate();
}

void glImageHost::createWindow(const char *windowName){
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

void glImageHost::createTexture(){

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

}

void glImageHost::show( void *image){
    
    glTexImage2D(GL_TEXTURE_2D, 0, type, windowWidth, windowHeight, 0, type, GL_UNSIGNED_BYTE, image );

    glDrawArrays(GL_TRIANGLE_STRIP, 0, NUM_VERTICES);
    glfwPollEvents();
    glfwSwapBuffers(window);
}


#endif