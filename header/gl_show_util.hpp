#ifndef __GLSHOWUTILHPP
#define __GLSHOWUTILHPP

#define VERTEX_DIMENSION (2)
#define NUM_VERTICES (4)
#define ERROR(s)   std::cerr <<"Error::"<<__FILE__<<"::"<<__LINE__<<"::"<<__FUNCTION__<<"::"<<(s)<<std::endl;



const float __VERTEX_ARRAY[] = {
    -1.0f, -1.0f,// lower left
    1.0f, -1.0f, // lower right
    -1.0f, 1.0f, // upper left
    1.0f, 1.0f   // upper right
};
const float __TEXTURE_COORED_ARRAY[] = {
    0.0f, 1.0f, // lower left
    1.0f, 1.0f,	// lower right
    0.0f, 0.0f,	// upper left
    1.0f, 0.0f,	 // upper right
};

enum colorType{
    GRAY = 1,
    RGB =  3,
    RGBA = 4
};

#endif