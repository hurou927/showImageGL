# showImageGL

Show image(__ONLY 8bits/component__) in NVIDIA-GPU device memory (global memory).

Require : glfw3 , CUDA


# Code

## include 

``` #include "header/gl_device_cuda.cuh" ```

## main

```cpp
    colorType type = RGB; // support GRAY, RGB or RGBA
    glImageDevice glh(imageWidth,imageHeight,type, windowWidth, windowHeight,windowName);
    // ::::::::::::::
    // cudaMalloc( and cudaMemcpy )
    // ::::::::::::::
    // Set device memory ptr. of showed image
    glh.setGlobalMemoryPtr((void *)gmem_RGBA);
    while (!glh.isClosed()) {
        // ::::::::::::::
        // GPU computing
        // ::::::::::::::
        // RGB to RGBA (CUDA does not support 3 components/pixel)
        // If you show RGB image, you must call RGB2RGBA func. or write CUDA code for RGBA image
        glImageDevice::RGB2RGBA(gmem_RGBA, gmem_RGB, imageWidth, imageHeight); 
        glh.show();
    }
```

# Build

* linux ``` nvcc showHostmem.cpp -lglfw3 -lGL ```

* windows ``` nvcc showHostmem.cpp -lglfw3 -lopengl32```

# License
This software is released under the MIT License, see ```LICENSE.txt```.
