# showImageGL

Show image in NVIDIA-GPU device memory (global memory).

Require : glfw3 , CUDA


# Code

```cpp 
    glImageDevice glh(imageWidth,imageHeight,type, windowWidth, windowHeight,windowName);
    // ::::::::::::::
    // cudaMalloc and cudaMemcpy
    // ::::::::::::::
    // Set device memory ptr. for showing image
    glh.setGlobalMemoryPtr((void *)global_memory_ptr_RGBA);
    while (!glh.isClosed()) {
        // ::::::::::::::
        // GPU computing
        // ::::::::::::::
        // RGB to RGBA (CUDA does not support 3 components/pixel)
        // If you show RGB image, you must call RGB2RGBA func. or write CUDA code for RGBA image
        glImageDevice::RGB2RGBA(cuda_globalMemory_RGBA,global_memory_ptr_RGB, imageWidth, imageHeight); 
    glh.show();
}
```

# build
``` nvcc  showHostmem.cpp -O2 -lglfw3 -lopengl32 ```

# License
This software is released under the MIT License, see ```LICENSE.txt```.