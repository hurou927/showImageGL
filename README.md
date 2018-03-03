# showImageGL

Show image in NVIDIA-GPU device memory (global memory).

Require : glfw3 , CUDA


# Code

```cpp 
    glImageDevice glh(imageWidth,imageHeight,type, windowWidth, windowHeight,windowName);
    // cudaMalloc and cudaMemcpy
    glh.setGlobalMemoryPtr((void *)global_memory_ptr_RGBA); // Set device memory ptr. for showing image
    while (!glh.isClosed()) {
        // GPU computing
        glImageDevice::RGB2RGBA(cuda_globalMemory_RGBA,global_memory_ptr_RGB, imageWidth, imageHeight); // RGB to RGBA (CUDA does not support 3 components/pixel)
    }
    glh.show();
}
```

# build
``` nvcc  showHostmem.cpp -O2 -lglfw3 -lopengl32 ```