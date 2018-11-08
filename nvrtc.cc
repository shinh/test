// Single-precision a-X plus Y from
// https://docs.nvidia.com/cuda/nvrtc/index.html

// g++ nvrtc.cc -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lnvrtc

#include <stdio.h>

#include <cuda.h>
#include <nvrtc.h>

void check_nvrtc(nvrtcResult status, const char* msg) {
    if (status != NVRTC_SUCCESS) {
        fprintf(stderr, "NVRTC: %s\n", nvrtcGetErrorString(status));
    }
}

#define CHECK_NVRTC(expr) check_nvrtc(expr, #expr)

int main() {
    int major, minor;
    CHECK_NVRTC(nvrtcVersion(&major, &minor));
    printf("Version: %d %d\n", major, minor);

    nvrtcProgram prog;
    const char *saxpy = (
        "extern \"C\" __global__\n"
        "void saxpy(float a, float *x, float *y, float *out, size_t n)\n"
        "{\n"
        "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (tid < n) {\n"
        "    out[tid] = a * x[tid] + y[tid];\n"
        "  }\n"
        "}\n");

    CHECK_NVRTC(nvrtcCreateProgram(&prog,
                                   saxpy,
                                   "saxpy.cu",
                                   0,
                                   nullptr,
                                   nullptr));

    const char *opts[] = {"--gpu-architecture=compute_60"};
    CHECK_NVRTC(nvrtcCompileProgram(prog,
                                    1,
                                    opts));

    // Obtain compilation log from the program.
    size_t logSize;
    CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
    char* log = new char[logSize];
    CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
    // Obtain PTX from the program.
    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = new char[ptxSize];
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx));

    CHECK_NVRTC(nvrtcDestroyProgram(&prog));

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&kernel, module, "saxpy");
#define NUM_THREADS 128
#define NUM_BLOCKS 32
    size_t n = NUM_THREADS * NUM_BLOCKS;
    size_t bufferSize = n * sizeof(float);
    float a = 5.1f;

    float* hX = new float[n],* hY = new float[n],* hOut = new float[n];
    for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }

    CUdeviceptr dX, dY, dOut;
    cuMemAlloc(&dX, bufferSize);
    cuMemAlloc(&dY, bufferSize);
    cuMemAlloc(&dOut, bufferSize);
    cuMemcpyHtoD(dX, hX, bufferSize);
    cuMemcpyHtoD(dY, hY, bufferSize);
    void *args[] = { &a, &dX, &dY, &dOut, &n };
    cuLaunchKernel(kernel,
                   NUM_THREADS, 1, 1,   // grid dim
                   NUM_BLOCKS, 1, 1,    // block dim
                   0, NULL,             // shared mem and stream
                   args,                // arguments
                   0);
    cuCtxSynchronize();
    cuMemcpyDtoH(hOut, dOut, bufferSize);

    for (size_t i = 0; i < n; ++i) {
        fprintf(stderr, "i=%zu x=%f y=%f o=%f\n", i, hX[i], hY[i], hOut[i]);
    }
}
