#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <vector>

#include <cuda.h>

const int A = 1920;
const int B = 132;
const int C = 396;

std::vector<float> AddAtNaive(const std::vector<float>& y,
                              const std::vector<int>& indices) {
    std::vector<float> x(A * B);
    for (int i = 0; i < A; ++i) {
        for (int j = 0; j < C; ++j) {
            x[i * B + indices[j]] += y[i * C + j];
        }
    }
    return x;
}

void* GPUMalloc(size_t sz) {
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, sz);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        abort();
    }
    return p;
}

void GPUMemcpy(void* dst, const void* src, size_t cnt, enum cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, cnt, kind);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        abort();
    }
}

__global__ void ZeroCudaKernel(float* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] = 0;
}

__global__ void AddAtCudaKernel(float* x, int* indices, float* y, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    atomicAdd(&x[i * B + indices[j]], y[i * C + j]);
}

std::vector<float> AddAtCuda(const std::vector<float>& y,
                             const std::vector<int>& indices,
                             int n) {
    std::vector<float> x(A * B);
    float* xg = (float*)GPUMalloc(x.size() * sizeof(x[0]));
    int* ig = (int*)GPUMalloc(indices.size() * sizeof(indices[0]));
    float* yg = (float*)GPUMalloc(y.size() * sizeof(y[0]));
    fprintf(stderr, "memcpy yg\n");
    GPUMemcpy(yg, y.data(), y.size() * sizeof(y[0]), cudaMemcpyHostToDevice);
    fprintf(stderr, "memcpy ig\n");
    GPUMemcpy(ig, indices.data(), indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice);

    fprintf(stderr, "take\n");
    clock_t start = clock();
    for (int i = 0; i < n; ++i) {
        ZeroCudaKernel<<<A, B>>>(xg);
        AddAtCudaKernel<<<A, C>>>(xg, ig, yg, 0);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        abort();
    }

    fprintf(stderr, "Elapsed %f\n",
            ((double)(clock() - start)) / CLOCKS_PER_SEC);

    fprintf(stderr, "memcpy yg\n");
    GPUMemcpy(&x[0], xg, x.size() * sizeof(x[0]), cudaMemcpyDeviceToHost);
    fprintf(stderr, "done\n");
    return x;
}

int main() {
    std::vector<float> y(A * C);
    std::vector<int> indices(C);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = i;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i * i % C;
    }

    std::vector<float> ex = AddAtNaive(y, indices);
    std::vector<float> ax = AddAtCuda(y, indices, 1);

    for (size_t i = 0; i < ex.size(); ++i) {
        // fprintf(stderr, "%f %f\n", ex[i], ax[i]);
        assert(abs(ex[i] - ax[i]) < 1e-10);
    }

    AddAtCuda(y, indices, 40);
}
