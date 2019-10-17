#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <vector>

#include <cuda.h>

const int A = 1920;
const int B = 132;
const int C = 396;

std::vector<float> TakeNaive(const std::vector<float>& x,
                             const std::vector<int>& indices) {
  std::vector<float> y(A * C * D);
  for (int i = 0; i < A; ++i) {
    for (int j = 0; j < C; ++j) {
      y[i * C + j] = x[i * B + indices[j]];
    }
  }
  return y;
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

__global__ void TakeCudaKernel(float* x, int* indices, float* y, int n) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  y[i * C + j] = x[i * B + indices[j]];
}

std::vector<float> TakeCuda(const std::vector<float>& x,
                            const std::vector<int>& indices,
                            int n) {
  std::vector<float> y(A * C);
  float* xg = (float*)GPUMalloc(x.size() * sizeof(x[0]));
  int* ig = (int*)GPUMalloc(indices.size() * sizeof(indices[0]));
  float* yg = (float*)GPUMalloc(y.size() * sizeof(y[0]));
  fprintf(stderr, "memcpy xg\n");
  GPUMemcpy(xg, x.data(), x.size() * sizeof(x[0]), cudaMemcpyHostToDevice);
  fprintf(stderr, "memcpy ig\n");
  GPUMemcpy(ig, indices.data(), indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice);

  fprintf(stderr, "take\n");
  clock_t start = clock();
  for (int i = 0; i < n; ++i) {
    TakeCudaKernel<<<A, C>>>(xg, ig, yg, 0);
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    abort();
  }

  fprintf(stderr, "Elapsed %f\n",
          ((double)(clock() - start)) / CLOCKS_PER_SEC);

  fprintf(stderr, "memcpy yg\n");
  GPUMemcpy(&y[0], yg, y.size() * sizeof(y[0]), cudaMemcpyDeviceToHost);
  fprintf(stderr, "done\n");
  return y;
}

int main() {
  std::vector<float> x(A * B);
  std::vector<int> indices(C);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = i;
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i * i % C;
  }

  std::vector<float> ey = TakeNaive(x, indices);
  std::vector<float> ay = TakeCuda(x, indices, 1);

  for (size_t i = 0; i < ey.size(); ++i) {
    //fprintf(stderr, "%zu %f %f\n", i, ey[i], ay[i]);
    assert(abs(ey[i] - ay[i]) < 1e-10);
  }

  TakeCuda(x, indices, 40);
}
