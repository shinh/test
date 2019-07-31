// Usage
//
// $ g++ -g -shared -fPIC -I /usr/local/cuda/include cutrace.cc -ldl -lcuda -o cutrace.so
// $ LD_PRELOAD=./cutrace.so some_program
//

#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <map>

#include <cuda.h>

namespace {

template <typename Fn>
class FunctionHook {
public:
    static_assert(sizeof(Fn) == sizeof(void*));

    FunctionHook(const char* fn_name, Fn hook_fn)
        : fn_name_(fn_name), hook_fn_(hook_fn) {
        orig_fn_ = (Fn)dlsym(RTLD_NEXT, fn_name_);
        assert(orig_fn_);
        if (mprotect((void*)((uintptr_t)orig_fn_ & ~4095),
                     4096 * 2,
                     PROT_READ | PROT_WRITE | PROT_EXEC) < 0) {
            perror("mprotect");
            abort();
        }
        memcpy(orig_code_, (void*)orig_fn_, 16);
        Patch();
    }

    void Patch() {
        uint8_t jmp[] = {0xff, 0x25, 0, 0, 0, 0};
        memcpy((void*)orig_fn_, jmp, 6);
        void* addr = (void*)hook_fn_;
        memcpy((char*)orig_fn_ + 6, &addr, sizeof(addr));
    }

    void Unpatch() {
        memcpy((void*)orig_fn_, orig_code_, 16);
    }

    Fn orig_fn() {
        return orig_fn_;
    }

private:
    const char* fn_name_;
    Fn orig_fn_;
    Fn hook_fn_;
    char orig_code_[16];
};

FunctionHook<decltype(cuLaunchKernel)*>* cuLaunchKernel_hook;
FunctionHook<decltype(cuModuleGetFunction)*>* cuModuleGetFunction_hook;

std::map<CUfunction, const char*> g_func_names;

CUresult CUDAAPI cuModuleGetFunction_mine(CUfunction *hfunc,
                                          CUmodule hmod,
                                          const char *name) {
    cuModuleGetFunction_hook->Unpatch();
    CUresult ret = cuModuleGetFunction_hook->orig_fn()(hfunc, hmod, name);
    cuModuleGetFunction_hook->Patch();

    fprintf(stderr, "[-] cuModuleGetFunction: name=%s ret=%d\n", name, ret);
    g_func_names[*hfunc] = name;
    return ret;
}

CUresult CUDAAPI cuLaunchKernel_mine(CUfunction f,
                                     unsigned int gridDimX,
                                     unsigned int gridDimY,
                                     unsigned int gridDimZ,
                                     unsigned int blockDimX,
                                     unsigned int blockDimY,
                                     unsigned int blockDimZ,
                                     unsigned int sharedMemBytes,
                                     CUstream hStream,
                                     void **kernelParams,
                                     void **extra) {
    cuLaunchKernel_hook->Unpatch();
    CUresult ret = cuLaunchKernel_hook->orig_fn()(
        f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
    cuLaunchKernel_hook->Patch();

    auto found = g_func_names.find(f);
    assert(found != g_func_names.end());
    const char* name = found->second;
    fprintf(stderr, "[-] cuLaunchKernel: name=%s\n", name);
    return ret;
}

__attribute__((constructor))
static void Init() {
    fprintf(stderr, "[-] Init cutrace\n");
    cuLaunchKernel_hook = new FunctionHook<decltype(cuLaunchKernel)*>("cuLaunchKernel", cuLaunchKernel_mine);
    cuModuleGetFunction_hook = new FunctionHook<decltype(cuModuleGetFunction)*>("cuModuleGetFunction", cuModuleGetFunction_mine);
}

}  // namespace
