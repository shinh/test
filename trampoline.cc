#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

void* MakeTrampoline(void* target, int prelude_size) {
    char* trampoline = (char*)mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(trampoline, target, prelude_size);
#if defined(__x86_64__)
    char* p = trampoline + prelude_size;
    // mov r11, <target after prelude>
    p[0] = 0x49;
    p[1] = 0xbb;
    *(char**)(p + 2) = (char*)target + prelude_size;
    p[10] = 0x41;
    p[11] = 0xff;
    p[12] = 0xe3;
#elif defined(__aarch64__)
    uint32_t* p = (uint32_t*)(trampoline + prelude_size);
    uint64_t t = (uint64_t)target + prelude_size;
    // mov     x17, #<target>
    p[0] = 0xd2800011 | ((t >> 0) & 0xffff) << 5;
    // movk    x17, #<target>, lsl #16
    p[1] = 0xf2a00011 | ((t >> 16) & 0xffff) << 5;
    // movk    x17, #<target>, lsl #32
    p[2] = 0xf2c00011 | ((t >> 32) & 0xffff) << 5;
    // movk    x17, #<target>, lsl #48
    p[3] = 0xf2e00011 | ((t >> 48) & 0xffff) << 5;
    // br x17
    p[4] = 0xd61f0220;
#else
# error "Unsupported architecture"
#endif

#if defined(__arm__) || defined(__aarch64__)
    __builtin___clear_cache(trampoline, trampoline + 4096);
#endif
    return trampoline;
}

#ifdef TEST

#include <stdio.h>

int ShowMessage(const char* ptr) {
    fprintf(stderr, "msg: %s\n", ptr);
    return 42;
}

int main() {
    auto fp = (decltype(&ShowMessage))MakeTrampoline((void*)&ShowMessage, 8);
    if (fp("hoge") != 42) {
        fprintf(stderr, "fail\n");
        return 1;
    }
}

#endif
