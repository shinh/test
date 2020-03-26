#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int ScanMemory(const char* map_file, const void* needle, size_t needle_size) {
    char buf[4000];
    FILE* fp = fopen("/proc/self/maps", "rb");
    int cnt = 0;
    while (fgets(buf, 3999, fp)) {
        uintptr_t begin, end;
        char r, w, x, pp;
        long offset, inode;
        int major, minor;
        char name[999];
        name[0] = 0;
        sscanf(buf, "%lx-%lx %c%c%c%c %lx %d:%d %ld%s",
               &begin, &end,
               &r, &w, &x, &pp, &offset, &major, &minor, &inode,
               name);

        if (r != 'r') continue;
        if (name[0] == '[' && name[1] == 'v') continue;

        void* p = (void*)begin;
        for (;;) {
            size_t size = end - (uintptr_t)p;
            void* found = memmem(p, size, needle, needle_size);
            if (found == NULL) break;
            if (found != needle) {
                ++cnt;
                fprintf(stderr, "Found at %p in %s\n", found, name);
            }
            p = found + needle_size;
        }
    }
    fclose(fp);
    return cnt;
}

#ifdef TEST

int main() {
    printf("found %d\n", ScanMemory("/proc/self/maps", "maps", 4));
}

#endif
