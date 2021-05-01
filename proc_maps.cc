// Example output:
//

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char* argv[]) {
    char buf[999];
    const char* pid = argc == 1 ? "self" : argv[1];

    sprintf(buf, "/proc/%s/maps", pid);
    FILE* maps_fp = fopen(buf, "rb");
    if (!maps_fp) {
        fprintf(stderr, "Cannot open %s: %s\n", buf, strerror(errno));
        exit(1);
    }

    sprintf(buf, "/proc/%s/pagemap", pid);
    FILE* pm_fp = fopen(buf, "rb");
    if (!pm_fp) {
        fprintf(stderr, "Cannot open %s: %s\n", buf, strerror(errno));
        exit(1);
    }

    while (fgets(buf, 998, maps_fp)) {
        unsigned long begin;
        unsigned long end;
        char r, w, x, p;
        long offset, inode;
        int major, minor;
        char name[999];
        name[0] = 0;
        sscanf(buf, "%lx-%lx %c%c%c%c %lx %d:%d %ld%s",
               &begin, &end, &r, &w, &x, &p, &offset, &major, &minor, &inode,
               name);

        printf("%s", buf);

        const int page_size = 4096;
        if (fseek(pm_fp, begin / page_size * sizeof(uint64_t), SEEK_SET) != 0) {
            fprintf(stderr, "Seek failed: %s\n", strerror(errno));
            exit(1);
        }

        if (!strcmp(name, "[vsyscall]")) {
            continue;
        }

        long num_in_ram = 0;
        long num_in_swap = 0;
        long num_zero = 0;
        long num_exclusive = 0;
        long num_soft_dirty = 0;
        long size = end - begin;
        for (long i = 0; i < size / page_size; ++i) {
            uint64_t info;
            if (fread(&info, sizeof(uint64_t), 1, pm_fp) != 1) {
                fprintf(stderr, "Read failed: %s\n", strerror(errno));
                exit(1);
            }

            if ((info >> 63) & 1) {
                ++num_in_ram;
            }
            if ((info >> 62) & 1) {
                ++num_in_swap;
            }
            if ((info >> 60) & 1) {
                ++num_zero;
            }
            if ((info >> 56) & 1) {
                ++num_exclusive;
            }
            if ((info >> 55) & 1) {
                ++num_soft_dirty;
            }

#if 0
            uint64_t page_id = info & ((1ULL << 55ULL) - 1ULL);
            printf("%lx page_id=%lu in_ram=%lu\n", info, page_id, ((info >> 63) & 1));
#endif
        }

        printf("%ld %ld %ld %ld %ld %ld\n",
               size, num_in_ram, num_in_swap, num_zero, num_exclusive, num_soft_dirty);
    }
    fclose(maps_fp);
    fclose(pm_fp);
}
