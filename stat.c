#include <stdio.h>
#include <sys/stat.h>

int main(int argc, const char* argv[]) {
    struct stat st;
    if (stat(argv[1], &st) != 0) {
        perror("stat");
        return 1;
    }

    printf("%ld %ld\n", st.st_atim.tv_sec, st.st_atim.tv_nsec);
    printf("%ld %ld\n", st.st_mtim.tv_sec, st.st_mtim.tv_nsec);
    printf("%ld %ld\n", st.st_ctim.tv_sec, st.st_ctim.tv_nsec);
}
