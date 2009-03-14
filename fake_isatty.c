#define _GNU_SOURCE

#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>

static int (*libc_isatty)(int fd) = NULL;

__attribute__((constructor)) static void init() {
    libc_isatty = (int (*)(int))dlsym(RTLD_NEXT, "isatty");
}

int isatty(int fd) {
    usleep(1);

    if (!libc_isatty) return 0;
    char buf[256];
    sprintf(buf, "/proc/self/fd/%d", fd);
    char fdp[256];
    int sz = readlink(buf, fdp, 255);
    if (sz <= 0 || sz == 255) return libc_isatty(fd);
    fdp[sz] = '\0';
    if (strncmp(fdp, "pipe:", 5)) return libc_isatty(fd);

    pid_t mypid = getpid();
    pid_t mygrp = getpgrp();
    DIR* dir = opendir("/proc");
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        int pid = atoi(ent->d_name);
        if (!pid || pid == mypid) continue;
        pid_t grp = getpgid(pid);
        if (grp == mygrp) {
            char exe[256];
            sprintf(buf, "/proc/%d/exe", pid);
            int sz2 = readlink(buf, exe, 255);
            if (sz2 <= 0 || sz2 == 255) continue;
            exe[sz2] = '\0';
            //printf("EXE %s=>%s\n", buf, exe);
            if (strcmp(exe, "/usr/bin/lv") &&
                strcmp(exe, "/usr/local/stow/w3m/bin/w3m")) {
                continue;
            }
            return 1;
#if 0
            sprintf(buf, "/proc/%d/fd", pid);
            DIR* dir2 = opendir(buf);
            struct dirent* ent2;
            while ((ent2 = readdir(dir2)) != NULL) {
                int fd2 = atoi(ent2->d_name);
                if (!fd2) continue;
                char buf2[256];
                sprintf(buf2, "%s/%d", buf, fd2);
                char fdp2[256];
                int sz2 = readlink(buf2, fdp2, 255);
                if (sz2 <= 0 || sz2 == 255) continue;
                fdp2[sz2] = '\0';
                //printf("--%s-- ++%s++ %d\n", fdp, fdp2, strcmp(fdp, fdp2));
                if (!strcmp(fdp, fdp2)) {
                    /*
                    char exe[256];
                    sprintf(buf2, "/proc/%d/exe", pid);
                    sz2 = readlink(buf2, exe, 255);
                    perror(buf2);
                    if (sz2 <= 0 || sz2 == 255) continue;
                    buf2[sz2] = '\0';
                    printf("EXE %s %s %s=%s\n", fdp, fdp2, buf2, exe);
                    if (!strcmp(exe, "/usr/bin/lv")) {
                        return 1;
                    }
                    */
                    return 1;
                }
            }
#endif
        }
    }
    return libc_isatty(fd);
}
