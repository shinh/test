#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>

int main() {
    char buf[256];
    printf("%d\n", readlink("/proc/self/fd/1", buf, 255));
    printf("%s\n", buf);

    printf("PID=%d\n", getpid());

    pid_t parent = getppid();
    printf("PPID=%d\n", parent);
    sprintf(buf, "/proc/%d/task", parent);
    DIR* dir = opendir(buf);
    struct dirent* ent;
    /*
    while ((ent = readdir(dir)) != NULL) {
        printf("%s\n", ent->d_name);
    }
    */

    pid_t mygrp = getpgrp();
    dir = opendir("/proc");
    while ((ent = readdir(dir)) != NULL) {
        int pid = atoi(ent->d_name);
        if (!pid) continue;
        pid_t grp = getpgid(pid);
        if (grp == mygrp) {
            printf("%d %d\n", pid, grp);
        }
    }

    printf("tcgetpgrp(1) = %d\n", tcgetpgrp(1));
}
