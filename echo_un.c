#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

int main() {
    int sock;
    struct sockaddr_un addr;

    if ((sock = socket(PF_UNIX, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        return 1;
    }

    {
        int v, l = 4;
        printf("%d\n", getsockopt(sock, SOL_SOCKET, SO_SNDLOWAT, &v, &l));
        printf("%d\n", v);
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = PF_UNIX;
    strcpy(addr.sun_path, "/tmp/hoge");
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    if (listen(sock, 1) < 0) {
        perror("listen");
        return 1;
    }

    while (1) {
        struct sockaddr_un dummy;
        int fd;
        int len = sizeof(dummy);

        if ((fd = accept(sock, (struct sockaddr*)&dummy, &len)) < 0) {
            perror("accept");
            return 1;
        }

        while (1) {
            int ret;
            char c;
            ret = read(fd, &c, 1);
            if (ret < 0) {
                perror("read");
                return 1;
            }
            else if (ret == 0) {
                break;
            }
            write(fd, &c, 1);
        }
        close(fd);
    }
}
