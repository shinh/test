#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void check(int r, const char* msg) {
  if (r < 0) {
    perror(msg);
    exit(1);
  }
}

#define CHECK(e) check(e, #e)

int main(int argc, char* argv[]) {
  int fds[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
  if (!fork()) {
    close(fds[0]);
    CHECK(dup2(fds[1], 0));
    CHECK(dup2(fds[1], 1));
    argv++;
    CHECK(execv(argv[0], argv));
  }

  close(fds[1]);
  char* buf = malloc(1000 * 1000 * 10);
  int buf_size = 0;
  while (1) {
    fd_set rd, wr;
    FD_ZERO(&rd);
    FD_ZERO(&wr);
    FD_SET(0, &rd);
    FD_SET(fds[0], &rd);
    if (buf_size)
      FD_SET(fds[0], &wr);
    CHECK(select(fds[0]+1, &rd, &wr, 0, 0));

    if (FD_ISSET(fds[0], &wr)) {
      int r = write(fds[0], buf, buf_size);
      check(r, "write to sock");
      if (r == 0) {
        fprintf(stderr, "write to sock broken\n");
        break;
      }
      buf += r;
      buf_size -= r;
      continue;
    }

    if (FD_ISSET(0, &rd)) {
      int r = read(0, buf + buf_size, 4096);
      check(r, "read from stdin");
      if (r == 0) {
        fprintf(stderr, "stdin closed\n");
        break;
      }
      buf_size += r;
    }

    if (FD_ISSET(fds[0], &rd)) {
      char b[4096];
      int r = read(fds[0], b, 4096);
      check(r, "read from sock");
      if (r == 0) {
        fprintf(stderr, "read from sock finished\n");
        break;
      }
      CHECK(write(1, b, r));
    }
  }
}
