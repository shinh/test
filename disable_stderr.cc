#include <assert.h>
#include <fcntl.h>
#include <unistd.h>

#include <iostream>

struct NullStderrScope {
    NullStderrScope() {
        null_fd = open("/dev/null", O_WRONLY);
        assert(0 != null_fd);
        backup_fd = dup(STDERR_FILENO);
        assert(0 <= backup_fd);
        assert(STDERR_FILENO == dup2(null_fd, STDERR_FILENO));
    }

    ~NullStderrScope() {
        assert(STDERR_FILENO == dup2(backup_fd, STDERR_FILENO));
        assert(0 == close(backup_fd));
        assert(0 == close(null_fd));
    }

    int null_fd;
    int backup_fd;
};

int main() {
    std::cerr << "start\n";
    {
        NullStderrScope null_stderr;
        std::cerr << "test\n";
    }
    std::cerr << "end\n";
}
