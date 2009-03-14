#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int fd = -1;

int main() {
    fd = open("/tmp/w", O_RDONLY);
}
