#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

int main() {
    struct rusage ru;
    printf("%d\n", getrusage(RUSAGE_SELF, &ru));
}
