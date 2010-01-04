#include <dirent.h>
#include <stdio.h>

int main() {
    DIR* dir = opendir(".");
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        printf("%s %d\n", ent->d_name, ent->d_type);
    }
}
