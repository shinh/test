#include <dirent.h>
#include <stdio.h>

const char* type_str(unsigned char type) {
    switch (type) {
    case DT_BLK:
        return "block device";
    case DT_CHR:
        return "character device";
    case DT_DIR:
        return "directory";
    case DT_FIFO:
        return "fifo";
    case DT_LNK:
        return "symlink";
    case DT_REG:
        return "regular file";
    case DT_SOCK:
        return "socket";
    case DT_UNKNOWN:
        return "unknown";
    default:
        return "???";
    }
}

int main() {
    DIR* dir = opendir(".");
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        printf("%30s\t%s(%d)\n",
               ent->d_name, type_str(ent->d_type), ent->d_type);
    }
}
