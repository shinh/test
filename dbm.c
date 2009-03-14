#define DB_DBM_HSEARCH  1

#include <db.h>
#include <fcntl.h>

int main() {
    //dbm_close(dbm_open("hoge.db", O_RDWR|O_CREAT, 0));
    printf("%p\n", dbm_open("no such file", O_RDWR, 0));
    //dbm_close(dbm_open("hoge.db", O_RDWR, 0));
}
