#include <stdio.h>

#include <pcl.h>

#define CO_STACK_SIZE (800 * 1024)

void spawn1(void* arg) {
    int i = 0;
    int fiber = (int)arg;
    printf("1 %d %d\n", fiber, i+=2);
    co_resume();
    printf("4 %d %d\n", fiber, i+=2);
    co_resume();
    while (i < 1000000) {
        i++;
        co_resume();
    }
    printf("spawn1 finished\n");
}

void spawn2(void* arg) {
    int fiber = (int)arg;
    int i = 0;
    printf("2 %d %d\n", fiber, i++);
    co_resume();
    printf("5 %d %d\n", fiber, i++);
    co_resume();
    while (i < 1000000) {
        i++;
        co_resume();
    }
    printf("spawn2 finished\n");
}

int main() {
    coroutine_t co1, co2;

    co1 = co_create(&spawn1, (void*)1, NULL, CO_STACK_SIZE);
    co2 = co_create(&spawn2, (void*)2, NULL, CO_STACK_SIZE);

    printf("ok?\n");

    co_call(co1);
    co_call(co2);
    printf("3\n");
    co_call(co1);
    co_call(co2);

    int i;
    for (i = 0; i < 1000000-4; i++) {
        co_call(co1);
        co_call(co2);
    }

    co_delete(co1);
    co_delete(co2);
}
