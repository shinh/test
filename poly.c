#include <stdio.h>

typedef void (*cmd_t)();

void cmd_a() {
    printf("A");
}

void cmd_b() {
    printf("B");
}

int main() {
    cmd_t cmds[3];
    cmds[0] = &cmd_a;
    cmds[1] = &cmd_b;
    cmds[2] = &cmd_a;
    int i;
    for (i = 0; i < 3; i++)
        cmds[i]();
    puts("");
}
