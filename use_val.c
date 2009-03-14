#include <stdio.h>
extern int val;
int mine;
int main() {
    printf("%d\n", val);
    val = 30;
    printf("%d\n", val);
    change_val();
    printf("%d\n", val);
    printf("%p %p\n", &val, &mine);
}
