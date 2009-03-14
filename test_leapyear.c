#include <stdio.h>
extern int leap(int x);
int cleap(int y) {
    return y%4==0 ? (y%100==0 ? (y%400==0 ? 1 : 0) : 1) : 0;
}
int main() {
#define TEST(y) printf("%d %d\n", y, leap(y))
    TEST(1999);
    TEST(2000);
    TEST(2001);
    TEST(2002);
    TEST(2003);
    TEST(2004);
    TEST(2005);
    TEST(2100);
    TEST(2200);
    TEST(2300);
    TEST(2400);
}
