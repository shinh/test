#include <stdio.h>

void Func1() {
 char a = 0x10;  // Initialized                                              

 printf("Func1()\n");
#ifdef NOPTR
 printf("  %x\n", a);
#else
 printf("  %p %x\n", &a, a);
#endif
}

void Func2() {
 char a;  // Not initialized                                                  
 char b;  // Not initialized                                                  

 printf("Func2()\n");
#ifdef NOPTR
 printf("  %x\n", a);
 printf("  %x\n", b);
#else
 printf("  %p %x\n", &a, a);
 printf("  %p %x\n", &b, b);
#endif
}

int main(int argc, char* argv[]) {
 Func1();
 Func2();
 return 0;
}
