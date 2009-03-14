#include <stdio.h>

class A {
public:
    A() {
        printf("hello\n");
    }
    A(const A& a) {
        printf("hello\n");
    }
};
A DoNothing ( A object)
{
		 return object;
}

main()
{
		 A object;
		 DoNothing(object);
}
