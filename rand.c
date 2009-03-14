#include <limits.h>
main(){
    int i,r;
    for(i=0;i<=INT_MAX;i++){
        srand(i);
        r = rand();
        if (r <= 100) {
            printf("%d %d\n",i,r);
        }
    }
}
