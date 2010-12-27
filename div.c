#include <stdlib.h>
main(){
    div_t r = div(42,5);
    printf("%d %d\n", r.quot, r.rem);
}
