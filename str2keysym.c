#include <X11/X.h>
#include <X11/keysym.h>

int main() {
    printf("%d %d\n", XK_BackSpace, XStringToKeysym("BackSpace"));
}
