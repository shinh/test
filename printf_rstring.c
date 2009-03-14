#include <ruby.h>

#define RSTRING_LENPTR(str) (int)RSTRING_LEN(str), RSTRING_PTR(str)

int main() {
    ruby_init();
    VALUE str = rb_str_new2("hoge-");
    printf("%.*s\n", RSTRING_LENPTR(str));
    //printf("%.*s\n", RSTRING(str)->as.heap);
}
