#define A A
#define B A
#define C B
#define D E
#define E D

#define STRING_HELPER(x) #x
#define STRING(x) STRING_HELPER(x)

int main() {
    puts(STRING(A));
    puts(STRING(B));
    puts(STRING(C));
    puts(STRING(D));
    puts(STRING(E));
}
