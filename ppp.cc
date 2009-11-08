struct S {
    static void f() {}
};
struct OK : S {};
class NOK : S {};

int main() {
    OK::f();
    NOK::f();
}
