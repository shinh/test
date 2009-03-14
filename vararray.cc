#include <iostream>
#include <alloca.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    char temp[n];
    //cout << sizeof(temp) << endl;
    asm("#COM");
    cout << (void*)temp << endl;
    asm("#COM");
    //cout << (void*)alloca(300) << endl;
}
