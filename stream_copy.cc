#include <iostream>
#include <fstream>
using namespace std;
int main() {
    ifstream ifs("tmp");
    for (char c; ifs.get(c); ) {
        cout<<c;
        char d;
        cin.get(d);
    }
    //for (char c; cin.get(c); ) cout<<c;
}
