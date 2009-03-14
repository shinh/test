#include <string>
#include <set>
#include <iostream>
using namespace std;

int main() {
    set<string> s;
    s.insert("A");
    s.insert("B.");
    cout << *s.lower_bound("B") << endl;
}
