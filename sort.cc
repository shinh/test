#include <vector>
#include <set>
#include <queue>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <functional>
using namespace std;

template <class T, template <class> class L, template <class> class C>
class priqueue {
};

int main() {
    cout << "vec\n";
    vector<int> v;
    v.push_back(3);
    v.push_back(2);
    v.push_back(4);
    sort(v.begin(), v.end());
    copy(v.begin(), v.end(), ostream_iterator<int>(cout, "\n"));

    cout << "set\n";
    set<int> s;
    s.insert(3);
    s.insert(2);
    s.insert(4);
    copy(s.begin(), s.end(), ostream_iterator<int>(cout, "\n"));

    cout << "pq\n";
    priority_queue<int> q;
    q.push(3);
    q.push(2);
    q.push(4);
    while (!q.empty()) {
        cout << q.top() << endl;
        q.pop();
    }
}
