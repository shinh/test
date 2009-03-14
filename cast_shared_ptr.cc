#include <boost/shared_ptr.hpp>

using namespace boost;

int main() {
    shared_ptr<int> i = new int;
    *i = 3;
    return i + 1;
}
