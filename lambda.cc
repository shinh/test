#include <algorithm>
#include <vector>
#include <iostream>
#include <boost/lambda/lambda.hpp>
using namespace std;
using namespace boost::lambda;
int main() {
    vector<int*> p;
    p.push_back(0);
    for_each(p.begin(), p.end(), cout<<_1);
//    for_each(p.begin(), p.end(), cout<<_1<<endl);
}

