#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<int>> vec_vec;
    vec_vec.push_back({});
    std::cout << vec_vec.size() << std::endl;  // 1
    vec_vec.insert(vec_vec.begin(), std::vector<int>{});
    std::cout << vec_vec.size() << std::endl;  // 2
    vec_vec.insert(vec_vec.begin(), {});
    std::cout << vec_vec.size() << std::endl;  // 2
}
