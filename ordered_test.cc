#include "ordered.h"

#include <iostream>
#include <string>

void test_map() {
    OrderedMap<std::string, int> m;
    printf("=== insert ===\n");
    {
        auto p = m.insert({"hoge", 42});
        printf("%s %d %d\n", p.first->first.c_str(), p.first->second, p.second);
    }
    {
        auto p = m.insert({"hoge", 42});
        printf("%s %d %d\n", p.first->first.c_str(), p.first->second, p.second);
    }
    {
        auto p = m.insert({"fuga", 99});
        printf("%s %d %d\n", p.first->first.c_str(), p.first->second, p.second);
    }

    printf("=== loop ===\n");
    for (auto& p : m) {
        printf("%s %d\n", p.first.c_str(), p.second);
    }

    printf("=== find ===\n");
    {
        auto found = m.find("hoge");
        if (found != m.end()) {
            printf("lookup ok: %s %d\n", found->first.c_str(), found->second);
        }
    }
    {
        auto found = m.find("hogee");
        printf("lookup should fail: %d\n", found == m.end());
    }

    printf("=== count ===\n");
    printf("hoge count %d\n", m.count("hoge"));
    printf("hogee count %d\n", m.count("hogee"));
}

void test_set() {
    OrderedSet<std::string> s;
    printf("=== insert ===\n");
    {
        auto p = s.insert("hoge");
        printf("%s %d\n", p.first->c_str(), p.second);
    }
    {
        auto p = s.insert("hoge");
        printf("%s %d\n", p.first->c_str(), p.second);
    }
    {
        auto p = s.insert("fuga");
        printf("%s %d\n", p.first->c_str(), p.second);
    }

    printf("=== loop ===\n");
    for (auto& v : s) {
        printf("%s\n", v.c_str());
    }

    printf("=== find ===\n");
    {
        auto found = s.find("hoge");
        if (found != s.end()) {
            printf("lookup ok: %s\n", found->c_str());
        }
    }
    {
        auto found = s.find("hogee");
        printf("lookup should fail: %d\n", found == s.end());
    }

    printf("=== count ===\n");
    printf("hoge count %d\n", s.count("hoge"));
    printf("hogee count %d\n", s.count("hogee"));
}

int main() {
    test_map();
    test_set();
}
