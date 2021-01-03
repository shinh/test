#pragma once

#include <map>
#include <vector>

namespace internal {

template <typename Key, typename Item, class ItemToKey>
class OrderedContainer {
public:
    using value_type = Item;
    using iterator = typename std::vector<Item>::iterator;
    using const_iterator = typename std::vector<Item>::const_iterator;

    iterator begin() { return items_.begin(); }
    iterator end() { return items_.end(); }
    const_iterator begin() const { return items_.begin(); }
    const_iterator end() const { return items_.end(); }

    std::pair<iterator, bool> insert(const value_type& v) {
        auto p = index_.emplace(ItemToKey()(v), items_.size());
        if (p.second) {
            items_.push_back(v);
            return std::make_pair(std::prev(end()), true);
        } else {
            return std::make_pair(begin() + p.first->second, false);
        }
    }

    const_iterator find(const Key& k) {
        auto found = index_.find(k);
        if (found == index_.end()) {
            return end();
        } else {
            return begin() + found->second;
        }
    }

    int count(const Key& k) {
        return index_.count(k);
    }

private:
    std::vector<Item> items_;
    std::map<Key, int> index_;
};

template <typename Key, typename Value>
struct PairToKey {
    Key operator()(const std::pair<Key, Value>& kv) {
        return kv.first;
    }
};

template <typename Key>
struct Identity {
    Key operator()(const Key& k) {
        return k;
    }
};

}  // namespace internal

template <typename Key, typename Value>
class OrderedMap : public internal::OrderedContainer<Key, std::pair<Key, Value>, internal::PairToKey<Key, Value>> {
};

template <typename Key>
class OrderedSet : public internal::OrderedContainer<Key, Key, internal::Identity<Key>> {
};
