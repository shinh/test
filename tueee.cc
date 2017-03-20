// Thue interepreter.

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <ext/rope>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace std;

typedef unsigned char byte;

#define LOG(...) do {                                                   \
    fprintf(stderr, __VA_ARGS__);                                       \
  } while (0)

#define ERROR(...) do {                                                 \
    fprintf(stderr, __VA_ARGS__);                                       \
    exit(1);                                                            \
  } while (0)

class SimpleTimer {
 public:
  explicit SimpleTimer(const string& name)
      : name_(name), elapsed_(0) {
  }

  ~SimpleTimer() {
    LOG("%s: %f\n", name_.c_str(), (double)elapsed_ / CLOCKS_PER_SEC);
  }

  void start() {
    start_ = clock();
  }

  void stop() {
    elapsed_ += clock() - start_;
  }

 private:
  const string name_;
  clock_t elapsed_;
  clock_t start_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(SimpleTimer* timer)
      : timer_(timer) {
    timer_->start();
  }

  ~ScopedTimer() {
    timer_->stop();
  }

 private:
  SimpleTimer* timer_;
};

class StringPiece {
 public:
  typedef const char* iterator;

  StringPiece() : ptr_(nullptr), len_(0) {
  }

  StringPiece(const char* ptr, size_t len)
      : ptr_(ptr), len_(len) {
  }

  iterator begin() const {
    return ptr_;
  }

  iterator end() const {
    return ptr_ + len_;
  }

 private:
  const char* ptr_;
  size_t len_;
};

class StringSlice {
 public:
  class iterator {
    friend StringSlice;

   public:
    iterator() = default;
    iterator(const StringSlice* self, size_t pos)
        : self_(self), pos_(pos) {
    }

    char operator*() const { return (*self_->str_)[pos_]; }

    iterator& operator++() { pos_++; return *this; }
    iterator& operator+=(size_t o) { pos_ += o; return *this; }

    bool operator==(const iterator& r) const { return pos_ == r.pos_; }
    bool operator!=(const iterator& r) const { return pos_ != r.pos_; }

    size_t pos() const { return pos_; }

   private:
    const StringSlice* self_;
    size_t pos_;
  };

  StringSlice() : str_(nullptr), begin_(0), end_(0) {
  }

  StringSlice(string* str, size_t begin, size_t end)
      : str_(str), begin_(begin), end_(end) {
  }

  iterator begin() const {
    return iterator(this, begin_);
  }

  iterator end() const {
    return iterator(this, end_);
  }

 private:
  const string* str_;
  size_t begin_;
  size_t end_;
};

class Parser {
 public:
  explicit Parser(const char* file)
      : filename(file),
        timer_("parse") {
    ScopedTimer scoped_timer(&timer_);
    char* buf = readFile(file);
    char* p = buf;
    while (true) {
      if (*p == 0) {
        ERROR("Premature end of a script: %s\n", file);
      }

      char* eol = p + strcspn(p, "\r\n");
      *eol = 0;
      while (eol[1] == '\r' || eol[1] == '\n') {
        eol++;
      }

      char* sep = strstr(p, "::=");
      if (!sep) {
        ERROR("Malformed production: \"%s\"!\n", p);
      }
      *sep = '\0';
      sep += 3;

      const string& lhs = p;
      if (lhs.find_first_not_of(" \t\v") == string::npos) {
        for (p = eol + 1; *p; p++) {
          if (*p != '\r' && *p != '\n')
            data += *p;
        }
        LOG("Initial:  \"%s\"\n", data.c_str());
        break;
      }

      LOG("Rule: %s => %s\n", lhs.c_str(), sep);
      rules.emplace_back(lhs, sep);
      p = eol + 1;
    }
    free(buf);
  }

  const string filename;
  vector<pair<string, string>> rules;
  string data;

 private:
  char* readFile(const char* file) {
    FILE* fp = fopen(file, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc(size + 1);
    fread(buf, 1, size, fp);
    buf[size] = '\0';
    return buf;
  }

  SimpleTimer timer_;
};

template <class S>
typename S::iterator str_begin(S& s) {
  return s.begin();
}
template <>
__gnu_cxx::crope::iterator str_begin<__gnu_cxx::crope>(__gnu_cxx::crope& s) {
  return s.mutable_begin();
}

template <class S>
typename S::iterator str_end(S& s) {
  return s.end();
}
template <>
__gnu_cxx::crope::iterator str_end<__gnu_cxx::crope>(__gnu_cxx::crope& s) {
  return s.mutable_end();
}

template <class str>
class SimpleRuleMatcher {
 public:
  typedef typename str::iterator iterator;

  explicit SimpleRuleMatcher(const vector<string>& rules)
      : rules_(rules),
        timer_("match") {
  }

  void match(str& s, vector<pair<size_t, iterator>>* out) {
    ScopedTimer scoped_timer(&timer_);
    for (size_t i = 0; i < rules_.size(); i++) {
      const string& rule = rules_[i];
      iterator found = search(str_begin(s), str_end(s),
                              rule.begin(), rule.end());
      if (found != str_end(s)) {
        out->emplace_back(i, found);
      }
    }
  }

 private:
  const vector<string> rules_;
  SimpleTimer timer_;
};

template <class str>
class PrefixRuleMatcher {
 public:
  typedef typename str::iterator iterator;

  explicit PrefixRuleMatcher(const vector<string>& rules)
      : timer_("match") {
    for (size_t i = 0; i < rules.size(); i++) {
      const string& rule = rules[i];
      assert(!rule.empty());
      byte c = rule[0];
      rules_[c].emplace_back(i, rule);
      prefixes_.push_back(rule[0]);
    }

    sort(prefixes_.begin(), prefixes_.end());
    prefixes_.resize(
        unique(prefixes_.begin(), prefixes_.end()) - prefixes_.begin());
    LOG("prefixes: %s\n", prefixes_.c_str());
  }

  void match(str& s, vector<pair<size_t, iterator>>* out) {
    ScopedTimer scoped_timer(&timer_);
    for (iterator iter = str_begin(s);; ++iter) {
      iter = find_first_of(iter, str_end(s),
                           prefixes_.begin(), prefixes_.end());
      if (iter == str_end(s))
        break;

      byte c = *iter;
      assert(!rules_[c].empty());
      for (const auto& p : rules_[c]) {
        const string& rule = p.second;
        size_t i = 0;
        for (iterator iter2 = iter; iter2 != str_end(s); ++iter2) {
          if (rule[i] != *iter2)
            break;
          i++;
          if (i == rule.size()) {
            out->emplace_back(p.first, iter);
            break;
          }
        }
      }
    }
  }

 private:
  vector<pair<size_t, string>> rules_[256];
  string prefixes_;
  SimpleTimer timer_;
};

template <class str, class Matcher>
class SimpleExecutor {
  typedef typename Matcher::iterator state_iterator;

 public:
  explicit SimpleExecutor(const Parser& parser)
      : filename_(parser.filename),
        state_(parser.data.c_str()),
        mt_(random_device()()),
        stdin_marker_(":::"),
        timer_("exec") {
    vector<string> lhs;
    for (const auto& rule : parser.rules) {
      lhs.push_back(rule.first);
      rules_.emplace_back(rule.first, rule.second.c_str());
    }
    matcher_.reset(new Matcher(lhs));
  }

  void run() {
    while (true) {
      //LOG("state=%s\n", state_.c_str());
      if (!step())
        break;
    }
  }

  string state() const { return string(state_.begin(), state_.end()); }

 private:
  bool step() {
    vector<pair<size_t, state_iterator>> matches;
    matcher_->match(state_, &matches);
    if (matches.empty())
      return false;

    ScopedTimer scoped_timer(&timer_);
    size_t index = mt_() % matches.size();
    const auto& matched = matches[index];
    const auto& rule = rules_[matched.first];
    const str& rhs = rule.second;
    state_iterator found = matched.second;
    state_iterator end = found + rule.first.size();

    if (rhs == stdin_marker_) {
      string repl;
      while (true) {
        int c = getc(stdin);
        if (c == EOF || c == '\n')
          break;
        if (c == '\r') {
          c = getchar();
          if (c != '\n')
            ungetc(c, stdin);
        }
        repl += c;
      }
      state_.replace(found, end, repl.c_str(), repl.size());
    } else if (!rhs.empty() && rhs[0] == '~') {
      printf("%s\n", rhs.substr(1).c_str());
      state_.replace(found, end, "");
    } else {
      state_.replace(found, end, rhs);
    }
    return true;
  }

  const string filename_;
  vector<pair<string, str>> rules_;
  str state_;
  mt19937 mt_;
  const str stdin_marker_;
  unique_ptr<Matcher> matcher_;
  SimpleTimer timer_;
};

class FastExecutor {
  typedef PrefixRuleMatcher<StringSlice> Matcher;
  typedef typename Matcher::iterator state_iterator;

 public:
  explicit FastExecutor(const Parser& parser)
      : filename_(parser.filename),
        state_(parser.data.c_str()),
        mt_(random_device()()),
        stdin_marker_(":::"),
        max_rule_len_(0),
        timer_("exec") {
    vector<string> lhs;
    for (const auto& rule : parser.rules) {
      lhs.push_back(rule.first);
      max_rule_len_ = max(max_rule_len_, rule.first.size());
      rules_.emplace_back(rule.first, rule.second.c_str());
    }
    matcher_.reset(new Matcher(lhs));
    dirty_ = StringSlice(&state_, 0, state_.size());
  }

  void run() {
    while (true) {
      //LOG("state=%s\n", state_.c_str());
      if (!step())
        break;
    }
  }

  string state() const { return string(state_.begin(), state_.end()); }

 private:
  bool step() {
    matcher_->match(dirty_, &matches_);
    if (matches_.empty()) {
      dirty_ = StringSlice(&state_, 0, state_.size());
      matches_.clear();
      matcher_->match(dirty_, &matches_);
      assert(matches_.empty());
      return false;
    }

    ScopedTimer scoped_timer(&timer_);
    size_t index = mt_() % matches_.size();
    const auto& matched = matches_[index];
    const auto& rule = rules_[matched.first];
    const string& rhs = rule.second;
    size_t found = matched.second.pos();
    size_t len = rule.first.size();
    size_t repl_len;

    if (rhs == stdin_marker_) {
      string repl;
      while (true) {
        int c = getc(stdin);
        if (c == EOF || c == '\n')
          break;
        if (c == '\r') {
          c = getchar();
          if (c != '\n')
            ungetc(c, stdin);
        }
        repl += c;
      }
      state_.replace(found, len, repl.c_str(), repl.size());
      repl_len = repl.size();
    } else if (!rhs.empty() && rhs[0] == '~') {
      printf("%s\n", rhs.substr(1).c_str());
      state_.replace(found, len, "");
      repl_len = 0;
    } else {
      state_.replace(found, len, rhs);
      repl_len = rhs.size();
    }

    // Safe but slow
#if 0
    matches_.clear();
    dirty_ = StringSlice(&state_, 0, state_.size());
#else
    ssize_t diff = repl_len - len;
    for (size_t i = 0; i < matches_.size(); ) {
      auto& p = matches_[i];
      size_t start = p.second.pos();
      size_t end = start + rules_[p.first].first.size();
      if ((found < start && found + len > start) ||
          (found >= start && found < end)) {
        matches_[i] = matches_.back();
        matches_.pop_back();
        continue;
      } else if (found < start) {
        p.second += diff;
        assert(p.second.pos() < state_.size());
      }
      i++;
    }
    ssize_t dirty_start = found - diff - max_rule_len_;
    size_t dirty_end = found + repl_len + max_rule_len_;
    if (dirty_start < 0) {
      dirty_start = 0;
    }
    if (dirty_end >= state_.size()) {
      dirty_end = state_.size();
    }
    dirty_ = StringSlice(&state_, dirty_start, dirty_end);
#endif
    return true;
  }

  const string filename_;
  vector<pair<string, string>> rules_;
  string state_;
  StringSlice dirty_;
  mt19937 mt_;
  const string stdin_marker_;
  unique_ptr<Matcher> matcher_;
  vector<pair<size_t, state_iterator>> matches_;
  size_t max_rule_len_;
  SimpleTimer timer_;
};

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <prog.t>\n", argv[0]);
    return 1;
  }
  Parser parser(argv[1]);
#if 0
  typedef __gnu_cxx::crope rope;
  //SimpleExecutor<rope, SimpleRuleMatcher<rope>> executor(parser);
  SimpleExecutor<rope, PrefixRuleMatcher<rope>> executor(parser);
#elif 0
  SimpleExecutor<string, PrefixRuleMatcher<string>> executor(parser);
#else
  FastExecutor executor(parser);
#endif
  executor.run();
  printf("Final:  \"%s\"\n", executor.state().c_str());
}
