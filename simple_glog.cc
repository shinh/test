#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <string>

enum LogLevel {
  INFO, FATAL
};

class LogStream {
 public:
  explicit LogStream(LogLevel l)
      : l_(l) {
  }

  ~LogStream() {
    fprintf(stderr, "LOG! %s\n", ss_.str().c_str());
    if (l_ == FATAL)
      abort();
  }

  template <class T>
  LogStream& operator<<(T v) {
    ss_ << v;
  }

 private:
  std::ostringstream ss_;
  LogLevel l_;
};

#define LOG(l) LogStream(l)

int main() {
  std::ostringstream oss;
  LOG(INFO) << "foo" << 42;
  LOG(FATAL) << "crash!";
}
