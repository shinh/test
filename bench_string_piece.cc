// A benchmark for StringPiece
// http://src.chromium.org/viewvc/chrome/trunk/src/base/string_piece.h

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <string>

#include "string_piece.h"

using base::StringPiece;
using namespace std;

void JoinFilePathStr(const string& dir, const string& base, string* out) {
    out->clear();
    out->append(dir);
    out->push_back('/');
    out->append(base);
}

void JoinFilePathSp(const StringPiece& dir, const StringPiece& base,
                    string* out) {
    dir.CopyToString(out);
    *out += '/';
    base.AppendToString(out);
}

#define BENCH(msg, expr) do {                                           \
        joined.clear();                                                 \
        time_t start = clock();                                         \
        for (int i = 0; i < 1000000; i++) {                             \
            expr;                                                       \
        }                                                               \
        int elapsed = clock() - start;                                  \
        assert(!strcmp(joined.c_str(), "/tmp/hoge.c"));                 \
        printf("%s %f\n", msg, (double)elapsed / CLOCKS_PER_SEC);       \
    } while (0)

int main() {
    const string& dir = "/tmp";
    const string& base = "hoge.c";
    string joined;

    BENCH("Str(const char*, const char*)",
          JoinFilePathStr("/tmp", "hoge.c", &joined));
    BENCH("Str(string, const char*)",
          JoinFilePathStr(dir, "hoge.c", &joined));
    BENCH("Str(const char*, string)",
          JoinFilePathStr("/tmp", base, &joined));
    BENCH("Str(string, string)",
          JoinFilePathStr(dir, base, &joined));

    BENCH("Sp(const char*, const char*)",
          JoinFilePathSp("/tmp", "hoge.c", &joined));
    BENCH("Sp(string, const char*)",
          JoinFilePathSp(dir, "hoge.c", &joined));
    BENCH("Sp(const char*, string)",
          JoinFilePathSp("/tmp", base, &joined));
    BENCH("Sp(string, string)",
          JoinFilePathSp(dir, base, &joined));
}

#if 0
int main() {
    const string& dir = "/tmp";
    const string& base = "hoge.c";
    string joined;

    JoinFilePathSp(dir, base, &joined);
    JoinFilePathSp("/tmp", base, &joined);
    JoinFilePathSp(dir, "hoge.c", &joined);
    JoinFilePathSp("/tmp", "hoge.c", &joined);
}
#endif
