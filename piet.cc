#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

using namespace std;

int g_cs = 1;
const char* g_filename;

typedef unsigned char Color;
typedef unsigned int uint;

enum {
  LR, R, DR,
  LY, Y, DY,
  LG, G, DG,
  LC, C, DC,
  LB, B, DB,
  LM, M, DM,
  WHITE, BLACK,
  DONE = 32
};

struct Pos {
  Pos(int x0, int y0)
      : x(x0), y(y0) {
  }
  int x, y;
};

unsigned char COLOR_TABLE[20][3] = {
  { 0xff, 0xc0, 0xc0 },
  { 0xff, 0x00, 0x00 },
  { 0xc0, 0x00, 0x00 },

  { 0xff, 0xff, 0xc0 },
  { 0xff, 0xff, 0x00 },
  { 0xc0, 0xc0, 0x00 },

  { 0xc0, 0xff, 0xc0 },
  { 0x00, 0xff, 0x00 },
  { 0x00, 0xc0, 0x00 },

  { 0xc0, 0xff, 0xff },
  { 0x00, 0xff, 0xff },
  { 0x00, 0xc0, 0xc0 },

  { 0xc0, 0xc0, 0xff },
  { 0x00, 0x00, 0xff },
  { 0x00, 0x00, 0xc0 },

  { 0xff, 0xc0, 0xff },
  { 0xff, 0x00, 0xff },
  { 0xc0, 0x00, 0xc0 },

  { 0xff, 0xff, 0xff },
  { 0x00, 0x00, 0x00 },
};

char* vformat(const char* fmt, va_list ap) {
  char buf[256];
  vsnprintf(buf, 255, fmt, ap);
  buf[255] = 0;
  return strdup(buf);
}

char* format(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char* r = vformat(fmt, ap);
  va_end(ap);
  return r;
}

static void error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char* r = vformat(fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", r);
  exit(1);
}

class Ppm {
 public:
  explicit Ppm(const char* filename)
      : filename_(filename) {
    char buf[72] = {};
    FILE* fp = fopen(filename, "rb");
    if (!fgets(buf, 71, fp))
      error("Failed to read PPM type");
    if (strcmp(buf, "P6\n"))
      error("Only P6 PPM is supported");
    if (!fgets(buf, 71, fp))
      error("Failed to read comment");

    if (!fgets(buf, 71, fp))
      error("Failed to read width and height");
    if (sscanf(buf, "%d%d", &w_, &h_) != 2)
      error("Broken width and height");

    if (!fgets(buf, 71, fp))
      error("Failed to read max value");

    c_ = (Color*)malloc(w_ * h_);
    for (uint y = 0; y < h_; y++) {
      for (uint x = 0; x < w_; x++) {
        char v[3];
        if (fread(v, 1, 3, fp) != 3)
          error("Premature end");
        Color c = -1;
        for (int i = 0; i < 20; i++) {
          if (v[0] == COLOR_TABLE[i][0] &&
              v[1] == COLOR_TABLE[i][1] &&
              v[2] == COLOR_TABLE[i][2]) {
            c = i;
            break;
          }
        }
        if (c == -1) {
          error("Unexpected color %d %d %d", v[0], v[1], v[2]);
        }
        c_[y * w_ + x] = c;
      }
    }
  }

  uint w() { return w_; }
  uint h() { return h_; }
  Color c(uint x, uint y) {
    return c_[y*w_+x];
  }
  void set(uint x, uint y, Color c) {
    c_[y*w_+x] = c;
  }

 private:
  const char* filename_;
  uint w_, h_;
  Color* c_;
};

void FindConnected(Ppm* p, uint x, uint y, Color c, vector<Pos>* ps) {
  if (x >= p->w() || y >= p->h())
    return;
  if (p->c(x, y) != c)
    return;
  ps->push_back(Pos(x, y));
  p->set(x, y, c | DONE);
  FindConnected(p, x+1, y, c, ps);
  FindConnected(p, x-1, y, c, ps);
  FindConnected(p, x, y+1, c, ps);
  FindConnected(p, x, y-1, c, ps);
}

void Translate(Ppm* p) {
  puts("#include <stdio.h>");
  puts("#include <stdlib.h>");
  puts("int main() {");

  vector<Pos> ps;
  for (uint y = 0; y < p->h(); y++) {
    for (uint x = 0; x < p->w(); x++) {
      Color c = p->c(x, y);
      if ((c & DONE) || c == WHITE || c == BLACK)
        continue;

      ps.clear();
      FindConnected(p, x, y, c, &ps);

      Pos mpos[8];
      for (int i = 0; i < 8; i++)
        mpos[i].x = ps[0];
      for (size_t i = 1; i < ps.size(); i++) {
        Pos p = ps[i];
        if (p.x > mpos[0].x && p.y < mpos[0].y)
          mpos[0] = p;
        if (p.x > mpos[1].x && p.y > mpos[1].y)
          mpos[1] = p;
        if (p.x > mpos[2].x && p.y > mpos[2].y)
          mpos[2] = p;
        if (p.x < mpos[3].x && p.y > mpos[3].y)
          mpos[3] = p;
        if (p.x < mpos[4].x && p.y > mpos[4].y)
          mpos[4] = p;
        if (p.x < mpos[5].x && p.y < mpos[5].y)
          mpos[5] = p;
        if (p.x < mpos[6].x && p.y < mpos[6].y)
          mpos[6] = p;
        if (p.x > mpos[7].x && p.y < mpos[7].y)
          mpos[7] = p;
      }

      bool is_exit = true;
      bool ok[8];
      for (int dpcc = 0; dpcc < 8; dpcc++) {
        int dp = dpcc / 2;
        int cc = dpcc % 2;
        Pos mp = mpos[dpcc];
        switch (dp) {
          case 0: mp.x++; break;
          case 1: mp.y++; break;
          case 2: mp.x--; break;
          case 3: mp.y--; break;
          default: error("dp=%d", dp);
        }
        ok[dpcc] = (mp.x < ppm->w() && mp.y < ppm->h() &&
                    ppm->c(mp.x, mp.y) != BLACK);
        if (ok[dpcc])
          is_exit = false;
      }

      if (is_exit) {
        for (int dpcc = 0; dpcc < 8; dpcc++) {
          int dp = dpcc / 2;
          int cc = dpcc % 2;
          for (size_t i = 0; i < ps.size(); i++) {
            Pos p = ps[i];
            printf(".L_%d_%d_%d_%d:\n", p.x, p.y, dp, cc);
          }
        }
        printf("exit(0);");
      }

      for (int dpcc = 0; dpcc < 8; dpcc++) {
        int dp = dpcc / 2;
        int cc = dpcc % 2;
        for (size_t i = 0; i < ps.size(); i++) {
          Pos p = ps[i];
          printf(".L_%d_%d_%d_%d:\n", p.x, p.y, dp, cc);
        }

        int fdp = dp;
        int fcc = cc;
        while (!ok[fdp * 2 + fcc]) {
          fcc = (fcc + 1) % 2;
          if (fcc == cc)
            fdp = (fdp + 1) % 4;
        }

        if (dp != fdp || cc != fcc) {
          printf("goto .L_%d_%d_%d_%d;", p.x, p.y, fdp, fcc);
          continue;
        }

        
      }
    }
  }

  puts("}");
}

void usage(const char* arg0) {
  fprintf(stderr, "Usage: %s <img.ppm>\n", arg0);
  exit(1);
}

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];
    if (!strcmp(arg, "-cs")) {
      g_cs = strtol(argv[++i], nullptr, 10);
    } else {
      if (g_filename) {
        usage(argv[0]);
      }
      g_filename = arg;
    }
  }
  if (!g_filename) {
    usage(argv[0]);
  }

  Ppm* ppm = new Ppm(g_filename);
  Translate(ppm);
}
