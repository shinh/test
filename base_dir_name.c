#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// You can check the default (POSIX for glibc) behavior by removing
// this line.
#define TEST_BOTH

#if defined(basename) && defined(TEST_BOTH)
# undef basename
# define posix_basename __xpg_basename
char* basename(char* path);
#endif

char* my_basename(char* path) {
  static char cur[2] = ".";
  if (!path || !*path)
    return cur;

  char* path_end = path + strlen(path) - 1;
  char* endp = path_end;
  while (*endp == '/') {
    if (path == endp)
      return path_end;
    endp--;
  }
  endp[1] = 0;

  char* beginp = endp;
  while (beginp != path) {
    beginp--;
    if (*beginp == '/') {
      beginp++;
      break;
    }
  }
  return beginp;
}

char* my_dirname(char* path) {
  static char cur[2] = ".";
  if (!path || !*path)
    return cur;

  char* endp = path + strlen(path) - 1;
  while (*endp == '/') {
    if (path == endp)
      return path;
    endp--;
  }
  endp[1] = 0;

  char* beginp = endp;
  while (*beginp != '/') {
    if (beginp == path)
      return cur;
    *beginp = 0;
    beginp--;
  }

  while (beginp != path) {
    if (beginp[-1] != '/')
      break;
    beginp--;
  }
  if (beginp != path)
    *beginp = 0;
  return path;
}

void show(const char* orig, const char* mod) {
  char buf[17];
  char* p = buf + sprintf(buf, "%s ", mod);
  if (abs(mod - orig) > 4000) {
    strcpy(p, "heap");
  } else {
    sprintf(p, "%d", (mod - orig));
  }
  printf("%-13s", buf);
}

void test(const char* cp) {
  char* p = strdup(cp);
  show(p, p);

  strcpy(p, cp);
  show(p, basename(p));

#ifdef posix_basename
  strcpy(p, cp);
  show(p, __xpg_basename(p));
#endif

  strcpy(p, cp);
  show(p, my_basename(p));

  strcpy(p, cp);
  show(p, dirname(p));

  strcpy(p, cp);
  show(p, my_dirname(p));

  free(p);
  puts("");
}

int main() {
#ifdef posix_basename
  printf("%-013s%-013s%-013s%-013s%-013s%-013s\n",
         "path", "base(g)", "base(p)", "base(my)", "dir", "dir(my)");
#else
  printf("%-013s%-013s%-013s%-013s%-013s\n",
         "path", "base", "base(my)", "dir", "dir(my)");
#endif
  test("/usr/lib");
  test("usr/lib");
  test("/usr/");
  test("//usr//");
  test("usr/");
  test("usr");
  test(".");
  test("..");
  test("/");
  test("//");
  test("");
  test("//u//s//");
  return 0;
}
