#include <stdio.h>
#include <stdlib.h>

int main(int c, char *v[]) {
  int n = 0, y, m, y0 = atoi(v[1])-2000, m0 = atoi(v[2]);
  if (m0 < 3) { y0--; m0 += 12; }
  for (y = y0; y < 14; y++)
    for (m = 3; m < 15; m++) {
      if (y == y0 && (m < m0 || (m == m0 && atoi(v[3]) > 13))) continue;
      if (!((26*-~m/10+y+y/4)%7)) {
      if (!((26*(m+1)/10+y+y/4)%7)) {
        printf("'%02d年%d月13日\n", (m>=13)?y+1:y, (m>=13)?m-12:m);
        n++;
      }
    }
  printf("総数：%d個\n", n);

  return 0;
}
