#include <stdio.h>
#include <time.h>

int main() {
    int data[100];
    int i, j, t, n = 100;

    for (i = 0; i < n; i++) {
        data[i] = i;
    }

    clock_t st = clock();
    for (t = 0; t < 1000000; t++) {
#if 0
        // yaneurao's insertion sort
        for (i = 1; i < n; i++)
        {
            int tmp = data[i];
            if (data[i-1] > tmp)
            {
                j = i;
                do {
                    data[j] = data[j-1];
                    --j;
                } while ( j > 0 && data[j-1] > tmp);
                data[j] = tmp;
            }
        }
#else
        for (i = 1; i < n; i++) {
            int tmp = data[i];
            for (j = i; data[j-1] > tmp;) {
                data[j] = data[j-1];
                j--;
                if (j <= 0) break;
            }
            data[j] = tmp;
        }
#endif
    }

    printf("%d\n", (int)(clock() - st));
}
