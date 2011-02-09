#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned char uchar;
typedef unsigned int uint32;

uint32 k[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32 rr(uint32 v, int n) {
    return (v >> n) | (v << (32 - n));
}

inline uint32 swap32(uint32 v) {
    return (v << 24) | (v << 8) & 0x00FF0000 | (v >> 8) & 0x0000FF00 | v >> 24;
}

void sha256_next(uint32* ptr, uint32* h) {
    uint32 w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = swap32(ptr[i]);
    }

    for (int i = 16; i < 64; i++) {
        uint32 s0 = rr(w[i-15], 7) ^ rr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32 s1 = rr(w[i-2], 17) ^ rr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32 a[8];
    for (int i = 0; i < 8; i++) {
        a[i] = h[i];
    }

    for (int i = 0; i < 64; i++) {
        uint32 s0 = rr(a[0], 2) ^ rr(a[0], 13) ^ rr(a[0], 22);
        uint32 maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
        uint32 t2 = s0 + maj;
        uint32 s1 = rr(a[4], 6) ^ rr(a[4], 11) ^ rr(a[4], 25);
        uint32 ch = (a[4] & a[5]) ^ ((~a[4]) & a[6]);
        uint32 t1 = a[7] + s1 + ch + k[i] + w[i];

        a[7] = a[6];
        a[6] = a[5];
        a[5] = a[4];
        a[4] = a[3] + t1;
        a[3] = a[2];
        a[2] = a[1];
        a[1] = a[0];
        a[0] = t1 + t2;
    }

    for (int i = 0; i < 8; i++) {
        h[i] += a[i];
    }
}

void sha256(void* ptr, long len, void* out) {
    uchar* p = (uchar*)ptr;
    uint32* h = (uint32*)out;
    h[0] = 0x6a09e667;
    h[1] = 0xbb67ae85;
    h[2] = 0x3c6ef372;
    h[3] = 0xa54ff53a;
    h[4] = 0x510e527f;
    h[5] = 0x9b05688c;
    h[6] = 0x1f83d9ab;
    h[7] = 0x5be0cd19;

    long l;
    for (l = 0; l + 64 <= len; l += 64) {
        sha256_next((uint32*)(p + l), h);
    }

    uchar b[64];
    int i;
    for (i = 0; l < len; i++, l++) {
        b[i] = p[l];
    }
    if (i < 56) {
        b[i] = 128;
    } else {
        assert(i < 64);
        b[i] = 128;
        for (i++; i < 64; i++) {
            b[i] = 0;
        }
        sha256_next((uint32*)b, h);

        i = -1;
    }

    for (i++; i < 56; i++) {
        b[i] = 0;
    }
    unsigned long long ll = (unsigned long long)len * 8;
    for (int i = 0; i < 8; i++) {
        b[56 + i] = ((uchar*)&ll)[7 - i];
    }
    sha256_next((uint32*)b, h);

    for (int i = 0; i < 8; i++) {
        h[i] = swap32(h[i]);
    }
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        exit(1);
    }

    FILE* fp = fopen(argv[1], "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc(size);
    fread(buf, 1, size, fp);

    uchar md[32];
    sha256(buf, size, md);
    for (int i = 0; i < 32; i++) {
        printf("%02x", md[i]);
    }
    puts("");
}
