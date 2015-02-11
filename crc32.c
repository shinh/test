#include <pthread.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>

uint32_t crc_table[256];
const int NUM_THREADS = 8;

void make_crc_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
        }
        crc_table[i] = c;
    }
}

uint32_t crc32(uint8_t* buf, size_t len) {
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        c = crc_table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFF;
}

uint32_t crc32_uint32(uint32_t v) {
  uint32_t c = 0xFFFFFFFF;
  for (int i = 0; i < 8; i++) {
    uint8_t b = "0123456789abcdef"[(v >> (4 * (7 - i))) & 15];
    c = crc_table[(c ^ b) & 0xFF] ^ (c >> 8);
  }
  c = crc_table[(c ^ 10) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFF;
}

uint32_t crc32_uint32_asm(uint32_t v) {
  uint32_t c = 0xFFFFFFFF;
  for (int i = 0; i < 8; i++) {
    uint8_t b = "0123456789abcdef"[(v >> (4 * (7 - i))) & 15];
    c = _mm_crc32_u8(c, b);
  }
  //c = _mm_crc32_u32(c, v);
  c = _mm_crc32_u8(c, 10);
  return c ^ 0xFFFFFFFF;
}

void* crc32_thread(void* data) {
  int id = (int)data;
  for (int i = 0; i < 4294967296ULL / NUM_THREADS; i++) {
    uint32_t v = (4294967296ULL / NUM_THREADS) * id + i;
    //uint32_t r = crc32_uint32(v);
    uint32_t r = crc32_uint32_asm(v);
    if (v == r) {
      printf("hit. %x\n", v);
    }
  }
  return NULL;
}

int main() {
  make_crc_table();

  if (1) {
    pthread_t th[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_create(&th[i], NULL, &crc32_thread, (void*)i);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(th[i], NULL);
    }
  } else {
    for (uint32_t i = 0; i < 4294967295U; i++) {
      uint32_t r = crc32_uint32(i);
      if (i == r) {
        printf("hit. %x\n", i);
      }
    }
  }
}
