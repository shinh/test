#define DECLARE_MALLOC(n)                          \
  void* n ## _malloc(size_t size);                 \
  void n ## _free(void* ptr);                      \
  void* n ## _calloc(size_t nmemb, size_t size);   \
  void* n ## _realloc(void* ptr, size_t size);

#define DEFINE_CALLOC(n)                            \
  void* n ## _calloc(size_t nmemb, size_t size) {   \
    size *= nmemb;                                  \
    void* r = n ## _malloc(size);                   \
    memset(r, 0, size);                             \
    return r;                                       \
  }

#define DEFINE_REALLOC(n)                        \
  void* n ## _realloc(void *ptr, size_t size) {  \
    void* r = n ## _malloc(size);                \
    memcpy(r, ptr, size);                        \
    n ## _free(ptr);                             \
    return r;                                    \
  }

extern "C" {
DECLARE_MALLOC(__libc)
}
DECLARE_MALLOC(kr)
DECLARE_MALLOC(mmap)
DECLARE_MALLOC(leak)
