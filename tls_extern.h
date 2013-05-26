#ifdef TLS_MODEL
#define ATTR_TLS_MODEL __attribute__((tls_model(TLS_MODEL)))
#else
#define ATTR_TLS_MODEL
#endif
extern __thread int tls ATTR_TLS_MODEL;
int tls_use();
