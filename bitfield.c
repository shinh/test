#define CHAR_BIT 8
typedef unsigned long st_data_t;
//#define ST_INDEX_BITS (sizeof(st_data_t) * CHAR_BIT)
#define ST_INDEX_BITS (sizeof(long long) * CHAR_BIT)
typedef struct {
    st_data_t s : 3;
    //st_data_t s : ST_INDEX_BITS - 1;
} S;
