// A repro to expose a bug in cuDNN's logging. This SEGV with
//
// CUDNN_LOGDEST_DBG=stderr CUDNN_LOGINFO_DBG=1
//
// Changing `max_seq_length` to a smaller value tells us it is
// iterating over seqLengthArray maxSeqLength times instead of
// batchSize time.
//
// I! CuDNN (v7300) function cudnnSetRNNDataDescriptor() called:
// i!     dataType: type=cudnnDataType_t; val=CUDNN_DATA_FLOAT (0);
// i!     layout: type=cudnnRNNDataLayout_t; val=CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED (0);
// i!     maxSeqLength: type=int; val=10;
// i!     batchSize: type=int; val=3;
// i!     vectorSize: type=int; val=10;
// i!     seqLengthArray: type=int; val=[1,1,1,-843385600,1816135666,1339328848,21923,1110940567,32541,1138875264];
// i!     paddingFill: type=CUDNN_DATA_FLOAT; val=0.000000;
// i! Time: 2018-11-16T14:13:43.062329 (0d+0h+0m+0s since start)
// i! Process=14030; Thread=14030; GPU=NULL; Handle=NULL; StreamId=NULL.
//

#include <stdio.h>
#include <cudnn.h>

void check_cudnn(cudnnStatus_t status, const char* msg, int lineno) {
    if (CUDNN_STATUS_SUCCESS != status) {
        fprintf(stderr, "CUDNN error: %s at line %d\n",
                cudnnGetErrorString(status), lineno);
    }
}

#define CHECK_CUDNN(expr) check_cudnn(expr, #expr, __LINE__)

int main() {
    int max_seq_length = 10000;
    // int max_seq_length = 10;
    int batch_size = 3;
    int input_size = 10;
    float pad = 0.0;
    int seq_lengths[] = {1, 1, 1};

    cudnnRNNDataDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&desc));
    CHECK_CUDNN(cudnnSetRNNDataDescriptor(
                    desc,
                    CUDNN_DATA_FLOAT,
                    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                    max_seq_length,
                    batch_size,
                    input_size,
                    seq_lengths,
                    &pad));
}
