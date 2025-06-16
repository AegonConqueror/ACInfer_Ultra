#ifndef ACINFER_ULTRA_PLUGIN_H
#define ACINFER_ULTRA_PLUGIN_H

typedef enum {
    STATUS_SUCCESS          = 0,
    STATUS_FAILURE          = 1,
    STATUS_BAD_PARAM        = 2,
    STATUS_NOT_SUPPORTED    = 3,
    STATUS_NOT_INITIALIZED  = 4
} pluginStatus_t;

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#endif // ACINFER_ULTRA_PLUGIN_H