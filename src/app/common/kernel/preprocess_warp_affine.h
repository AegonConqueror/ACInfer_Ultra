/**
 * *****************************************************************************
 * File name:   preprocess_warp_affine.h
 * 
 * @brief  自定义预处理kernel
 * 
 * 
 * Created by Aegon on 2024-09-18
 * Copyright © 2024 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_PREPROCESS_WARP_AFFINE_H
#define ACINFER_ULTRA_PREPROCESS_WARP_AFFINE_H

#include <string>
#include <cuda.h>
#include "trt/trt_cuda.h"

namespace ACKernel {
    enum class NormType  : int { None = 0, MeanStd = 1, AlphaBeta = 2 };
    enum class ColorType : int { BGR  = 0, RGB     = 1 };

    struct Norm{
        float mean[3];
        float std[3];
        float alpha;
        float beta;
        NormType type;
        ColorType cvt;

        static Norm mean_std(const float mean[3], const float std[3], float alpha, ColorType cvt);

        static Norm alpha_beta(float alpha, float beta, ColorType channel_type);

        static Norm None();
    };

    void warp_affine_and_normalize_batchM_invoker(
        uint8_t* src, int src_pitch, int src_w, int src_h,
        float* dst, int dst_pitch, int dst_w, int dst_h,
        float* d_M_d2i, Norm* norm, uint8_t fill_value,
        int N = 1
    );
} // namespace ACKernel


#endif // ACINFER_ULTRA_PREPROCESS_WARP_AFFINE_H