
#include "preprocess_warp_affine.h"

namespace ACKernel {

    Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ColorType channel_type){

		Norm out;
		out.type  = NormType::MeanStd;
		out.alpha = alpha;
		out.cvt = channel_type;
		memcpy(out.mean, mean, sizeof(out.mean));
		memcpy(out.std,  std,  sizeof(out.std));
		return out;
	}

	Norm Norm::alpha_beta(float alpha, float beta, ColorType channel_type){

		Norm out;
		out.type = NormType::AlphaBeta;
		out.alpha = alpha;
		out.beta = beta;
		out.cvt = channel_type;
		return out;
	}

	Norm Norm::None(){
		return Norm();
	}

    __global__ void warp_affine_and_normalize_kernel(
        const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
        float* __restrict__ dst,         int dst_pitch, int dst_w, int dst_h,
        float* __restrict__ M_d2i,       Norm* norm,    uint8_t fill_value,
        int N
    ) {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        int b  = blockIdx.z;
        if (b >= N || dx >= dst_w || dy >= dst_h) return;

        const float* M = M_d2i + b * 6;
        float m0 = M[0], m1 = M[1], m2 = M[2];
        float m3 = M[3], m4 = M[4], m5 = M[5];

        float fx = __fmaf_rn(m0, (float)dx, __fmaf_rn(m1, (float)dy, m2));
        float fy = __fmaf_rn(m3, (float)dx, __fmaf_rn(m4, (float)dy, m5));

        int x0 = __float2int_rd(fx);
        int y0 = __float2int_rd(fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float lx = fx - (float)x0;
        float ly = fy - (float)y0;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float w00 = hy * hx;
        float w01 = hy * lx;
        float w10 = ly * hx;
        float w11 = ly * lx;

        int x0c = max(0, min(src_w - 1, x0));
        int x1c = max(0, min(src_w - 1, x1));
        int y0c = max(0, min(src_h - 1, y0));
        int y1c = max(0, min(src_h - 1, y1));

        const uint8_t* r0 = src + y0c * src_pitch;
        const uint8_t* r1 = src + y1c * src_pitch;

        const uint8_t* p00 = r0 + x0c * 3;
        const uint8_t* p01 = r0 + x1c * 3;
        const uint8_t* p10 = r1 + x0c * 3;
        const uint8_t* p11 = r1 + x1c * 3;

        float m00 = (x0 >= 0 && x0 < src_w && y0 >= 0 && y0 < src_h) ? 1.0f : 0.0f;
        float m01 = (x1 >= 0 && x1 < src_w && y0 >= 0 && y0 < src_h) ? 1.0f : 0.0f;
        float m10 = (x0 >= 0 && x0 < src_w && y1 >= 0 && y1 < src_h) ? 1.0f : 0.0f;
        float m11 = (x1 >= 0 && x1 < src_w && y1 >= 0 && y1 < src_h) ? 1.0f : 0.0f;

        float sw = w00*m00 + w01*m01 + w10*m10 + w11*m11;
        float f  = (float)fill_value;

        float B = f * (1.0f - sw) + w00*m00 * p00[0] + w01*m01 * p01[0] + w10*m10 * p10[0] + w11*m11 * p11[0];
        float G = f * (1.0f - sw) + w00*m00 * p00[1] + w01*m01 * p01[1] + w10*m10 * p10[1] + w11*m11 * p11[1];
        float R = f * (1.0f - sw) + w00*m00 * p00[2] + w01*m01 * p01[2] + w10*m10 * p10[2] + w11*m11 * p11[2];

        float c0, c1, c2;

        if (norm->cvt == ColorType::RGB)    { c0 = R; c1 = G; c2 = B; }
        else                                { c0 = B; c1 = G; c2 = R; }

        if(norm->type == NormType::MeanStd){
			c0 = (c0 * norm->alpha - norm->mean[0]) / norm->std[0];
			c1 = (c1 * norm->alpha - norm->mean[1]) / norm->std[1];
			c2 = (c2 * norm->alpha - norm->mean[2]) / norm->std[2];
		}else if(norm->type == NormType::AlphaBeta){
			c0 = c0 * norm->alpha + norm->beta;
			c1 = c1 * norm->alpha + norm->beta;
			c2 = c2 * norm->alpha + norm->beta;
		}

        // -> NCHW
        size_t HW  = (size_t)dst_w * dst_h;
        size_t idx = (size_t)dy * dst_w + dx;

        float* out_base = dst + b * dst_pitch;
        out_base[0*HW + idx] = c0;
        out_base[1*HW + idx] = c1;
        out_base[2*HW + idx] = c2;
    }

    void warp_affine_and_normalize_batchM_invoker(
        uint8_t* src, int src_pitch, int src_w, int src_h,
        float* dst, int dst_pitch, int dst_w, int dst_h,
        float* d_M_d2i, Norm* norm, uint8_t fill_value,
        int N
    ) {
        int thread_size = 32;
        dim3 block(thread_size, thread_size, 1);
        dim3 grid(
            (dst_w + block.x - 1) / block.x, 
            (dst_h + block.y - 1) / block.y,
            N
        );

        checkCudaKernel(
            warp_affine_and_normalize_kernel<<<grid, block, 0, nullptr>>>(
                src, src_pitch, src_w, src_h,
                dst, dst_pitch, dst_w, dst_h,
                d_M_d2i, norm, fill_value, N
            )
        );
    }
} // namespace ACKernel

