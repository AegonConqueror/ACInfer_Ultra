#include "warp_affine.h"

namespace yolo {
    void AffineMatrix::compute(const int i_width, const int i_height, const int d_width, const int d_height) {
        float scale = std::min(float(d_width) / i_width, float(d_height) / i_height);

        i2d[0] = i2d[4] = scale;
        i2d[1] = i2d[3] = .0;
        i2d[2] = -scale * i_width * 0.5 + d_width * 0.5;
        i2d[5] = -scale * i_height * 0.5 + d_height * 0.5;


        d2i[0] = d2i[4] = 1. / scale;
        d2i[1] = d2i[3] = .0;
        d2i[2] = -1 * i2d[2] / scale;
        d2i[5] = -1 * i2d[5] / scale;
    }
} // namespace yolo

namespace dwpose {
    void AffineMatrix::xyxy2cs(const cv::Rect& bbox, const int input_w, const int input_h, float padding) {
        int x1 = bbox.x;
        int y1 = bbox.y;
        int x2 = bbox.x + bbox.width;
        int y2 = bbox.y + bbox.height;

        float cx = (x1 + x2) * 0.5f;
        float cy = (y1 + y2) * 0.5f;
        float w = (x2 - x1) * padding;
        float h = (y2 - y1) * padding;

        center = cv::Point2f(cx, cy);

        float aspect_ratio = (float)input_w / (float)input_h;
        if (w > aspect_ratio * h) {
            h = w / aspect_ratio;
        } else if (w < aspect_ratio * h) {
            w = h * aspect_ratio;
        }
        scale = cv::Point2f(w, h);
    }

    void AffineMatrix::getAffineTransform(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst) {
        cv::Mat M;
        const double x1 = src[0].x, y1 = src[0].y;
        const double x2 = src[1].x, y2 = src[1].y;
        const double x3 = src[2].x, y3 = src[2].y;

        // 3x3 矩阵 M0 = [[x, y, 1], ...] 的行列式
        const double C11 = (y2 - y3);
        const double C12 = (x3 - x2);
        const double C13 = (x2 * y3 - x3 * y2);

        const double C21 = (y3 - y1);
        const double C22 = (x1 - x3);
        const double C23 = (x3 * y1 - x1 * y3);

        const double C31 = (y1 - y2);
        const double C32 = (x2 - x1);
        const double C33 = (x1 * y2 - x2 * y1);

        const double det = x1 * C11 + y1 * C12 + 1.0 * C13;
        if (std::fabs(det) < 1e-12) {
            i2d[0] = 0; i2d[1] = 0; i2d[2] = 0; 
            i2d[3] = 0; i2d[4] = 0; i2d[5] = 0; 
        };

        // inv(M0) = (1/det) * adj(M0) = (1/det) * C^T
        const double inv00 = C11 / det, inv01 = C21 / det, inv02 = C31 / det;
        const double inv10 = C12 / det, inv11 = C22 / det, inv12 = C32 / det;
        const double inv20 = C13 / det, inv21 = C23 / det, inv22 = C33 / det;

        // 右侧向量 u = [x'_1, x'_2, x'_3]^T, v = [y'_1, y'_2, y'_3]^T
        const double u1 = dst[0].x, u2 = dst[1].x, u3 = dst[2].x;
        const double v1 = dst[0].y, v2 = dst[1].y, v3 = dst[2].y;

        // [a, b, tx]^T = inv(M0) * u
        const double a  = inv00 * u1 + inv01 * u2 + inv02 * u3;
        const double b  = inv10 * u1 + inv11 * u2 + inv12 * u3;
        const double tx = inv20 * u1 + inv21 * u2 + inv22 * u3;

        // [c, d, ty]^T = inv(M0) * v
        const double c  = inv00 * v1 + inv01 * v2 + inv02 * v3;
        const double d  = inv10 * v1 + inv11 * v2 + inv12 * v3;
        const double ty = inv20 * v1 + inv21 * v2 + inv22 * v3;

        i2d[0] = a; i2d[1] = b; i2d[2] = tx;
        i2d[3] = c; i2d[4] = d; i2d[5] = ty;

        double dev = a * d - b * c;

        if (std::fabs(dev) < 1e-12){
            std::fill(d2i, d2i+6, 0.0f);
        }
        else{
            double inv = 1.0 / dev;
            d2i[0] =  d * inv;
            d2i[1] = -b * inv;
            d2i[2] = (b * ty - d * tx) * inv;
            d2i[3] = -c * inv;
            d2i[4] =  a * inv;
            d2i[5] = (c * tx - a * ty) * inv;
        }
    }

    void AffineMatrix::compute(const cv::Rect& bbox, float rot , const int d_width, const int d_height) {
            xyxy2cs(bbox, d_width, d_height);

            float rot_rad = rot * CV_PI / 180.0;
            // 分别定义源点和目标点
            float src_w = scale.x;
            float src_h = scale.y;

            // 方向向量
            cv::Point2f src_dir = cv::Point2f(0.0, -0.5 * src_w);
            cv::Point2f dst_dir = cv::Point2f(0.0, -0.5 * d_width);

            // 旋转
            float sn = sin(rot_rad);
            float cs = cos(rot_rad);

            cv::Point2f src_rotated = cv::Point2f(
                src_dir.x * cs - src_dir.y * sn,
                src_dir.x * sn + src_dir.y * cs
            );

            cv::Point2f dst_rotated = dst_dir; // 目标方向固定

            // 三个点确定仿射
            std::vector<cv::Point2f> src_pts(3);
            std::vector<cv::Point2f> dst_pts(3);

            src_pts[0] = center;
            src_pts[1] = center + src_rotated;
            // 第三个点是求垂直方向
            cv::Point2f src_third = center + cv::Point2f(-src_rotated.y, src_rotated.x);
            src_pts[2] = src_third;

            cv::Point2f dst_center = cv::Point2f(d_width / 2.0, d_height / 2.0);
            dst_pts[0] = dst_center;
            dst_pts[1] = dst_center + dst_rotated;
            dst_pts[2] = dst_center + cv::Point2f(-dst_rotated.y, dst_rotated.x);

            getAffineTransform(src_pts, dst_pts);
        }
} // namespace dwpose