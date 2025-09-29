#ifndef ACINFER_ULTRA_WARP_AFFINE_H
#define ACINFER_ULTRA_WARP_AFFINE_H

#include <opencv2/opencv.hpp>

namespace yolo {
    struct AffineMatrix {
        float i2d[6];
        float d2i[6];

        void compute(const int i_width, const int i_height, const int d_width, const int d_height);
    };
} // namespace yolo

namespace dwpose {
    struct AffineMatrix {
        float i2d[6];
        float d2i[6];

        cv::Point2f center;
        cv::Point2f scale;

        void xyxy2cs(const cv::Rect& bbox, const int input_w, const int input_h, float padding = 1.25f);

        void getAffineTransform(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst);

        void compute(const cv::Rect& bbox, float rot , const int d_width, const int d_height);
    };
    typedef std::vector<AffineMatrix> AffineMatries;
} // namespace dwpose


#endif // ACINFER_ULTRA_WARP_AFFINE_H