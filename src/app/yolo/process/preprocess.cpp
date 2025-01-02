
#include "preprocess.h"
#include "utils/utils.h"

LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio) {
    if (img.channels() != 3) {
        LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    float img_width   = img.cols;
    float img_height  = img.rows;

    int letterbox_width  = 0;
    int letterbox_height = 0;

    LetterBoxInfo info;
    int padding_hor = 0;
    int padding_ver = 0;

    if (img_width / img_height > wh_ratio) {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        info.pad = (letterbox_height - img_height) / 2.f;
        padding_hor = 0;
        padding_ver = info.pad;
    } else {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        info.pad = (letterbox_width - img_width) / 2.f;
        padding_hor = info.pad;
        padding_ver = 0;
    }

    cv::copyMakeBorder(
        img, img_letterbox, 
        padding_ver, padding_ver, padding_hor, padding_hor, 
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
    );
    return info;
}