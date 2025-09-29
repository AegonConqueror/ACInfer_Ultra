#include "ac_draw.h"

#include <random>

namespace iDraw {

    static const unsigned char colors[80][3] = {
        {56,  0,   255}, {226, 255, 0  }, {0,   94,  255}, {0,   37,  255}, {0,   255, 94},
        {255, 226, 0  }, {0,   18,  255}, {255, 151, 0  }, {170, 0,   255}, {0,   255, 56},
        {255, 0,   75 }, {0,   75,  255}, {0,   255, 169}, {255, 0,   207}, {75,  255, 0  },
        {207, 0,   255}, {37,  0,   255}, {0,   207, 255}, {94,  0,   255}, {0,   255, 113},
        {255, 18,  0  }, {255, 0,   56 }, {18,  0,   255}, {0,   255, 226}, {170, 255, 0  },
        {255, 0,   245}, {151, 255, 0  }, {132, 255, 0  }, {75,  0,   255}, {151, 0,   255},
        {0,   151, 255}, {132, 0,   255}, {0,   255, 245}, {255, 132, 0  }, {226, 0,   255},
        {255, 37,  0  }, {207, 255, 0  }, {0,   255, 207}, {94,  255, 0  }, {0,   226, 255},
        {56,  255, 0  }, {255, 94,  0  }, {255, 113, 0  }, {0,   132, 255}, {255, 0,   132},
        {255, 170, 0  }, {255, 0,   188}, {113, 255, 0  }, {245, 0,   255}, {113, 0,   255},
        {255, 188, 0  }, {0,   113, 255}, {255, 0,   0  }, {0,   56,  255}, {255, 0,   113},
        {0,   255, 188}, {255, 0,   94 }, {255, 0,   18 }, {18,  255, 0  }, {0,   255, 132},
        {0,   188, 255}, {0,   245, 255}, {0,   169, 255}, {37,  255, 0  }, {255, 0,   151},
        {188, 0,   255}, {0,   255, 37 }, {0,   255, 0  }, {255, 0,   170}, {255, 0,   37},
        {255, 75,  0  }, {0,   0,   255}, {255, 207, 0  }, {255, 0,   226}, {255, 245, 0  },
        {188, 255, 0  }, {0,   255, 18 }, {0,   255, 75 }, {0,   255, 151}, {255, 56,  0  },
    };

    void draw_box(cv::Mat &img, const cv::Rect_<float> &box, const int class_id) {
        const unsigned char* color = colors[class_id % 80];
        cv::Scalar box_color(color[0], color[1], color[2]);
        cv::Scalar text_color(255, 255, 255);

        int lw = std::max(static_cast<int>(round((img.rows + img.cols) / 2.0 * 0.002)), 2);
        int tf = std::max(lw - 1, 1);
        float sf = lw / 3.0;
        
        cv::rectangle(img, box, box_color, lw, cv::LINE_AA);
    }

    void draw_box_label(
        cv::Mat &img, const cv::Rect_<float> box, 
        const int class_id, const float conf,
        const std::vector<std::string> &class_names
    ) {
        const unsigned char* color = colors[class_id % 80];
        cv::Scalar box_color(color[0], color[1], color[2]);
        cv::Scalar text_color(255, 255, 255);
        
        int lw = std::max(static_cast<int>(round((img.rows + img.cols) / 2.0 * 0.002)), 2);
        int tf = std::max(lw - 1, 1);
        float sf = lw / 3.0;
        cv::rectangle(img, box, box_color, lw, cv::LINE_AA);
        char draw_string[100];
        if (class_names.empty()) {
            sprintf(draw_string, "id%02d: %.2f", class_id, conf);
        } else {
            std::string class_name  = class_names[class_id];
            sprintf(draw_string, "%s: %.2f", class_name.c_str(), conf);
        }
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(draw_string, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &baseline);
        int w = text_size.width;
        int h = text_size.height;
        bool outside = box.y - h >= 3;
        cv::Point text_origin = outside ? cv::Point(box.x, box.y - h - 3) : cv::Point(box.x, box.y + h + 3);
        cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(box.x + w, text_origin.y), box_color, -1, cv::LINE_AA);
        cv::Point t = outside ? cv::Point(box.x, box.y - 2) : cv::Point(box.x, box.y + h + 2);
        cv::putText(img, draw_string, t, cv::FONT_HERSHEY_SIMPLEX, sf, text_color, tf, cv::LINE_AA);
    }

    void draw_mask(cv::Mat &img, const cv::Rect_<float> &box, cv::Mat &roi) {
        cv::Mat seg_mask_item = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        cv::Scalar seg_color(dis(gen), dis(gen), dis(gen));

        seg_mask_item(box).setTo(seg_color, roi);
        cv::addWeighted(img, 1, seg_mask_item, 0.45, 0, img);
    }
} // namespace iDraw