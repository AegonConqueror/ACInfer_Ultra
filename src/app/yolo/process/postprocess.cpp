#include "postprocess.h"
#include <string.h>
#include <stdlib.h>

#include <algorithm>

#include "utils/utils.h"

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

namespace yolov8 {

    typedef struct Yolov8Rect{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int classId;
        float mask[32];
        std::vector<KeyPoint> keyPoints;
    } Yolov8Rect;

    static int headNum = 3;
    static int strides[3] = {8, 16, 32};

    static inline float fast_exp(float x) {
        union
        {
            uint32_t i;
            float f;
        } v;
        v.i = (12102203.1616540672 * x + 1064807160.56887296);
        return v.f;
    }

    float sigmoid(float x) {
        return 1 / (1 + fast_exp(-x));
    }

    static inline float IOU(
        float XMin1, float YMin1, float XMax1, float YMax1, 
        float XMin2, float YMin2, float XMax2, float YMax2
    ) {
        float Inter = 0;
        float Total = 0;
        float XMin = 0;
        float YMin = 0;
        float XMax = 0;
        float YMax = 0;
        float Area1 = 0;
        float Area2 = 0;
        float InterWidth = 0;
        float InterHeight = 0;

        XMin = ZQ_MAX(XMin1, XMin2);
        YMin = ZQ_MAX(YMin1, YMin2);
        XMax = ZQ_MIN(XMax1, XMax2);
        YMax = ZQ_MIN(YMax1, YMax2);

        InterWidth = XMax - XMin;
        InterHeight = YMax - YMin;

        InterWidth = (InterWidth >= 0) ? InterWidth : 0;
        InterHeight = (InterHeight >= 0) ? InterHeight : 0;

        Inter = InterWidth * InterHeight;

        Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
        Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

        Total = Area1 + Area2 - Inter;

        return float(Inter) / float(Total);
    }

    std::vector<float> GenerateMeshgrid(int input_w, int input_h){
        std::vector<float> meshgrid;
        for (int index = 0; index < headNum; index++) {
            for (int i = 0; i < static_cast<int>(input_w / strides[index]); i++) {
                for (int j = 0; j < static_cast<int>(input_h / strides[index]); j++) {
                    meshgrid.push_back(float(j + 0.5));
                    meshgrid.push_back(float(i + 0.5));
                }
            }
        }
        return meshgrid;
    }

    void Postprocess_POSE_float(
        float* reg, float* cls, float *ps, std::vector<float>& pose_rects, KeyPointsArray& key_points,
        int anchors, int input_w, int input_h, int class_num, int keypoint_num, float conf_thres, float nms_thres
    ) {
        static auto meshgrid = GenerateMeshgrid(input_w, input_h);

        int grid_index = -2;

        float cls_max = 0;
        int cls_index = 0;

        std::vector<Yolov8Rect> pose_results;

        for (int i = 0; i < anchors; i++) {
            grid_index += 2;
            for (int cl = 0; cl < class_num; cl++) {
                float cls_val = cls[cl * anchors + i];
                if (0 == cl) {
                    cls_max = cls_val;
                    cls_index = cl;
                } else {
                    if (cls_val > cls_max) {
                        cls_max = cls_val;
                        cls_index = cl;
                    }
                }

                if (cls_max > conf_thres) {
                    Yolov8Rect temp;

                    int head_index = (i < 6400) ? 0 : (i < 8000) ? 1 : 2;

                    float xmin = (meshgrid[grid_index + 0] - reg[0 * anchors + i]) * strides[head_index];
                    float ymin = (meshgrid[grid_index + 1] - reg[1 * anchors + i]) * strides[head_index];
                    float xmax = (meshgrid[grid_index + 0] + reg[2 * anchors + i]) * strides[head_index];
                    float ymax = (meshgrid[grid_index + 1] + reg[3 * anchors + i]) * strides[head_index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < input_w ? xmax : input_w;
                    ymax = ymax < input_h ? ymax : input_h;

                    if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h) {
                        temp.xmin = xmin / input_w;
                        temp.ymin = ymin / input_h;
                        temp.xmax = xmax / input_w;
                        temp.ymax = ymax / input_h;
                        temp.classId = cls_index;
                        temp.score = cls_max;

                        for (int kc = 0; kc < keypoint_num; kc++) {
                            KeyPoint kp;
                            kp.x = (ps[(kc * 3 + 0) * anchors + i] * 2 + (meshgrid[grid_index + 0] - 0.5)) * strides[head_index] / input_w;
                            kp.y = (ps[(kc * 3 + 1) * anchors + i] * 2 + (meshgrid[grid_index + 1] - 0.5)) * strides[head_index] / input_h;
                            kp.score = sigmoid(ps[(kc * 3 + 2) * anchors + i]);
                            kp.id = kc;
                            temp.keyPoints.push_back(kp);
                        }
                        pose_results.push_back(temp);
                    }
                }
            }
        } 

        std::sort(
            pose_results.begin(), pose_results.end(),
            [](Yolov8Rect &Rect1, Yolov8Rect &Rect2) -> bool
            { return (Rect1.score > Rect2.score); }
        );

        for (int i = 0; i < pose_results.size(); ++i) {
            float xmin1 = pose_results[i].xmin;
            float ymin1 = pose_results[i].ymin;
            float xmax1 = pose_results[i].xmax;
            float ymax1 = pose_results[i].ymax;
            int classId = pose_results[i].classId;
            float score = pose_results[i].score;

            if (classId != -1) {
                pose_rects.push_back(float(classId));
                pose_rects.push_back(float(score));
                pose_rects.push_back(float(xmin1));
                pose_rects.push_back(float(ymin1));
                pose_rects.push_back(float(xmax1));
                pose_rects.push_back(float(ymax1));

                std::map<int, KeyPoint> kps;
                
                for (int kn = 0; kn < keypoint_num; kn++) {
                    kps.insert({kn, pose_results[i].keyPoints[kn]});
                }

                key_points.push_back(kps);

                for (int j = i + 1; j < pose_results.size(); ++j) {
                    float xmin2 = pose_results[j].xmin;
                    float ymin2 = pose_results[j].ymin;
                    float xmax2 = pose_results[j].xmax;
                    float ymax2 = pose_results[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nms_thres) {
                        pose_results[j].classId = -1;
                    }
                }
            }
        }
    }

    void Postprocess_DET_float(
        float* reg, float* cls, std::vector<float>& det_rects,
        int anchors, int input_w, int input_h, int class_num,
        float conf_thres, float nms_thres
    ) {
        static auto meshgrid = GenerateMeshgrid(input_w, input_h);

        int grid_index = -2;

        float cls_max = 0;
        int cls_index = 0;

        std::vector<Yolov8Rect> det_results;

        for (int i = 0; i < anchors; i++) {
            grid_index += 2;
            for (int cl = 0; cl < class_num; cl++) {
                float cls_val = cls[cl * anchors + i];
                if (0 == cl) {
                    cls_max = cls_val;
                    cls_index = cl;
                } else {
                    if (cls_val > cls_max) {
                        cls_max = cls_val;
                        cls_index = cl;
                    }
                }

                if (cls_max > conf_thres) {
                    Yolov8Rect temp;

                    int head_index = (i < 6400) ? 0 : (i < 8000) ? 1 : 2;

                    float xmin = (meshgrid[grid_index + 0] - reg[0 * anchors + i]) * strides[head_index];
                    float ymin = (meshgrid[grid_index + 1] - reg[1 * anchors + i]) * strides[head_index];
                    float xmax = (meshgrid[grid_index + 0] + reg[2 * anchors + i]) * strides[head_index];
                    float ymax = (meshgrid[grid_index + 1] + reg[3 * anchors + i]) * strides[head_index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < input_w ? xmax : input_w;
                    ymax = ymax < input_h ? ymax : input_h;

                    if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h) {
                        temp.xmin = xmin / input_w;
                        temp.ymin = ymin / input_h;
                        temp.xmax = xmax / input_w;
                        temp.ymax = ymax / input_h;
                        temp.classId = cls_index;
                        temp.score = cls_max;
                        det_results.push_back(temp);
                    }
                }
            }
        }

        std::sort(
            det_results.begin(), det_results.end(),
            [](Yolov8Rect &Rect1, Yolov8Rect &Rect2) -> bool
            { return (Rect1.score > Rect2.score); }
        );

        for (int i = 0; i < det_results.size(); ++i) {
            float xmin1 = det_results[i].xmin;
            float ymin1 = det_results[i].ymin;
            float xmax1 = det_results[i].xmax;
            float ymax1 = det_results[i].ymax;
            int classId = det_results[i].classId;
            float score = det_results[i].score;

            if (classId != -1) {
                det_rects.push_back(float(classId));
                det_rects.push_back(float(score));
                det_rects.push_back(float(xmin1));
                det_rects.push_back(float(ymin1));
                det_rects.push_back(float(xmax1));
                det_rects.push_back(float(ymax1));

                for (int j = i + 1; j < det_results.size(); ++j) {
                    float xmin2 = det_results[j].xmin;
                    float ymin2 = det_results[j].ymin;
                    float xmax2 = det_results[j].xmax;
                    float ymax2 = det_results[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nms_thres) {
                        det_results[j].classId = -1;
                    }
                }
            }
        }
    }
} // namespace yolov8