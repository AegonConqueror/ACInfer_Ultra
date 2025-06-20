
#include "yolov8_postprocess.h"

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
    // static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    int maskNum = 32;
    int mask_seg_w = 160;
    int mask_seg_h = 160;
    // int keypoint_num = 17;

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

    void PostprocessSplit_POSE(
        float **preds, std::vector<float> &pose_rects,
        std::vector<std::map<int, KeyPoint>> &key_points,
        int input_w, int input_h, int class_num, int keypoint_num,
        float conf_thres, float nms_thres
    ) {

        static auto meshgrid = GenerateMeshgrid(input_w, input_h);

        int grid_index = -2;
    
        float cls_max = 0;
        int cls_index = 0;

        std::vector<Yolov8Rect> pose_results;

        for (int head_index = 0; head_index < headNum; head_index++) {
            float* reg  = (float* )preds[head_index * 2 + 0];
            float* cls  = (float* )preds[head_index * 2 + 1];
            float* pose = (float* )preds[head_index + headNum * 2];

            float cls_val = 0;

            int grid_w = static_cast<int>(input_w / strides[head_index]);
            int grid_h = static_cast<int>(input_h / strides[head_index]);

            for (int h = 0; h < grid_w; h++) {
                for (int w = 0; w < grid_h; w++) {
                    grid_index += 2;
                    for (int cl = 0; cl < class_num; cl++) {
                        cls_val = sigmoid(cls[cl * grid_w * grid_h + h * grid_h + w]);
                        if (0 == cl) {
                            cls_max = cls_val;
                            cls_index = cl;
                        } else {
                            if (cls_val > cls_max) {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }

                    if (cls_max > conf_thres) {
                        Yolov8Rect temp;

                        float xmin = (meshgrid[grid_index + 0] - reg[0 * grid_w * grid_h + h * grid_h + w]) * strides[head_index];
                        float ymin = (meshgrid[grid_index + 1] - reg[1 * grid_w * grid_h + h * grid_h + w]) * strides[head_index];
                        float xmax = (meshgrid[grid_index + 0] + reg[2 * grid_w * grid_h + h * grid_h + w]) * strides[head_index];
                        float ymax = (meshgrid[grid_index + 1] + reg[3 * grid_w * grid_h + h * grid_h + w]) * strides[head_index];

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
                                kp.x = (pose[(kc * 3 + 0) * grid_w * grid_h + h * grid_h + w] * 2 + (meshgrid[grid_index + 0] - 0.5)) * strides[head_index] / input_w;
                                kp.y = (pose[(kc * 3 + 1) * grid_w * grid_h + h * grid_h + w] * 2 + (meshgrid[grid_index + 1] - 0.5)) * strides[head_index] / input_h;
                                kp.score = sigmoid(pose[(kc * 3 + 2) * grid_w * grid_h + h * grid_h + w]);
                                kp.id = kc;
                                temp.keyPoints.push_back(kp);
                            }
                            pose_results.push_back(temp);
                        }
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


} // namespace yolov8
