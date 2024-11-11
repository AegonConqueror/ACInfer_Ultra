
#include "postprocess.h"

#include <algorithm>
#include "utils/utils.h"

namespace yolov8 {

    // 辅助函数：将 (x, y, w, h) 转为 (x1, y1, x2, y2)
    Box xywh2xyxy(Box box) {
        Box ret;
        ret.x = box.x - box.w / 2;  // x1
        ret.y = box.y - box.h / 2;  // y1
        ret.w = box.x + box.w / 2;  // x2
        ret.h = box.y + box.h / 2;  // y2
        return ret;
    }

    float get_inter(Box* box1, Box* box2) {
        Box b1 = xywh2xyxy(*box1);
        Box b2 = xywh2xyxy(*box2);

        if (b1.x >= b2.w || b1.w <= b2.x || b1.y >= b2.h || b1.h <= b2.y) {
            return 0.0f;
        }

        float inter_w = std::min(b1.w, b2.w) - std::max(b1.x, b2.x);
        float inter_h = std::min(b1.h, b2.h) - std::max(b1.y, b2.y);
        return inter_w * inter_h;
    }

    float iou(Box *box1, Box *box2) {
        float inter_area = get_inter(box1, box2);
        float box1_area = box1->w * box1->h;
        float box2_area = box2->w * box2->h;
        return inter_area / (box1_area + box2_area - inter_area);
    }

    void PostprocessNormal(std::vector<Box>& res, float *pred, int num_boxes, int num_class, float conf_thres, float iou_thres) {
        float *output = new float[num_boxes * (num_class + 4 + 1)];

        for (size_t i = 0; i < num_boxes; i++) {
            for (size_t j = 0; j < num_class + 4; j++)
            {
                output[i * (num_class + 4 + 1) + j] = pred[j * num_boxes + i];
            }
            float max_conf = *std::max_element(output + i * (num_class + 4 + 1) + 4, output + (i + 1) * (num_class + 4 + 1));
            output[i * (num_class + 4 + 1) + 4] = max_conf;
        }

        std::vector<Box> boxes;
        for (size_t i = 0; i < num_boxes; i++) {
            
            if (output[i * (num_class + 5) + 4] > conf_thres) {
                Box box;
                box.x = output[i * (num_class + 5)];
                box.y = output[i * (num_class + 5) + 1];
                box.w = output[i * (num_class + 5) + 2];
                box.h = output[i * (num_class + 5) + 3];
                box.conf = output[i * (num_class + 5) + 4];
                box.cls = std::max_element(output + i * (num_class + 5) + 5, output + (i + 1) * (num_class + 5)) - (output + i * (num_class + 5) + 5);
                boxes.push_back(box);
            }
        }

        // 按置信度排序
        std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
            return a.conf > b.conf;
        });


        std::vector<bool> suppressed(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;

            res.push_back(boxes[i]);

            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                
                if (iou(&boxes[i], &boxes[j]) > iou_thres) {
                    suppressed[j] = true;
                }
            }
        }
        delete output;
    }

    // ========================================================================================================
    typedef struct DetectRect{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int classId;
    } DetectRect;

    static int headNum = 3;
    static int strides[3] = {8, 16, 32};
    static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

    static inline float fast_exp(float x) {
        // return exp(x);
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
    ){
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

    std::vector<float> GenerateMeshgrid(){
        std::vector<float> meshgrid;

        for (int index = 0; index < headNum; index++) {
            for (int i = 0; i < mapSize[index][0]; i++) {
                for (int j = 0; j < mapSize[index][1]; j++) {
                    meshgrid.push_back(float(j + 0.5));
                    meshgrid.push_back(float(i + 0.5));
                }
            }
        }
        return meshgrid;
    }

    void PostprocessSpilt(
        float **pBlob, std::vector<float> &DetectiontRects,
        int input_width, int input_height, int class_num,
        float conf_thres, float nms_thres
    ){
        static auto meshgrid = GenerateMeshgrid();

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        DetectRect temp;
        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++) {
            float *reg = (float *)pBlob[index * 2 + 0];
            float *cls = (float *)pBlob[index * 2 + 1];

            for (int h = 0; h < mapSize[index][0]; h++) {
                for (int w = 0; w < mapSize[index][1]; w++){
                    gridIndex += 2;

                    for (int cl = 0; cl < class_num; cl++) {
                        cls_val = sigmoid(
                            cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);

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

                    if (cls_max > conf_thres){
                        xmin = (meshgrid[gridIndex + 0] -
                                reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_width ? xmax : input_width;
                        ymax = ymax < input_height ? ymax : input_height;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_width && ymax <= input_height) {
                            temp.xmin = xmin / input_width;
                            temp.ymin = ymin / input_height;
                            temp.xmax = xmax / input_width;
                            temp.ymax = ymax / input_height;
                            temp.classId = cls_index;
                            temp.score = cls_max;
                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(
            detectRects.begin(), detectRects.end(), 
            [](DetectRect &Rect1, DetectRect &Rect2) -> bool { 
                return (Rect1.score > Rect2.score); 
            }
        );

        for (int i = 0; i < detectRects.size(); ++i){
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;

            if (classId != -1){
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                for (int j = i + 1; j < detectRects.size(); ++j) {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nms_thres) {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }
    }

    
} // namespace yolov8
