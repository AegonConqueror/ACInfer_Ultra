
#ifndef ACINFER_ULTRA_YOLOV8_TYPE_H
#define ACINFER_ULTRA_YOLOV8_TYPE_H

#include <opencv2/opencv.hpp>

enum class TaskType : int {
    YOLOv8_DET   = 0,
    YOLOv8_SEG   = 1,
    YOLOv8_POSE  = 2
};

typedef struct KeyPoint {
    float x, y;
    float score;
    int id;

    KeyPoint() = default;
    KeyPoint(float x, float y, float score, int id): x(x), y(y), score(score), id(id) {}
} KeyPoint;

typedef struct yolov8Rect {
    float xmin, ymin, xmax, ymax;
    float score;
    int classId;
    std::vector<KeyPoint> keyPoints;
} yolov8Rect;

typedef struct yolov8_result {
    cv::Rect                    box;
    int                         class_id;
    float                       confidence;
    cv::Mat                     mask;
    std::map<int, KeyPoint>     keypoints;
} yolov8_result;

#endif // ACINFER_ULTRA_YOLOV8_TYPE_H