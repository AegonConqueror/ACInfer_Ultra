#ifndef ACINFER_ULTRA_YOLO_TYPE_H
#define ACINFER_ULTRA_YOLO_TYPE_H

#include <vector>
#include <map>

struct Box{
    int left;
    int top;
    int right;
    int bottom;

    Box() = default;
    Box(int left, int top, int right, int bottom) :left(left), top(top), right(right), bottom(bottom){}
};

struct KeyPoint {
    float x, y, score;
    int id;

    KeyPoint() = default;
    KeyPoint(float x, float y, float score, int id): x(x), y(y), score(score), id(id) {}
};
typedef std::map<int, KeyPoint> KeyPoints;
typedef std::vector<KeyPoints> KeyPointsArray;

struct yolo_result {
    int         classId;
    float       score;
    Box         box;
    KeyPoints   keypoints;
};
typedef std::vector<yolo_result> YoloResults;

#endif // ACINFER_ULTRA_YOLO_TYPE_H