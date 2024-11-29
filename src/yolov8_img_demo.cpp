
#include <iostream>
#include <stdio.h>

#include "detect/yolo.h"

std::vector<std::string> class_names = {
    "person", "no_gesture", "call", "dislike", "fist", "four", "like",
    "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock",
    "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
    "face"
};

int main(int argc, char **argv){

    auto det_onnx_file = "./onnx/yolov8_dict.onnx";
    auto img_file   = "./data/person3.jpg";

    auto yolo_detector = YOLO::CreateDetector(det_onnx_file, YOLO::Type::V8);

    cv::Mat image = cv::imread(img_file);
    std::vector<YOLO::detect_result> results;

    yolo_detector->Run(image, results);

    for (auto ibox : results) {
        iDraw::draw_box_label(image, ibox.box, ibox.class_id, ibox.confidence);
    }
    results.clear();

    char save_file[100];
    sprintf(save_file, "./output/yolov8s_test.jpg");
    cv::imwrite(save_file, image);
    return 0;

}