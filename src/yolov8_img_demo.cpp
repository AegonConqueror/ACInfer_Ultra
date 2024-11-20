
#include <iostream>
#include <stdio.h>

#include "detect/yolo.h"

std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
    "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "
};

int main(int argc, char **argv){ 

    auto det_onnx_file = "./onnx/yolov8s_coco.onnx";
    auto img_file   = "./data/street.jpg";

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