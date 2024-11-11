
#include <iostream>

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

    auto det_onnx_file = "./weights/yolov8_coco_f16.onnx";
    auto img_file   = "./data/car.jpg";

    YOLO::Detector yolov8_detector(det_onnx_file, 0);

    cv::Mat image = cv::imread(img_file);
    std::vector<YOLO::detect_result> results;

    yolov8_detector.Run(image, results);

    for (auto ibox : results) {
        cv::rectangle(image, ibox.box, (255, 255, 0), 2);
        std::string class_name  = class_names[ibox.class_id];
        char draw_string[100];
        sprintf(draw_string, "%s   %.2f", class_name.c_str(), ibox.confidence);
        printf("name %s\n", class_name.c_str());
        cv::putText(image, draw_string, cv::Point(ibox.box.x, ibox.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2);
    }
    results.clear();

    char save_file[100];
    sprintf(save_file, "./output/yolov8s_test.jpg");
    cv::imwrite(save_file, image);
    return 0;

}