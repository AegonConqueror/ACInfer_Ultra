
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "detect/yolo.h"
#include "feature/ReID.h"
#include "tracker/botsort/botsort.h"


auto det_onnx_file = "./onnx/yolov8_dict_f16.onnx";
auto reid_onnx_file = "./onnx/feature_f16.onnx";

void test_botsort(
    cv::Mat& frame, std::vector<BotSORT::Detection>& results, 
    std::shared_ptr<BotSORT::Tracker>& tracker
) {
    std::vector<BotSORT::Detection> objects;
    for (BotSORT::Detection dr : results) {
        if(dr.class_id == 0) {
            objects.push_back(dr);
        }
    }

    tracker->update(objects, frame);

    static std::map<int, cv::Scalar> track_colors;
    cv::Scalar detection_color = cv::Scalar(0, 0, 0);
    for (const auto &det: objects) {
        cv::rectangle(frame, det.bbox_tlwh, detection_color, 1);
    }
    auto tracks = tracker->get_tracks();
    for (auto obj : tracks) {
        std::vector<float> bbox_tlwh = obj->get_box_tlwh();
        cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

        if (track_colors.find(obj->get_track_id()) == track_colors.end()) {
            track_colors[obj->get_track_id()] = color;
        } else {
            color = track_colors[obj->get_track_id()];
        }

        cv::rectangle(frame,
                      cv::Rect(static_cast<int>(bbox_tlwh[0]),
                               static_cast<int>(bbox_tlwh[1]),
                               static_cast<int>(bbox_tlwh[2]),
                               static_cast<int>(bbox_tlwh[3])),
                      color, 2);
        cv::putText(frame, std::to_string(obj->get_track_id()),
                    cv::Point(static_cast<int>(bbox_tlwh[0]),
                              static_cast<int>(bbox_tlwh[1])),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);

        cv::rectangle(frame, cv::Rect(10, 10, 20, 20), detection_color, -1);
        cv::putText(frame, "Detection", cv::Point(40, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, detection_color, 2);
    }
}

int main(int argc, char **argv){

    auto yolo_detector = YOLO::CreateDetector(det_onnx_file, YOLO::Type::V8);

    auto reid_model = CreateReID(reid_onnx_file);
    auto tracker = BotSORT::create_tracker();
    
    LOG_INFO("begin read video\n");
    cv::VideoWriter video(
        "./output/botsort_10fps.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 
        10, cv::Size(2112, 1188)
    );

    std::vector<YOLO::detect_result> results;
    for (int i = 1; i < 1453; i++) {
        char img_file[100];
        sprintf(img_file, "./data/frame_10fps/frame_%04d.jpg", i);
        cv::Mat frame = cv::imread(img_file);
        
        yolo_detector->Run(frame, results);

        std::vector<BotSORT::Detection> detections;
        for (auto result : results) {
            BotSORT::Detection det;
            det.class_id = 0;
            det.bbox_tlwh = result.box;
            det.confidence = result.confidence;
            cv::Mat patch = frame(det.bbox_tlwh);
            auto feature = reid_model->get_features(patch);
            det.feature = feature;
            detections.push_back(det);
        }

        test_botsort(frame, detections, tracker);

        results.clear();
        video.write(frame);
        printf("dealed index %d\n", i);
    }
    video.release();
    return 0;

}