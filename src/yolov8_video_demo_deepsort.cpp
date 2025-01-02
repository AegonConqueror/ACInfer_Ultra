
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "detect/yolo.h"
#include "feature/ReID.h"
#include "tracker/deepsort/deepsort.h"

auto det_onnx_file = "./onnx/coco_pose_n_relu.onnx";
auto reid_onnx_file = "./onnx/osnet_x0_25_market1501.onnx";

int main(int argc, char **argv){

    auto yolo_detector = YOLO::CreateDetector(det_onnx_file, YOLO::Type::V8);
    auto reid_fature = CreateReID(reid_onnx_file);
    auto deep_tracker = DeepSort::create_tracker();
    
    LOG_INFO("begin read video\n");
    cv::VideoCapture capture("./data/video/sort_test.mp4");
    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }

    int video_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int video_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);

    printf("video_width %d video_height %d\n",video_width, video_height);
    
    cv::VideoWriter video(
        "./output/sort_test.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), 
        fps, cv::Size(video_width, video_height)
    );

    std::vector<YOLO::detect_result> results;
    int index = 0;
    while (true) {
        LOG_INFO("dealing frame index %d", index);
        cv::Mat frame;
        if (!capture.read(frame)) {
            printf("\n Cannot read the video file. please check your video.\n");
            break;
        }

        yolo_detector->Run(frame, results);

        DeepSort::BBoxes deep_bboxes;
        for(auto dr : results) {
            if (dr.class_id == 0) {
                DeepSort::Box deep_box(dr.box.x, dr.box.y, dr.box.x + dr.box.width, dr.box.y + dr.box.height);
                cv::Mat image_patch = frame(dr.box).clone();
                Eigen::Matrix<float, 1, 512> feature = reid_fature->get_features(image_patch);
                cv::Mat featureMat(1, 512, CV_32F, feature.data());
                deep_box.feature = featureMat;
                deep_bboxes.push_back(deep_box);
                iDraw::draw_box_label(frame, dr.box, dr.class_id, dr.confidence);
            }
        }

        deep_tracker->update(deep_bboxes);
        auto track_objects = deep_tracker->get_objects();

        for (auto track : track_objects){
            if (!track->is_confirmed() || track->time_since_update() > 1) 
                continue;
            cv::Rect box = DeepSort::convert_box_to_rect(track->last_position());
            int track_id = track->id();
            iDraw::draw_box_label(frame, box, track_id, 0.0f);
        }

        results.clear();
        index++;
    
        video.write(frame);
    }

    capture.release();
    video.release();
    return 0;

}