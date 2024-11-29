
#ifndef ACENGINE_BOTSORT_H
#define ACENGINE_BOTSORT_H

#include <vector>
#include <string>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>

namespace BotSORT {

    constexpr uint32_t FEATURE_DIM = 512;

    struct Detection {
        cv::Rect_<float> bbox_tlwh;
        int class_id;
        float confidence;
        Eigen::Matrix<float, 1, 512> feature;
    };

    enum TrackState{
        New         = 0,
        Tracked     = 1,
        Lost        = 2,
        LongLost    = 3,
        Removed     = 4
    };

    // TODO: only support SparseOptFlow method now!
    enum GMC_Method{
        ORB             = 0,
        SIFT            = 1,
        ECC             = 2,
        SparseOptFlow   = 3
    };

    struct TrackerConfig {
        bool reid_enabled = true;
        bool gmc_enabled = true;

        float track_high_thresh = 0.5;
        float track_low_thresh = 0.1;
        float new_track_thresh = 0.6;
        float track_buffer = 30;
        float match_thresh = 0.8;

        float proximity_thresh = 0.5;
        float appearance_thresh = 0.25;

        GMC_Method gmc_method = GMC_Method::SparseOptFlow;

        float frame_rate = 30;
        float lambda = 0.985;

        std::string distance_metric = "cosine"; // euclidean  cosine
    };

    class TrackObject {
    public:
        virtual int                 get_cls_id()        const = 0;
        virtual int                 get_track_id()      const = 0;
        virtual int                 get_state()         const = 0;
        virtual std::vector<float>  get_box_tlwh()      const = 0;
    };

    class Tracker {
    public:
        virtual std::vector<TrackObject *> get_tracks() = 0;
        virtual void update(const std::vector<Detection> &detections, const cv::Mat &frame) = 0;
    };

    std::shared_ptr<Tracker> create_tracker(
        const TrackerConfig& config = TrackerConfig()
    );
    
} // namespace BotSORT


#endif // ACENGINE_BOTSORT_H