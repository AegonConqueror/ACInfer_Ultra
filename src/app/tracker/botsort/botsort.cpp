
#include "botsort.h"

#include <cstdint>
#include <limits.h>
#include <iostream>

#include <opencv2/core/eigen.hpp>
#include <unordered_set>

#include "lapjv.h"

namespace BotSORT {
    static float chi2inv95[10] = {
        0,
        3.8415,
        5.9915,
        7.8147,
        9.4877,
        11.070,
        12.592,
        14.067,
        15.507,
        16.919
    };  

    struct AssociationData {
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_track_indices;
        std::vector<int> unmatched_det_indices;
    };

    using CostMatrix                = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using FeatureMatrix             = Eigen::Matrix<float, 1, FEATURE_DIM>;
    using KFDataStateSpace          = std::pair<Eigen::Matrix<float, 1, 8>, Eigen::Matrix<float, 8, 8>>;
    using KFDataMeasurementSpace    = std::pair<Eigen::Matrix<float, 1, 4>, Eigen::Matrix<float, 4, 4>>;

    class KalmanFilter {
    public:
        KalmanFilter(double dt) {
            motion_mat_.setIdentity();
            update_mat_.setIdentity();

            for (Eigen::Index i = 0; i < 4; i++) {
                motion_mat_(i, i + 4) = 1.0;
            }
        }

        KFDataStateSpace init(const Eigen::Matrix<float, 1, 4> &measurement) const {

            Eigen::Matrix<float, 1, 4> mean_pos = measurement;
            Eigen::Matrix<float, 1, 4> mean_vel;

            for (int i = 0; i < 4; i++) {
                mean_vel(i) = 0;
            }

            Eigen::Matrix<float, 1, 8> mean;
            for (int i = 0; i < 8; i++) {
                if (i < 4) {
                    mean(i) = mean_pos(i);
                } else {
                    mean(i) = mean_vel(i - 4);
                }
            }
            Eigen::Matrix<float, 1, 8> std;
            std(0) = 2 * std_weight_position_ * measurement[3];
            std(1) = 2 * std_weight_position_ * measurement[3];
            std(2) = 2 * std_weight_position_ * measurement[3];
            std(3) = 2 * std_weight_position_ * measurement[3];
            std(4) = 10 * std_weight_velocity_ * measurement[3];
            std(5) = 10 * std_weight_velocity_ * measurement[3];
            std(6) = 10 * std_weight_velocity_ * measurement[3];
            std(7) = 10 * std_weight_velocity_ * measurement[3];

            Eigen::Matrix<float, 1, 8> tmp = std.array().square();
            Eigen::Matrix<float, 8, 8> var = tmp.asDiagonal();
            return std::make_pair(mean, var);
        }

        void predict(Eigen::Matrix<float, 1, 8> &mean, Eigen::Matrix<float, 8, 8> &covariance) {
            Eigen::Matrix<float, 1, 4> std_pos;
            std_pos << std_weight_position_ * mean(3),
                std_weight_position_ * mean(3),
                std_weight_position_ * mean(3),
                std_weight_position_ * mean(3);
            Eigen::Matrix<float, 1, 4> std_vel;
            std_vel << std_weight_velocity_ * mean(3),
                std_weight_velocity_ * mean(3),
                std_weight_velocity_ * mean(3),
                std_weight_velocity_ * mean(3);
            Eigen::Matrix<float, 1, 8> tmp;
            tmp.block<1, 4>(0, 0) = std_pos;
            tmp.block<1, 4>(0, 4) = std_vel;
            tmp = tmp.array().square();
            Eigen::Matrix<float, 8, 8> motion_cov = tmp.asDiagonal();
            Eigen::Matrix<float, 1, 8> mean1 = this->motion_mat_ * mean.transpose();
            Eigen::Matrix<float, 8, 8> covariance1 = this->motion_mat_ * covariance *(motion_mat_.transpose());
            covariance1 += motion_cov;

            mean = mean1;
            covariance = covariance1;
        }

        KFDataMeasurementSpace project(
            const Eigen::Matrix<float, 1, 8> &mean, 
            const Eigen::Matrix<float, 8, 8> &covariance
        ) const {
            Eigen::Matrix<float, 1, 4> std;
            std << std_weight_position_ * mean(3), 
                std_weight_position_ * mean(3),
                std_weight_position_ * mean(3), 
                std_weight_position_ * mean(3);
            Eigen::Matrix<float, 1, 4> mean1 = update_mat_ * mean.transpose();
            Eigen::Matrix<float, 4, 4> covariance1 = update_mat_ * covariance * (update_mat_.transpose());
            Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
            diag = diag.array().square().matrix();
            covariance1 += diag;
            return std::make_pair(mean1, covariance1);
        }

        KFDataStateSpace update(
            const Eigen::Matrix<float, 1, 8> &mean,
            const Eigen::Matrix<float, 8, 8> &covariance,
            const Eigen::Matrix<float, 1, 4> &measurement
        ) {
            KFDataMeasurementSpace pa = project(mean, covariance);
            Eigen::Matrix<float, 1, 4> projected_mean = pa.first;
            Eigen::Matrix<float, 4, 4> projected_cov = pa.second;

            Eigen::Matrix<float, 4, 8> B = (covariance * (update_mat_.transpose())).transpose();
            Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
            Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
            auto tmp = innovation * (kalman_gain.transpose());
            Eigen::Matrix<float, 1, 8> new_mean = (mean.array() + tmp.array()).matrix();
            Eigen::Matrix<float, 8, 8> new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
            return std::make_pair(new_mean, new_covariance);
        }

        Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
            const Eigen::Matrix<float, 1, 8> &mean,
            const Eigen::Matrix<float, 8, 8> &covariance,
            const std::vector<Eigen::Matrix<float, 1, 4>> &measurements,
            bool only_position = false
        ) const {

            KFDataMeasurementSpace pa = this->project(mean, covariance);
            if (only_position) {
                printf("not implement!");
                exit(0);
            }
            Eigen::Matrix<float, 1, 4> mean1 = pa.first;
            Eigen::Matrix<float, 4, 4> covariance1 = pa.second;

            //    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
            Eigen::Matrix<float, -1, 4> d(measurements.size(), 4);
            int pos = 0;
            for (Eigen::Matrix<float, 1, 4> box : measurements) {
                d.row(pos++) = box - mean1;
            }
            Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
            Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
            auto zz = ((z.array())*(z.array())).matrix();
            auto square_maha = zz.colwise().sum();
            return square_maha;
        } 

    private:
        float std_weight_position_ = 1.0 / 20;
        float std_weight_velocity_ = 1.0 / 160;

        Eigen::Matrix<float, 8, 8> motion_mat_;
        Eigen::Matrix<float, 4, 8> update_mat_;
    };
    
    inline float iou(const std::vector<float> &tlwh_a, const std::vector<float> &tlwh_b) {
        float left = std::max(tlwh_a[0], tlwh_b[0]);
        float top = std::max(tlwh_a[1], tlwh_b[1]);
        float right = std::min(tlwh_a[0] + tlwh_a[2], tlwh_b[0] + tlwh_b[2]);
        float bottom = std::min(tlwh_a[1] + tlwh_a[3], tlwh_b[1] + tlwh_b[3]);
        float area_i = std::max(right - left + 1, 0.0f) * std::max(bottom - top + 1, 0.0f);
        float area_a = (tlwh_a[2] + 1) * (tlwh_a[3] + 1);
        float area_b = (tlwh_b[2] + 1) * (tlwh_b[3] + 1);
        return area_i / (area_a + area_b - area_i);
    }

    inline float cosine_distance(
        const std::unique_ptr<FeatureMatrix> &x,
        const std::shared_ptr<FeatureMatrix> &y
    ) {
        return 1.0f - (x->dot(*y) / (x->norm() * y->norm() + 1e-5f));
    }

    inline float euclidean_distance(
        const std::unique_ptr<FeatureMatrix> &x,
        const std::shared_ptr<FeatureMatrix> &y) {
        return (x->transpose() - y->transpose()).norm();
    }

    double lapjv(
        CostMatrix &cost, std::vector<int> &rowsol,
        std::vector<int> &colsol, bool extend_cost = false,
        float cost_limit = std::numeric_limits<float>::max(),
        bool return_cost = true
    ) {
        std::vector<std::vector<float>> cost_c;

        for (Eigen::Index i = 0; i < cost.rows(); i++) {
            std::vector<float> row;
            for (Eigen::Index j = 0; j < cost.cols(); j++) {
                row.push_back(cost(i, j));
            }
            cost_c.push_back(row);
        }

        std::vector<std::vector<float>> cost_c_extended;

        int n_rows = static_cast<int>(cost.rows());
        int n_cols = static_cast<int>(cost.cols());
        rowsol.resize(n_rows);
        colsol.resize(n_cols);

        int n = 0;
        if (n_rows == n_cols) { 
            n = n_rows; 
        } else {
            if (!extend_cost) {
                printf("set extend_cost=True\n");
                exit(0);
            }
        }

        if (extend_cost || cost_limit < LONG_MAX) {
            n = n_rows + n_cols;
            cost_c_extended.resize(n);
            for (int i = 0; i < cost_c_extended.size(); i++)
                cost_c_extended[i].resize(n);

            if (cost_limit < LONG_MAX) {
                for (int i = 0; i < cost_c_extended.size(); i++) {
                    for (int j = 0; j < cost_c_extended[i].size(); j++) {
                        cost_c_extended[i][j] = cost_limit / 2.0;
                    }
                }
            } else {
                float cost_max = -1;
                for (int i = 0; i < cost_c.size(); i++) {
                    for (int j = 0; j < cost_c[i].size(); j++) {
                        if (cost_c[i][j] > cost_max) 
                            cost_max = cost_c[i][j];
                    }
                }
                for (int i = 0; i < cost_c_extended.size(); i++) {
                    for (int j = 0; j < cost_c_extended[i].size(); j++) {
                        cost_c_extended[i][j] = cost_max + 1;
                    }
                }
            }

            for (int i = n_rows; i < cost_c_extended.size(); i++) {
                for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = 0;
                }
            }
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < n_cols; j++) {
                    cost_c_extended[i][j] = cost_c[i][j];
                }
            }

            cost_c.clear();
            cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
        }

        double **cost_ptr;
        cost_ptr = new double *[n];
        for (int i = 0; i < n; i++) {
            cost_ptr[i] = new double[n];
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) { 
                cost_ptr[i][j] = cost_c[i][j]; 
            }
        }

        int *x_c = new int[n];
        int *y_c = new int[n];

        int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
        if (ret != 0) {
            printf("Calculate Wrong!\n");
            exit(0);
        }

        double opt = 0.0;

        if (n != n_rows) {
            for (int i = 0; i < n; i++) {
                if (x_c[i] >= n_cols) x_c[i] = -1;
                if (y_c[i] >= n_rows) y_c[i] = -1;
            }
            for (int i = 0; i < n_rows; i++) { 
                rowsol[i] = x_c[i]; 
            }

            for (int i = 0; i < n_cols; i++) { 
                colsol[i] = y_c[i]; 
            }

            if (return_cost) {
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol[i] != -1) { 
                        opt += cost_ptr[i][rowsol[i]]; 
                    }
                }
            }
        } else if (return_cost) {
            for (int i = 0; i < rowsol.size(); i++) {
                opt += cost_ptr[i][rowsol[i]];
            }
        }

        for (int i = 0; i < n; i++) { 
            delete[] cost_ptr[i]; 
        }
        delete[] cost_ptr;
        delete[] x_c;
        delete[] y_c;

        return opt;
    }

    class Track : public TrackObject {
    public:
        Track(
            std::vector<float> tlwh, float score, uint8_t class_id,
            std::shared_ptr<FeatureMatrix> feat = nullptr, int feat_history_size = 50
        ) : det_tlwh(std::move(tlwh)), _score(score), _class_id(class_id), 
            tracklet_len(0), is_activated(false), state(TrackState::New) 
        {
            if (feat) {
                _feat_history_size = feat_history_size;
                _update_features(std::make_shared<FeatureMatrix>(*feat));
            } else {
                curr_feat = nullptr;
                smooth_feat = nullptr;
                _feat_history_size = 0;
            }

            _update_class_id(class_id, score);
            _update_tracklet_tlwh_inplace();
        }

        static int next_id() {
            static int _count = 0;
            _count++;
            return _count;
        }

        uint32_t end_frame() const {
            return frame_id;
        }

        void mark_lost() {
            state = TrackState::Lost;
        }

        void mark_long_lost() {
            state = TrackState::LongLost;
        }

        void mark_removed() {
            state = TrackState::Removed;
        }

        std::vector<float> get_tlwh() const {
            return _tlwh;
        }

        float get_score() const {
            return _score;
        }

        void activate(KalmanFilter &kalman_filter, uint32_t frame_id) {
            track_id = next_id();

            Eigen::Matrix<float, 1, 4> detection_bbox;
            _populate_DetVec_xywh(detection_bbox, det_tlwh);

            KFDataStateSpace state_space = kalman_filter.init(detection_bbox);
            mean = state_space.first;
            covariance = state_space.second;

            if (frame_id == 1) {
                is_activated = true;
            }
            this->frame_id = frame_id;
            start_frame = frame_id;
            state = TrackState::Tracked;
            tracklet_len = 1;
            _update_tracklet_tlwh_inplace();
        }

        void re_activate(
            KalmanFilter &kalman_filter, Track &new_track,
            uint32_t frame_id, bool new_id = false
        ) {
            Eigen::Matrix<float, 1, 4> new_track_bbox;
            _populate_DetVec_xywh(new_track_bbox, new_track._tlwh);

            KFDataStateSpace state_space = kalman_filter.update(mean, covariance, new_track_bbox);
            mean = state_space.first;
            covariance = state_space.second;

            if (new_track.curr_feat) {
                _update_features(new_track.curr_feat);
            }

            if (new_id) {
                track_id = next_id();
            }

            tracklet_len = 0;
            state = TrackState::Tracked;
            is_activated = true;
            _score = new_track._score;
            this->frame_id = frame_id;

            _update_class_id(new_track._class_id, new_track._score);
            _update_tracklet_tlwh_inplace();
        }

        void predict(KalmanFilter &kalman_filter) {
            if (state != TrackState::Tracked) {
                mean(6) = 0, mean(7) = 0;
            }
            
            kalman_filter.predict(mean, covariance);
            _update_tracklet_tlwh_inplace();
        }

        void static multi_predict(std::vector<std::shared_ptr<Track>> &tracks, KalmanFilter &kalman_filter) {
            for (std::shared_ptr<Track> &track: tracks) {
                track->predict(kalman_filter);
            }
        }

        void apply_camera_motion(const Eigen::Matrix<float, 3, 3> &H) {
            Eigen::MatrixXf R = H.block(0, 0, 2, 2);
            Eigen::VectorXf t = H.block(0, 2, 2, 1);

            Eigen::Matrix<float, 8, 8> R8x8 = Eigen::Matrix<float, 8, 8>::Identity();
            R8x8.block(0, 0, 2, 2) = R;

            mean = R8x8 * mean.transpose();
            mean.head(2) += t;
            covariance = R8x8 * covariance * R8x8.transpose();
        }

        void static multi_gmc(
            std::vector<std::shared_ptr<Track>> &tracks, 
            const Eigen::Matrix<float, 3, 3> &H
        ) {
            for (std::shared_ptr<Track> &track: tracks) {
                track->apply_camera_motion(H);
            }
        }

        void update(KalmanFilter &kalman_filter, Track &new_track, uint32_t frame_id) {
            Eigen::Matrix<float, 1, 4> new_track_bbox;
            _populate_DetVec_xywh(new_track_bbox, new_track._tlwh);

            KFDataStateSpace state_space = kalman_filter.update(mean, covariance, new_track_bbox);

            if (new_track.curr_feat) {
                _update_features(new_track.curr_feat);
            }

            mean = state_space.first;
            covariance = state_space.second;
            state = TrackState::Tracked;
            is_activated = true;
            _score = new_track._score;
            tracklet_len++;
            this->frame_id = frame_id;

            _update_class_id(new_track._class_id, new_track._score);
            _update_tracklet_tlwh_inplace();
        }

    private:
        void _update_features(const std::shared_ptr<FeatureMatrix> &feat) {
            *feat /= feat->norm();

            if (_feat_history.empty()) {
                curr_feat = feat;
                smooth_feat = std::make_unique<FeatureMatrix>(*curr_feat);
            } else {
                *smooth_feat = _alpha * (*smooth_feat) + (1 - _alpha) * (*feat);
            }

            if (_feat_history.size() == _feat_history_size) {
                _feat_history.pop_front();
            }
            _feat_history.push_back(curr_feat);
            *smooth_feat /= smooth_feat->norm();
        }

        static void _populate_DetVec_xywh(Eigen::Matrix<float, 1, 4> &bbox_xywh, const std::vector<float> &tlwh) {
            bbox_xywh << tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2], tlwh[3];
        }

        void _update_tracklet_tlwh_inplace() {
            if (state == TrackState::New) {
                _tlwh = det_tlwh;
                return;
            }
            _tlwh = {mean(0) - mean(2) / 2, mean(1) - mean(3) / 2, mean(2), mean(3)};
        }

        void _update_class_id(uint8_t class_id, float score) {
            if (!_class_hist.empty()) {
                int max_freq = 0;
                bool found = false;

                for (auto &class_hist: _class_hist) {
                    if (class_hist.first == class_id) {
                        class_hist.second += score;
                        found = true;
                    }
                    if (static_cast<int>(class_hist.second) > max_freq) {
                        max_freq = static_cast<int>(class_hist.second);
                        _class_id = class_hist.first;
                    }
                }

                if (!found) {
                    _class_hist.emplace_back(class_id, score);
                    _class_id = class_id;
                }
            } else {
                _class_hist.emplace_back(class_id, score);
                _class_id = class_id;
            }
        }

        virtual int get_cls_id() const override {
            return _class_id;
        }

        virtual int get_track_id() const override {
            return track_id;
        }

        virtual int get_state() const override {
            return state;
        }
        
        virtual std::vector<float> get_box_tlwh() const override {
            return get_tlwh();
        }

    public:
        bool is_activated;
        int track_id;
        int state;
        uint8_t _class_id;
        uint32_t frame_id;
        uint32_t tracklet_len;
        uint32_t start_frame;
        std::vector<float> det_tlwh;
        std::shared_ptr<FeatureMatrix> curr_feat;
        std::unique_ptr<FeatureMatrix> smooth_feat;
        Eigen::Matrix<float, 1, 8> mean;
        Eigen::Matrix<float, 8, 8> covariance;

    private:
        std::vector<float> _tlwh;
        std::vector<std::pair<uint8_t, float>> _class_hist;
        float _score;
        float _alpha = 0.9;
        int _feat_history_size;
        std::deque<std::shared_ptr<FeatureMatrix>> _feat_history;
    };
    
    class GMC {
    public:
        GMC() : _useHarrisDetector(false), _maxCorners(1000), _blockSize(3), 
            _ransac_max_iters(500), _qualityLevel(0.01), _k(0.04), _minDistance(1.0), 
            _downscale(2.0), _inlier_ratio(0.5), _ransac_conf(0.99) {}

        Eigen::Matrix<float, 3, 3> apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) {
            // Initialization
            int height = frame_raw.rows;
            int width = frame_raw.cols;

            Eigen::Matrix<float, 3, 3> H;
            H.setIdentity();

            cv::Mat frame;
            cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);

            // Downscale
            if (_downscale > 1.0F) {
                width /= _downscale, height /= _downscale;
                cv::resize(frame, frame, cv::Size(width, height));
            }

            // Detect keypoints
            std::vector<cv::Point2f> keypoints;
            cv::goodFeaturesToTrack(
                frame, keypoints, _maxCorners, _qualityLevel,
                _minDistance, cv::noArray(), _blockSize,
                _useHarrisDetector, _k
            );

            if (!_first_frame_initialized || _prev_keypoints.size() == 0) {
                /**
                 *  If this is the first frame, there is nothing to match
                 *  Save the keypoints and descriptors, return identity matrix 
                 */
                _first_frame_initialized = true;
                _prev_frame = frame.clone();
                _prev_keypoints = keypoints;
                return H;
            }

            // Find correspondences between the previous and current frame
            std::vector<cv::Point2f> matched_keypoints;
            std::vector<uchar> status;
            std::vector<float> err;
            try {
                cv::calcOpticalFlowPyrLK(_prev_frame, frame, _prev_keypoints,
                                        matched_keypoints, status, err);
            } catch (const cv::Exception &e) {
                printf("Warning: Could not find correspondences for GMC\n");
                return H;
            }

            // Keep good matches
            std::vector<cv::Point2f> prev_points, curr_points;
            for (size_t i = 0; i < matched_keypoints.size(); i++) {
                if (status[i]) {
                    prev_points.push_back(_prev_keypoints[i]);
                    curr_points.push_back(matched_keypoints[i]);
                }
            }

            // Estimate affine matrix
            if (prev_points.size() > 4) {
                cv::Mat inliers;
                cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3,
                                        inliers, _ransac_max_iters, _ransac_conf);

                double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
                if (inlier_ratio > _inlier_ratio) {
                    cv2eigen(homography, H);
                    if (_downscale > 1.0) {
                        H(0, 2) *= _downscale;
                        H(1, 2) *= _downscale;
                    }
                }
            }

            _prev_frame = frame.clone();
            _prev_keypoints = keypoints;
            return H;
        }

    private:
        float _downscale;

        bool _first_frame_initialized = false;
        cv::Mat _prev_frame;
        std::vector<cv::Point2f> _prev_keypoints;

        // Parameters
        int _maxCorners;
        int _blockSize;
        int _ransac_max_iters;
        double _qualityLevel;
        double  _k;
        double _minDistance;
        bool _useHarrisDetector;
        float _inlier_ratio;
        float _ransac_conf;
    };
    
    std::tuple<CostMatrix, CostMatrix> iou_distance(
        const std::vector<std::shared_ptr<Track>> &tracks,
        const std::vector<std::shared_ptr<Track>> &detections,
        float max_iou_distance
    ) {
        size_t num_tracks = tracks.size();
        size_t num_detections = detections.size();

        CostMatrix cost_matrix = 
            Eigen::MatrixXf::Zero(
                static_cast<Eigen::Index>(num_tracks),
                static_cast<Eigen::Index>(num_detections)
            );
        CostMatrix iou_dists_mask =
            Eigen::MatrixXf::Zero(
                static_cast<Eigen::Index>(num_tracks),
                static_cast<Eigen::Index>(num_detections)
            );

        if (num_tracks > 0 && num_detections > 0) {
            for (int i = 0; i < num_tracks; i++) {
                for (int j = 0; j < num_detections; j++) {
                    cost_matrix(i, j) = 1.0F - iou(tracks[i]->get_tlwh(), detections[j]->get_tlwh());

                    if (cost_matrix(i, j) > max_iou_distance) {
                        iou_dists_mask(i, j) = 1.0F;
                    }
                }
            }
        }

        return {cost_matrix, iou_dists_mask};
    }

    CostMatrix iou_distance(
        const std::vector<std::shared_ptr<Track>> &tracks,
        const std::vector<std::shared_ptr<Track>> &detections
    ) {
        size_t num_tracks = tracks.size();
        size_t num_detections = detections.size();

        CostMatrix cost_matrix = Eigen::MatrixXf::Zero(
            static_cast<Eigen::Index>(num_tracks),
            static_cast<Eigen::Index>(num_detections)
        );
        if (num_tracks > 0 && num_detections > 0) {
            for (int i = 0; i < num_tracks; i++) {
                for (int j = 0; j < num_detections; j++) {
                    cost_matrix(i, j) = 1.0F - iou(tracks[i]->get_tlwh(), detections[j]->get_tlwh());
                }
            }
        }

        return cost_matrix;
    }

    void fuse_score(
        CostMatrix &cost_matrix,
        const std::vector<std::shared_ptr<Track>> &detections
    ) {
        if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0)
            return;

        for (Eigen::Index i = 0; i < cost_matrix.rows(); i++) {
            for (Eigen::Index j = 0; j < cost_matrix.cols(); j++) {
                cost_matrix(i, j) = 1.0F - ((1.0F - cost_matrix(i, j)) * detections[j]->get_score());
            }
        }
    }

    std::tuple<CostMatrix, CostMatrix> embedding_distance(
        const std::vector<std::shared_ptr<Track>> &tracks,
        const std::vector<std::shared_ptr<Track>> &detections,
        float max_embedding_distance,
        const std::string &distance_metric
    ) {
        if (!(distance_metric == "euclidean" || distance_metric == "cosine")) {
            printf("Invalid distance metric, Only 'euclidean' and 'cosine' are supported.\n");
            exit(1);
        }

        size_t num_tracks = tracks.size();
        size_t num_detections = detections.size();

        CostMatrix cost_matrix =
                Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks), static_cast<Eigen::Index>(num_detections));
        CostMatrix embedding_dists_mask =
                Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks), static_cast<Eigen::Index>(num_detections));

        if (num_tracks > 0 && num_detections > 0) {
            for (int i = 0; i < num_tracks; i++) {
                for (int j = 0; j < num_detections; j++) {
                    if (distance_metric == "euclidean")
                        cost_matrix(i, j) = std::max(
                            0.0f, euclidean_distance(tracks[i]->smooth_feat, detections[j]->curr_feat)
                        );
                    else
                        cost_matrix(i, j) = std::max(
                            0.0f, cosine_distance(tracks[i]->smooth_feat, detections[j]->curr_feat)
                        );

                    if (cost_matrix(i, j) > max_embedding_distance) {
                        embedding_dists_mask(i, j) = 1.0F;
                    }
                }
            }
        }

        return {cost_matrix, embedding_dists_mask};
    }

    void fuse_motion(
        const KalmanFilter &KF, CostMatrix &cost_matrix,
        const std::vector<std::shared_ptr<Track>> &tracks,
        const std::vector<std::shared_ptr<Track>> &detections,
        float lambda = 0.98, bool only_position = false
    ) {
        if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
            return;
        }

        uint8_t gating_dim = only_position ? 2 : 4;
        const double gating_threshold = chi2inv95[gating_dim];

        std::vector<Eigen::Matrix<float, 1, 4>> measurements;
        std::vector<float> det_xywh;
        for (const std::shared_ptr<Track> &detection: detections) {
            Eigen::Matrix<float, 1, 4> det;
            det_xywh = detection->get_tlwh();
            det << det_xywh[0], det_xywh[1], det_xywh[2], det_xywh[3];
            measurements.emplace_back(det);
        }

        for (Eigen::Index i = 0; i < tracks.size(); i++) {
            Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance = KF.gating_distance(
                tracks[i]->mean, 
                tracks[i]->covariance,
                measurements, only_position
            );
            for (Eigen::Index j = 0; j < gating_distance.size(); j++) {
                if (gating_distance(0, j) > gating_threshold) {
                    cost_matrix(i, j) = std::numeric_limits<float>::infinity();
                }
                cost_matrix(i, j) = lambda * cost_matrix(i, j) + (1 - lambda) * gating_distance[j];
            }
        }
    }

    CostMatrix fuse_iou_with_emb(
        CostMatrix &iou_dist, 
        CostMatrix &emb_dist,
        const CostMatrix &iou_dists_mask,
        const CostMatrix &emb_dists_mask
    ) {
        if (emb_dist.rows() == 0 || emb_dist.cols() == 0) {
            // Embedding distance is not available, mask off iou distance
            for (Eigen::Index i = 0; i < iou_dist.rows(); i++) {
                for (Eigen::Index j = 0; j < iou_dist.cols(); j++) {
                    if (static_cast<bool>(iou_dists_mask(i, j))) {
                        iou_dist(i, j) = 1.0F;
                    }
                }
            }
            return iou_dist;
        }

        // If IoU distance is larger than threshold, don't use embedding at all
        for (Eigen::Index i = 0; i < iou_dist.rows(); i++) {
            for (Eigen::Index j = 0; j < iou_dist.cols(); j++) {
                if (static_cast<bool>(iou_dists_mask(i, j))) {
                    emb_dist(i, j) = 1.0F;
                }
            }
        }

        // If emb distance is larger than threshold, set the emb distance to inf
        for (Eigen::Index i = 0; i < emb_dist.rows(); i++) {
            for (Eigen::Index j = 0; j < emb_dist.cols(); j++) {
                if (static_cast<bool>(emb_dists_mask(i, j))) {
                    emb_dist(i, j) = 1.0F;
                }
            }
        }

        // Fuse iou and emb distance by taking the element-wise minimum
        CostMatrix  cost_matrix =
                Eigen::MatrixXf::Zero(iou_dist.rows(), iou_dist.cols());
        for (Eigen::Index i = 0; i < iou_dist.rows(); i++) {
            for (Eigen::Index j = 0; j < iou_dist.cols(); j++) {
                cost_matrix(i, j) = std::min(iou_dist(i, j), emb_dist(i, j));
            }
        }

        return cost_matrix;
    }

    AssociationData linear_assignment(
        CostMatrix &cost_matrix, float thresh
    ) {
        // If cost matrix is empty, all the tracks and detections are unmatched
        AssociationData associations;
        if (cost_matrix.size() == 0) {
            for (int i = 0; i < cost_matrix.rows(); i++) {
                associations.unmatched_track_indices.emplace_back(i);
            }

            for (int i = 0; i < cost_matrix.cols(); i++) {
                associations.unmatched_det_indices.emplace_back(i);
            }

            return associations;
        }

        std::vector<int> rowsol, colsol;
        double total_cost = lapjv(cost_matrix, rowsol, colsol, true, thresh);

        for (int i = 0; i < rowsol.size(); i++) {
            if (rowsol[i] >= 0) {
                associations.matches.emplace_back(i, rowsol[i]);
            } else {
                associations.unmatched_track_indices.emplace_back(i);
            }
        }

        for (int i = 0; i < colsol.size(); i++) {
            if (colsol[i] < 0) {
                associations.unmatched_det_indices.emplace_back(i);
            }
        }

        return associations;
    }

    class BotSort : public Tracker {
    public:
        BotSort(const TrackerConfig& config) : 
            gmc_enabled_(config.gmc_enabled), reid_enabled_(config.reid_enabled), 
            track_high_thresh_(config.track_high_thresh), track_low_thresh_(config.track_low_thresh), 
            new_track_thresh_(config.new_track_thresh), track_buffer_(config.track_buffer), 
            match_thresh_(config.match_thresh), proximity_thresh_(config.proximity_thresh), 
            appearance_thresh_(config.appearance_thresh),  lambda_(config.lambda), 
            frame_rate_(config.frame_rate), distance_metric_(config.distance_metric){

            buffer_size_ = static_cast<uint8_t>(frame_rate_ / 30.0 * track_buffer_);
            max_time_lost_ = buffer_size_;

            frame_id_ = 0;

            kalman_filter_ = std::make_unique<KalmanFilter>(static_cast<double>(1.0 / frame_rate_));

            if (gmc_enabled_) {
                gmc_ = std::make_unique<GMC>();
            }
        }

        virtual std::vector<TrackObject *> get_tracks() override {
            std::vector<TrackObject *> objects_ptr;
            for (auto& obj : objects_) {
                objects_ptr.push_back(obj.get());
            }
            return objects_ptr;
        }
        
        virtual void update(const std::vector<Detection> &detections, const cv::Mat &frame) override {
            // /*----------------CREATE TRACK OBJECT FOR ALL THE DETECTIONS----------------/
            // For all detections, extract features, create tracks and classify on the segregate of confidence
            frame_id_++;
            std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
            std::vector<std::shared_ptr<Track>> detections_high_conf,
                    detections_low_conf;
            detections_low_conf.reserve(detections.size()),
                    detections_high_conf.reserve(detections.size());

            if (!detections.empty())
            {
                for (Detection &detection:
                    const_cast<std::vector<Detection> &>(detections))
                {
                    detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
                    detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
                    detection.bbox_tlwh.width =
                            std::min(static_cast<float>(frame.cols - 1),
                                    detection.bbox_tlwh.width);
                    detection.bbox_tlwh.height =
                            std::min(static_cast<float>(frame.rows - 1),
                                    detection.bbox_tlwh.height);

                    std::shared_ptr<Track> tracklet;
                    std::vector<float> tlwh = {
                            detection.bbox_tlwh.x, detection.bbox_tlwh.y,
                            detection.bbox_tlwh.width, detection.bbox_tlwh.height};

                    if (detection.confidence > track_low_thresh_)
                    {
                        if (reid_enabled_)
                        {
                            FeatureMatrix embedding = detection.feature;
                            // TODO
                            std::shared_ptr<FeatureMatrix> embedding_ptr = std::make_shared<FeatureMatrix>(embedding);
                            tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id, embedding_ptr);
                        }
                        else
                            tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id);

                        if (detection.confidence >= track_high_thresh_)
                            detections_high_conf.push_back(tracklet);
                        else
                            detections_low_conf.push_back(tracklet);
                    }
                }
            }

            // Segregate tracks in unconfirmed and tracked tracks
            std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
            for (const std::shared_ptr<Track> &track: tracked_tracks_)
            {
                if (!track->is_activated)
                {
                    unconfirmed_tracks.push_back(track);
                }
                else
                {
                    tracked_tracks.push_back(track);
                }
            }
            // /*----------------CREATE TRACK OBJECT FOR ALL THE DETECTIONS----------------/

            // /*----------------Apply KF predict and GMC before running association algorithm----------------/
            // Merge currently tracked tracks and lost tracks
            std::vector<std::shared_ptr<Track>> tracks_pool;
            tracks_pool = _merge_track_lists(tracked_tracks, lost_tracks_);

            // Predict the location of the tracks with KF (even for lost tracks)
            Track::multi_predict(tracks_pool, *kalman_filter_);

            // Estimate camera motion and apply camera motion compensation
            if (gmc_enabled_) {
                Eigen::Matrix<float, 3, 3> H = gmc_->apply(frame, detections);
                Track::multi_gmc(tracks_pool, H);
                Track::multi_gmc(unconfirmed_tracks, H);
            }
            // /*----------------Apply KF predict and GMC before running association algorithm----------------/

            // /*----------------First association, with high score detection boxes----------------/
            // Find IoU distance between all tracked tracks and high confidence detections
            CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
                    emd_dist_mask_1st_association;

            std::tie(iou_dists, iou_dists_mask_1st_association) =
                    iou_distance(tracks_pool, detections_high_conf, proximity_thresh_);
            fuse_score(iou_dists,
                    detections_high_conf);// Fuse the score with IoU distance

            if (reid_enabled_)
            {
                // If re-ID is enabled, find the embedding distance between all tracked tracks and high confidence detections
                std::tie(raw_emd_dist, emd_dist_mask_1st_association) =
                        embedding_distance(tracks_pool, detections_high_conf,
                                        appearance_thresh_,
                                        "euclidean");
                fuse_motion(*kalman_filter_, raw_emd_dist, tracks_pool,
                            detections_high_conf,
                            lambda_);// Fuse the motion with embedding distance
            }

            // Fuse the IoU distance and embedding distance to get the final distance matrix
            CostMatrix distances_first_association = fuse_iou_with_emb(
                    iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
                    emd_dist_mask_1st_association);

            // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
            AssociationData first_associations =
                    linear_assignment(distances_first_association, match_thresh_);

            // Update the tracks with the associated detections
            for (const std::pair<int, int> &match: first_associations.matches)
            {
                const std::shared_ptr<Track> &track = tracks_pool[match.first];
                const std::shared_ptr<Track> &detection =
                        detections_high_conf[match.second];

                // If track was being actively tracked, we update the track with the new associated detection
                if (track->state == TrackState::Tracked)
                {
                    track->update(*kalman_filter_, *detection, frame_id_);
                    activated_tracks.push_back(track);
                }
                else
                {
                    // If track was not being actively tracked, we re-activate the track with the new associated detection
                    // NOTE: There should be a minimum number of frames before a track is re-activated
                    track->re_activate(*kalman_filter_, *detection, frame_id_, false);
                    refind_tracks.push_back(track);
                }
            }
            // /*----------------First association, with high score detection boxes----------------/

            // /*----------------Second association, with low score detection boxes----------------/
            // Get all unmatched but tracked tracks after the first association, these tracks will be used for the second association
            std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
            for (int track_idx: first_associations.unmatched_track_indices)
            {
                const std::shared_ptr<Track> &track = tracks_pool[track_idx];
                if (track->state == TrackState::Tracked)
                {
                    unmatched_tracks_after_1st_association.push_back(track);
                }
            }

            // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
            CostMatrix iou_dists_second;
            iou_dists_second = iou_distance(unmatched_tracks_after_1st_association,
                                            detections_low_conf);

            // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
            AssociationData second_associations =
                    linear_assignment(iou_dists_second, 0.5);

            // Update the tracks with the associated detections
            for (const std::pair<int, int> &match: second_associations.matches)
            {
                const std::shared_ptr<Track> &track =
                        unmatched_tracks_after_1st_association[match.first];
                const std::shared_ptr<Track> &detection =
                        detections_low_conf[match.second];

                // If track was being actively tracked, we update the track with the new associated detection
                if (track->state == TrackState::Tracked)
                {
                    track->update(*kalman_filter_, *detection, frame_id_);
                    activated_tracks.push_back(track);
                }
                else
                {
                    // If track was not being actively tracked, we re-activate the track with the new associated detection
                    // NOTE: There should be a minimum number of frames before a track is re-activated
                    track->re_activate(*kalman_filter_, *detection, frame_id_, false);
                    refind_tracks.push_back(track);
                }
            }

            // The tracks that are not associated with any detection even after the second association are marked as lost
            std::vector<std::shared_ptr<Track>> lost_tracks;
            for (int unmatched_track_index: second_associations.unmatched_track_indices)
            {
                const std::shared_ptr<Track> &track =
                        unmatched_tracks_after_1st_association[unmatched_track_index];
                if (track->state != TrackState::Lost)
                {
                    track->mark_lost();
                    lost_tracks.push_back(track);
                }
            }
            // /*----------------Second association, with low score detection boxes----------------/


            // /*----------------Deal with unconfirmed tracks----------------/
            std::vector<std::shared_ptr<Track>> unmatched_detections_after_1st_association;
            for (int detection_idx: first_associations.unmatched_det_indices)
            {
                const std::shared_ptr<Track> &detection =
                        detections_high_conf[detection_idx];
                unmatched_detections_after_1st_association.push_back(detection);
            }

            //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
            CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
                    iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed;

            std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) = iou_distance(
                    unconfirmed_tracks, unmatched_detections_after_1st_association,
                    proximity_thresh_);
            fuse_score(iou_dists_unconfirmed,
                    unmatched_detections_after_1st_association);

            if (reid_enabled_)
            {
                // Find embedding distance between unconfirmed tracks and high confidence detections left after the first association
                std::tie(raw_emd_dist_unconfirmed, emd_dist_mask_unconfirmed) =
                        embedding_distance(unconfirmed_tracks,
                                        unmatched_detections_after_1st_association,
                                        appearance_thresh_,
                                        "euclidean");
                fuse_motion(*kalman_filter_, raw_emd_dist_unconfirmed,
                            unconfirmed_tracks,
                            unmatched_detections_after_1st_association, lambda_);
            }

            // Fuse the IoU distance and the embedding distance
            CostMatrix distances_unconfirmed = fuse_iou_with_emb(
                    iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
                    iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed);

            // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
            AssociationData unconfirmed_associations =
                    linear_assignment(distances_unconfirmed, 0.7);

            for (const std::pair<int, int> &match: unconfirmed_associations.matches)
            {
                const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
                const std::shared_ptr<Track> &detection =
                        unmatched_detections_after_1st_association[match.second];

                // If the unconfirmed track is associated with a detection we update the track with the new associated detection
                // and add the track to the activated tracks list
                track->update(*kalman_filter_, *detection, frame_id_);
                activated_tracks.push_back(track);
            }

            // All the unconfirmed tracks that are not associated with any detection are marked as removed
            std::vector<std::shared_ptr<Track>> removed_tracks;
            for (int unmatched_track_index:
                unconfirmed_associations.unmatched_track_indices)
            {
                const std::shared_ptr<Track> &track =
                        unconfirmed_tracks[unmatched_track_index];
                track->mark_removed();
                removed_tracks.push_back(track);
            }
            // /*----------------Deal with unconfirmed tracks----------------/


            // /*----------------Initialize new tracks----------------/
            std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
            for (int detection_idx: unconfirmed_associations.unmatched_det_indices)
            {
                const std::shared_ptr<Track> &detection =
                        unmatched_detections_after_1st_association[detection_idx];
                unmatched_high_conf_detections.push_back(detection);
            }

            // Initialize new tracks for the high confidence detections left after all the associations
            for (const std::shared_ptr<Track> &detection:
                unmatched_high_conf_detections)
            {
                if (detection->get_score() >= new_track_thresh_)
                {
                    detection->activate(*kalman_filter_, frame_id_);
                    activated_tracks.push_back(detection);
                }
            }
            // /*----------------Initialize new tracks----------------/

            // /*----------------Update lost tracks state----------------/
            for (const std::shared_ptr<Track> &track: lost_tracks_)
            {
                if (frame_id_ - track->end_frame() > max_time_lost_)
                {
                    track->mark_removed();
                    removed_tracks.push_back(track);
                }
            }
            // /*----------------Update lost tracks state----------------/

            // /*----------------Clean up the track lists----------------/
            std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
            for (const std::shared_ptr<Track> &_tracked_track: tracked_tracks_)
            {
                if (_tracked_track->state == TrackState::Tracked)
                {
                    updated_tracked_tracks.push_back(_tracked_track);
                }
            }
            tracked_tracks_ =
                    _merge_track_lists(updated_tracked_tracks, activated_tracks);
            tracked_tracks_ = _merge_track_lists(tracked_tracks_, refind_tracks);

            lost_tracks_ = _merge_track_lists(lost_tracks_, lost_tracks);
            lost_tracks_ = _remove_from_list(lost_tracks_, tracked_tracks_);
            lost_tracks_ = _remove_from_list(lost_tracks_, removed_tracks);

            std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned,
                    lost_tracks_cleaned;
            _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned,
                                    tracked_tracks_, lost_tracks_);
            tracked_tracks_ = tracked_tracks_cleaned,
            lost_tracks_ = lost_tracks_cleaned;
            // /*----------------Clean up the track lists----------------/
            
            // /*----------------Update output tracks----------------/
            objects_.clear();
            for (const std::shared_ptr<Track> &track: tracked_tracks_) {
                if (track->is_activated) {
                    objects_.push_back(track);
                }
            }
            // /*----------------Update output tracks----------------/
        }

    private:
        static std::vector<std::shared_ptr<Track>> _merge_track_lists(
            std::vector<std::shared_ptr<Track>> &tracks_list_a,
            std::vector<std::shared_ptr<Track>> &tracks_list_b
        ) {
            std::map<int, bool> exists;
            std::vector<std::shared_ptr<Track>> merged_tracks_list;

            for (const std::shared_ptr<Track> &track: tracks_list_a) {
                exists[track->track_id] = true;
                merged_tracks_list.push_back(track);
            }

            for (const std::shared_ptr<Track> &track: tracks_list_b) {
                if (exists.find(track->track_id) == exists.end()) {
                    exists[track->track_id] = true;
                    merged_tracks_list.push_back(track);
                }
            }

            return merged_tracks_list;
        }

        static std::vector<std::shared_ptr<Track>> _remove_from_list(
            std::vector<std::shared_ptr<Track>> &tracks_list,
            std::vector<std::shared_ptr<Track>> &tracks_to_remove
        ) {
            std::map<int, bool> exists;
            std::vector<std::shared_ptr<Track>> new_tracks_list;

            for (const std::shared_ptr<Track> &track: tracks_to_remove) {
                exists[track->track_id] = true;
            }

            for (const std::shared_ptr<Track> &track: tracks_list) {
                if (exists.find(track->track_id) == exists.end()) {
                    new_tracks_list.push_back(track);
                }
            }
            return new_tracks_list;
        }

        static void _remove_duplicate_tracks(
            std::vector<std::shared_ptr<Track>> &result_tracks_a,
            std::vector<std::shared_ptr<Track>> &result_tracks_b,
            std::vector<std::shared_ptr<Track>> &tracks_list_a,
            std::vector<std::shared_ptr<Track>> &tracks_list_b
        ) {
            CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);
            std::unordered_set<size_t> dup_a, dup_b;
            for (Eigen::Index i = 0; i < iou_dists.rows(); i++) {
                for (Eigen::Index j = 0; j < iou_dists.cols(); j++) {
                    if (iou_dists(i, j) < 0.15) {
                        int time_a = static_cast<int>(tracks_list_a[i]->frame_id - tracks_list_a[i]->start_frame);
                        int time_b = static_cast<int>(tracks_list_b[j]->frame_id - tracks_list_b[j]->start_frame);

                        // We make an assumption that the longer trajectory is the correct one
                        if (time_a > time_b) {
                            dup_b.insert(j);// In list b, track with index j is a duplicate
                        } else {
                            dup_a.insert(i);// In list a, track with index i is a duplicate
                        }
                    }
                }
            }

            // Remove duplicates from the lists
            for (size_t i = 0; i < tracks_list_a.size(); i++) {
                if (dup_a.find(i) == dup_a.end()) {
                    result_tracks_a.push_back(tracks_list_a[i]);
                }
            }

            for (size_t i = 0; i < tracks_list_b.size(); i++) {
                if (dup_b.find(i) == dup_b.end()) {
                    result_tracks_b.push_back(tracks_list_b[i]);
                }
            }
        }

    private:
        bool reid_enabled_;

        bool gmc_enabled_;

        std::string distance_metric_;

        uint8_t track_buffer_;
        uint8_t frame_rate_;
        uint8_t buffer_size_;
        uint8_t max_time_lost_;

        unsigned int frame_id_;

        float track_high_thresh_;
        float track_low_thresh_;
        float new_track_thresh_;

        float match_thresh_;
        float proximity_thresh_;
        float appearance_thresh_;
        float lambda_;

        std::vector<std::shared_ptr<Track>> tracked_tracks_;
        std::vector<std::shared_ptr<Track>> lost_tracks_;

        std::unique_ptr<KalmanFilter> kalman_filter_;
        std::unique_ptr<GMC> gmc_;

        std::vector<std::shared_ptr<Track>> objects_;
    };

    std::shared_ptr<Tracker> create_tracker(const TrackerConfig& config) {
        std::shared_ptr<BotSort> tracker_ptr(new BotSort(config));
        return tracker_ptr;
    }

} // namespace BotSORT
