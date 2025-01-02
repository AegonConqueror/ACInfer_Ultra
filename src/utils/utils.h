/**
 * *****************************************************************************
 * File name:   utils.h
 * 
 * @brief  Aegon common utils
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACENGINE_UTILS_H
#define ACENGINE_UTILS_H

#include <string>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#define LOG_INFO(...)           iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Info,   __VA_ARGS__)
#define LOG_ERROR(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Error,   __VA_ARGS__)
#define LOG_WARNING(...)        iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Warning,   __VA_ARGS__)
#define LOG_FATAL(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Fatal,   __VA_ARGS__)
#define LOG_DEBUG(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Debug,   __VA_ARGS__)

namespace iLog {

    enum class LogLevel : int {
        Info    = 0,
        Error   = 1,
        Warning = 2,
        Fatal   = 3,
        Debug   = 4
    };

    std::string format(const char* fmt, ...);

    void set_log_level(LogLevel level);
    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);
   
} // namespace iLog

namespace iTools {
    
    template <typename T>
    T vectorProduct(const std::vector<T> &v) {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    std::string vector_shape_string(const std::vector<int64_t>& shape);

    /**
     * ! 需要释放内存防止内存泄漏
     */
    float* halfToFloat(void* pred_half, std::vector<int> shape);

} // namespace iTools


namespace iFile {
    
    std::string file_name(const std::string& path, bool include_suffix=true);
    std::vector<uint8_t> load_file(const std::string& file);
    std::string directory(const std::string& path);

    std::vector<std::string> find_files(const std::string& directory);
    void read_bin(const std::string &fileName, void *&inputBuff, uint32_t &fileSize);
    
} // namespace iFile

namespace iTime {
    std::string time_now();
    long long timestamp_now();
    double timestamp_now_float();
    
} // namespace iTime

namespace iDraw {
    void draw_box_label(
        cv::Mat &img, const cv::Rect_<float> box, 
        const int class_id, const float conf,
        const std::vector<std::string> &class_names={}
    );

    void draw_mask(cv::Mat &img, const cv::Rect_<float> &box, cv::Mat &roi);
} // namespace iDraw


#endif // ACENGINE_UTILS_H