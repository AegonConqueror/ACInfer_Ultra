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

} // namespace iTools


namespace iFile {

    bool mkdir(const std::string& path);

    bool mkdirs(const std::string& path);
    
    /**
     * @brief  从给定路径中提取文件名
     */
    std::string file_name(const std::string& path, bool include_suffix=true);

    /**
     * @brief  查看文件吃否存在
     */
    bool exists(const std::string& path);

    /**
     * @brief 加载模型文件
     */
    std::vector<uint8_t> load_file(const std::string& file);


    std::string directory(const std::string& path);

    /**
     * @brief 在指定目录中查找图片文件
     */
    std::vector<std::string> find_files(const std::string& directory);
    
    void read_bin(const std::string &fileName, void *&inputBuff, uint32_t &fileSize);

    std::vector<std::vector<float>> readLabelFile(const std::string& file_path);

    /**
     * @brief 保存文件
     */
    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);
    bool save_file(const std::string& file, const std::vector<uint8_t>& data, bool mk_dirs = true);
    bool save_file(const std::string& file, const std::string& data, bool mk_dirs = true);
    
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