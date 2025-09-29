/**
 * *****************************************************************************
 * File name:   ac_utils.h
 * 
 * @brief  Aegon common utils
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_AC_UTILS_H
#define ACINFER_ULTRA_AC_UTILS_H

#include <string>
#include <vector>
#include <numeric>

#define LOG_INFO(...)           iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Info, __VA_ARGS__)
#define LOG_ERROR(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Error, __VA_ARGS__)
#define LOG_WARNING(...)        iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Warning, __VA_ARGS__)
#define LOG_FATAL(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Fatal, __VA_ARGS__)
#define LOG_DEBUG(...)          iLog::__log_func(__FILE__, __LINE__, iLog::LogLevel::Debug, __VA_ARGS__)

namespace iLog {
    enum class LogLevel : int {
        Info    = 0,
        Error   = 1,
        Warning = 2,
        Fatal   = 3,
        Debug   = 4
    };

    void set_log_level(LogLevel level);

    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);

} // namespace iLog

namespace iTools {

    std::string align_blank(const std::string& input, int align_size, char blank=' ');

    std::string str_format(const char* fmt, ...);
    
    template <typename T>
    T vector_shape_numel(const std::vector<T>& v) {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    std::string vector_shape_string(const std::vector<int64_t>& shape);

} // namespace iTools

namespace iTime {

    std::string time_now();

    long long timestamp_now();

    double timestamp_now_float();
    
} // namespace iTime

namespace iFile {

    bool mkdir(const std::string& path);

    bool mkdirs(const std::string& path);
    
    /**
     * @brief  提取文件名
     */
    std::string file_name(const std::string& path, bool include_suffix=true);

    /**
     * @brief  查看文件吃否存在
     */
    bool exists(const std::string& path);

    /**
     * @brief 加载文件
     */
    std::vector<uint8_t> load_file(const std::string& file);


    std::string directory(const std::string& path);

    /**
     * @brief 在指定目录中查找图片文件
     */
    std::vector<std::string> find_files(const std::string& directory);
    
    void read_bin(const std::string& fileName, void *&inputBuff, uint32_t& fileSize);

    std::vector<std::vector<float>> readLabelFile(const std::string& file_path);

    /**
     * @brief 保存文件
     */
    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);
    bool save_file(const std::string& file, const std::vector<uint8_t>& data, bool mk_dirs = true);
    bool save_file(const std::string& file, const std::string& data, bool mk_dirs = true);
    
} // namespace iFile

#endif // ACINFER_ULTRA_AC_UTILS_H