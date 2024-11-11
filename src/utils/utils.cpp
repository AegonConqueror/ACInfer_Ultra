
#include "utils.h"

#include <fstream>
#include <stdarg.h>
#include <chrono>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace iLog {

    static LogLevel g_level = LogLevel::Warning;

    std::string format(const char* fmt, ...){
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

    void set_log_level(LogLevel level){
        g_level = level;
    }

    const char* level_string(LogLevel level){
        switch (level){
            case LogLevel::Info    : return "info";
            case LogLevel::Error   : return "error";
            case LogLevel::Warning : return "warning";
            case LogLevel::Fatal   : return "fatal";
            case LogLevel::Debug   : return "debug";
            default : return "unknow";
        }
    }

    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){
        if (level > g_level) return;
        
        va_list vl;
        va_start(vl, fmt);

        char buffer[2048];
        
        auto now = iTime::time_now();
        std::string filename = iFile::file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Warning) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Info) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]", level_string(level));
        }else {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
        }

        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
        fprintf(stdout, "%s\n", buffer);
    }

} // namespace iLog

namespace iTools {
    
    std::string vector_shape_string(const std::vector<int64_t>& shape){
        std::string result;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == shape.size() - 1) {
                result += std::to_string(shape[i]);
            } else {
                result += std::to_string(shape[i]) + "x";
            }
        }
        return result;
    }
    
    float* halfToFloat(void* pred_half, std::vector<int> shape){
        size_t size = 1;
        for (auto dim : shape){
            size = size * dim;
        }
        
        cv::Mat float16MatFromHalf(1, size, CV_16F, pred_half);
        cv::Mat float32MatBack;
        float16MatFromHalf.convertTo(float32MatBack, CV_32F);

        float* pred_float = new float[size];
        memcpy(pred_float, float32MatBack.ptr<float>(), size * sizeof(float));
        return pred_float;
    }

} // namespace iTools

namespace iFile {
    std::string file_name(const std::string& path, bool include_suffix){
        if (path.empty()) return "";

        int p = path.rfind('/');
        p += 1;

        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    std::vector<uint8_t> load_file(const std::string& file){
        std::ifstream in(file, std::ios::in | std::ios::binary);
        if (!in.is_open())
            return {};

        in.seekg(0, std::ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0){
            in.seekg(0, std::ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }

    bool isJpg(const std::string& filename) {
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        return ext == "jpg" || ext == "jpeg" || ext == "png";
    }

    std::vector<std::string> find_files( const std::string& directory ){
        std::vector<std::string> imageFiles;
        struct dirent *entry;
        DIR *dir = opendir(directory.c_str());

        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            struct stat s;
            std::string fullPath = directory + "/" + filename;
            if (stat(fullPath.c_str(), &s) == 0 && S_ISREG(s.st_mode)){
                if (isJpg(filename)) {
                    imageFiles.push_back(fullPath);
                }
            }
        }

        closedir(dir);
        return imageFiles;
    }

    std::string directory(const std::string& path){
        if (path.empty())
            return ".";

        int p = path.rfind('/');
        if(p == -1)
            return ".";

        return path.substr(0, p + 1);
    }

    void read_bin(const std::string &fileName, void *&inputBuff, uint32_t &fileSize){
        std::ifstream binFile(fileName, std::ifstream::binary);
        binFile.seekg(0, binFile.end);
        uint32_t binFileBufferLen = binFile.tellg();
        binFile.seekg(0, binFile.beg);
 
        inputBuff = malloc(binFileBufferLen);
        binFile.read(static_cast<char*>(inputBuff), binFileBufferLen);
        binFile.close();
        fileSize = binFileBufferLen;
    }
    
} // namespace iFile

namespace iTime {

    std::string time_now(){
        char time_string[20];
        time_t timep;
        time(&timep);
        tm& t = *(tm*)localtime(&timep);

        sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
        return time_string;
    }

    long long timestamp_now(){
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }

    double timestamp_now_float(){
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count() / 1000.0;
    }
    
} // namespace iTime

