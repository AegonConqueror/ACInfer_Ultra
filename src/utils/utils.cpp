
#include "utils.h"
#include <stddef.h>
#include <fstream>
#include <stdarg.h>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <random>

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
    
    float* halfToFloat(void* pred_half, std::vector<int> shape) {
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

    bool mkdir(const std::string& path){
        return ::mkdir(path.c_str(), 0755) == 0;
    }

    bool mkdirs(const std::string& path){
        if (path.empty()) return false;
        if (exists(path)) return true;

        std::string _path = path;
        char* dir_ptr = (char*)_path.c_str();
        char* iter_ptr = dir_ptr;
        
        bool keep_going = *iter_ptr not_eq 0;
        while (keep_going){
            if (*iter_ptr == 0)
                keep_going = false;
            
            if ((*iter_ptr == '/' and iter_ptr not_eq dir_ptr) or *iter_ptr == 0){
                char old = *iter_ptr;
                *iter_ptr = 0;
                if (!exists(dir_ptr)){
                    if (!mkdir(dir_ptr)){
                        if(!exists(dir_ptr)){
                            LOG_ERROR("mkdirs %s return false.", dir_ptr);
                            return false;
                        }
                    }
                }
                *iter_ptr = old;
            }
            iter_ptr++;
        }
        return true;
    }

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

    bool exists(const std::string& path){
        return access(path.c_str(), R_OK) == 0;
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

    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs){
        if (mk_dirs){
            int p = (int)file.rfind('/');
            if (p not_eq -1){
                if (!mkdirs(file.substr(0, p)))
                    return false;
            }
        }
        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data and length > 0){
            if (fwrite(data, 1, length, f) not_eq length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }

    bool save_file(const std::string& file, const std::vector<uint8_t>& data, bool mk_dirs){
        return save_file(file, data.data(), data.size(), mk_dirs);
    }

    bool save_file(const std::string& file, const std::string& data, bool mk_dirs){
        return save_file(file, data.data(), data.size(), mk_dirs);
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

namespace iDraw {

    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };

    void draw_box_label(
        cv::Mat &img, const cv::Rect_<float> box, 
        const int class_id, const float conf,
        const std::vector<std::string> &class_names
    ) {
        const unsigned char* color = colors[class_id % 80];
        cv::Scalar box_color(color[0], color[1], color[2]);
        cv::Scalar text_color(255, 255, 255);
        
        int lw = std::max(static_cast<int>(round((img.rows + img.cols) / 2.0 * 0.002)), 2);
        int tf = std::max(lw - 1, 1);
        float sf = lw / 3.0;
        cv::rectangle(img, box, box_color, lw, cv::LINE_AA);
        char draw_string[100];
        if (class_names.empty()) {
            sprintf(draw_string, "id%02d: %.2f", class_id, conf);
        } else {
            std::string class_name  = class_names[class_id];
            sprintf(draw_string, "%s: %.2f", class_name.c_str(), conf);
        }
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(draw_string, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &baseline);
        int w = text_size.width;
        int h = text_size.height;
        bool outside = box.y - h >= 3;
        cv::Point text_origin = outside ? cv::Point(box.x, box.y - h - 3) : cv::Point(box.x, box.y + h + 3);
        cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(box.x + w, text_origin.y), box_color, -1, cv::LINE_AA);
        cv::Point t = outside ? cv::Point(box.x, box.y - 2) : cv::Point(box.x, box.y + h + 2);
        cv::putText(img, draw_string, t, cv::FONT_HERSHEY_SIMPLEX, sf, text_color, tf, cv::LINE_AA);
    }

    void draw_mask(cv::Mat &img, const cv::Rect_<float> &box, cv::Mat &roi) {
        cv::Mat seg_mask_item = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        cv::Scalar seg_color(dis(gen), dis(gen), dis(gen));

        seg_mask_item(box).setTo(seg_color, roi);
        cv::addWeighted(img, 1, seg_mask_item, 0.45, 0, img);
    }
} // namespace iDraw

