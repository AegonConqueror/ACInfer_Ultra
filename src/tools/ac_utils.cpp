#include "ac_utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stddef.h>
#include <stdarg.h>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace iLog {
    static LogLevel g_level = LogLevel::Warning;

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

        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            abort();
        }
    }
} // namespace iLog

namespace iTools {

    std::string align_blank(const std::string& input, int align_size, char blank) {
        if(input.size() >= align_size) return input;
        std::string output = input;
        for(int i = 0; i < align_size - input.size(); ++i)
            output.push_back(blank);
        return output;
    }
    
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

    std::string str_format(const char* fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

} // namespace iTools

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

    std::vector<std::string> find_files(const std::string& directory){
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

    std::vector<std::vector<float>> readLabelFile(const std::string& file_path) {
        std::vector<std::vector<float>> data;
        std::ifstream infile(file_path);
        
        if (!infile.is_open()) {
            std::cerr << "Failed to open file: " << file_path << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::vector<float> row;
            float value;
            while (iss >> value) {
                row.push_back(value);
            }
            if (!row.empty()) {
                data.push_back(row);
            }
        }

        infile.close();
        return data;
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

