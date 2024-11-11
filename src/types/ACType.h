/**
 * *****************************************************************************
 * File name:   error.h
 * 
 * @brief  错误码定义
 * 
 * 
 * Created by Aegon on 2023-06-28
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_ACTYPE_H
#define ACINFER_ULTRA_ACTYPE_H

#ifdef _WIN32
#ifdef IPEINF_EXPORTS
#define MS_DLL __declspec(dllexport)
#else
#define MS_DLL __declspec(dllimport)
#endif
#else
#define MS_DLL
#endif

#ifdef __cplusplus
extern "C" {
#endif
	typedef void* AC_HANDLE;

#ifdef __cplusplus
}
#endif

typedef enum {
    SUCCESS = 0,                 // 成功
    DEVICE_INIT_FAIL = -1,       // 硬件初始化失败
    FILE_READ_FAIL = -2,         // 文件读取失败
    LOAD_MODEL_FAIL = -3,        // 加载模型失败
    INPUT_ATTR_ERROR = -4,       // 输入数据属性错误
    OUTPUT_ATTR_ERROR = -5,      // 输出数据属性错误
    RUNTIME_ERROR = -6,          // 运行时错误
    MODEL_NOT_LOAD = -7,         // 模型未加载
} error_e;

#endif // ACINFER_ULTRA_ACTYPE_H