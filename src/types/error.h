/**
 * *****************************************************************************
 * File name:   error.h
 * 
 * @brief  错误码定义
 * 
 * 
 * Created by Aegon on 2024-12-27
 * Copyright © 2024 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_ERROR_H
#define ACINFER_ULTRA_ERROR_H

typedef enum {
    SUCCESS             = 0,    // 成功
    DEVICE_INIT_FAIL    = -1,   // 硬件初始化失败
    FILE_READ_FAIL      = -2,   // 文件读取失败
    LOAD_MODEL_FAIL     = -3,   // 加载模型失败
    INPUT_ATTR_ERROR    = -4,   // 输入数据属性错误
    OUTPUT_ATTR_ERROR   = -5,   // 输出数据属性错误
    SET_INPUT_FAIL      = -6,   // 设置输入数据错误
    RUNTIME_ERROR       = -7,   // 运行时错误
    MODEL_NOT_LOAD      = -8,   // 模型未加载
    PROCESS_FAIL        = -9,   // 数据处理失败
    TIMEOUT             = -10,  // 超时
} error_e;

#endif // ACINFER_ULTRA_ERROR_H