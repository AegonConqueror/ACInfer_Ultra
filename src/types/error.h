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

typedef enum ac_error {
    AC_SUCCESS             =  0,   // 成功
    AC_DEVICE_INIT_FAIL    = -1,   // 硬件初始化失败
    AC_FILE_READ_FAIL      = -2,   // 文件读取失败
    AC_LOAD_MODEL_FAIL     = -3,   // 加载模型失败
    AC_INPUT_ATTR_ERROR    = -4,   // 输入数据属性错误
    AC_OUTPUT_ATTR_ERROR   = -5,   // 输出数据属性错误
    AC_SET_INPUT_FAIL      = -6,   // 设置输入数据错误
    AC_RUNTIME_ERROR       = -7,   // 运行时错误
    AC_MODEL_NOT_LOAD      = -8,   // 模型未加载
    AC_PROCESS_FAIL        = -9,   // 数据处理失败
    AC_TASK_CONFLICT       = -10,  // 任务冲突
    AC_TIMEOUT             = -11,  // 超时
} ac_error_e;

#endif // ACINFER_ULTRA_ERROR_H