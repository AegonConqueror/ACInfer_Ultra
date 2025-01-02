
source /home/aegon/Ascend/ascend-toolkit/latest/x86_64-linux/bin/setenv.bash

atc --model=./onnx/yolov8_gestures_like_fist_n_relu_1217.onnx --framework=5 --output=./om/yolov8_gestures_like_fist_n_relu_1217 --soc_version=OPTG --input_fp16_nodes="data" --enable_single_stream=true

chmod 664 ./om/yolov8_gestures_like_fist_n_relu_1217.om