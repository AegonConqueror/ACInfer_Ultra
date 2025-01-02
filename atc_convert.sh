
source /home/aegon/Ascend/ascend-toolkit/latest/x86_64-linux/bin/setenv.bash

atc --model=./onnx/yolov8_person_gestures_like_only_n_relu_1220.onnx --framework=5 --output=./om/yolov8_person_gestures_like_only_n_relu_1220 --soc_version=OPTG --input_fp16_nodes="images" --enable_single_stream=true

chmod 664 ./om/yolov8_person_gestures_like_only_n_relu_1220.om