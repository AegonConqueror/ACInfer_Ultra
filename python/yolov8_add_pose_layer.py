
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parent

split_onnx_file = "./onnx/modified/yolo11s_silu_pose_split.onnx"
dst_onnx_file = "./onnx/modified/yolo11s_silu_pose_plugin.onnx"

input_width = 640
input_height = 640
max_output_boxes = 20
min_stride = 8
num_keypoints = 17
socre_threshold = 0.25
nms_threshold = 0.45

pose_layer_attrs = {
    "input_width": input_width,
    "input_height": input_width,
    "max_output_boxes": max_output_boxes,
    "max_output_boxes": max_output_boxes,
    "min_stride": min_stride,
    "num_keypoints": num_keypoints,
    "socre_threshold": socre_threshold,
    "nms_threshold": nms_threshold,
    "plugin_version": "1"
}

det_shape = (1, 1)
cls_shape = (1, max_output_boxes)
scores_shape = (1, max_output_boxes)
boxes_shape = (1, max_output_boxes, 4)
kps_shape = (1, max_output_boxes, 3 * 17)

num_detections = gs.Variable(name="NumDetections", dtype=np.int32, shape=det_shape)
detection_classes = gs.Variable(name="DetectionClasses", dtype=np.int32, shape=cls_shape)
detection_scores = gs.Variable(name="DetectionScores", dtype=np.float32, shape=scores_shape)
detection_boxes = gs.Variable(name="DetectionBoxes", dtype=np.float32, shape=boxes_shape)
detection_keypoints = gs.Variable(name="DetectionKeyPoints", dtype=np.float32, shape=kps_shape)

yolov8_pose_layer_outputs = [num_detections, detection_classes, detection_scores, detection_boxes, detection_keypoints]

# ======================================= 模型读取 =======================================
graph = gs.import_onnx(onnx.load(ROOT / split_onnx_file))

yolov8_pose_layer_node = gs.Node(
    name="yolov8_pose_layer",
    op="Yolov8PoseLayer",
    attrs=pose_layer_attrs,
    inputs=graph.outputs, 
    outputs=yolov8_pose_layer_outputs
)
graph.nodes.append(yolov8_pose_layer_node)

graph.outputs = yolov8_pose_layer_outputs

graph.cleanup()

onnx.save(gs.export_onnx(graph), ROOT / dst_onnx_file)
print('done!')
