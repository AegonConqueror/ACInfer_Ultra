
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from functools import reduce

head_num = 3
map_size = [80, 40, 20]
max_stride = 32
max_objects = 20

def generate_meshgrid():
    meshgrid = []
    for index in range(head_num):
        h = map_size[index]
        w = map_size[index]
        for i in range(h):
            for j in range(w):
                meshgrid.append(float(j + 0.5))
                meshgrid.append(float(i + 0.5))
    return np.array(meshgrid, dtype=np.float32)

meshgrid = gs.Constant(name="mesh_grid", values=generate_meshgrid())

pose_layer_attrs = {
    "max_objects": max_objects,
    "max_stride": max_stride,
    "socre_threshold": 0.45,
    "nms_threshold": 0.45,
    "plugin_version": "1"
}

det_shape = (1, 1)
cls_shape = (1, 20)
scores_shape = (1, 20)
boxes_shape = (1, 20, 4)
kps_shape = (1, 20, 3)

num_detections = gs.Variable(name="NumDetections", dtype=np.int32, shape=det_shape)
detection_classes = gs.Variable(name="DetectionClasses", dtype=np.int32, shape=cls_shape)
detection_scores = gs.Variable(name="DetectionScores", dtype=np.float32, shape=scores_shape)
detection_boxes = gs.Variable(name="DetectionBoxes", dtype=np.float32, shape=boxes_shape)
detection_keypoints = gs.Variable(name="DetectionKeyPoints", dtype=np.float32, shape=kps_shape)

yolov8_pose_layer_outputs = [num_detections, detection_classes, detection_scores, detection_boxes, detection_keypoints]


# ======================================= 模型读取 =======================================
graph = gs.import_onnx(onnx.load("./onnx/old/coco_pose_n_relu_person_face_0410.onnx"))
# print([*graph.outputs, meshgrid, headStarts])

yolov8_pose_layer_node = gs.Node(
    name="yolov8_pose_layer",
    op="Yolov8PoseLayer",
    attrs=pose_layer_attrs,
    inputs=[*graph.outputs, meshgrid], 
    outputs=yolov8_pose_layer_outputs
)
graph.nodes.append(yolov8_pose_layer_node)

graph.outputs = yolov8_pose_layer_outputs

graph.cleanup()

onnx.save(gs.export_onnx(graph), "./onnx/yolov8/yolov8_pose_plugin_player.onnx")
print('done!')
