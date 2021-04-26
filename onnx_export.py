import torch
import cv2
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

model_path = "models/mobilenet-v1-ssd-mp-0_675.pth"
label_path = "models/voc-model-labels.txt"

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)

net.load(model_path)
net.eval()
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

dummy_input = torch.randn(1, 3, 300, 300)
torch.onnx.export(predictor.core, dummy_input, "mb1-ssd.onnx", opset_version=11)
