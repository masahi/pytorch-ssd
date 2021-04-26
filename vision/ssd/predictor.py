import torch
from torch import nn
import torchvision

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class PredictorCore(nn.Module):
    def __init__(
        self,
        net,
        candidate_size,
        iou_threshold,
        score_threshold
    ):
        super().__init__()
        self.net = net
        self.candidate_size = candidate_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def forward(self, images):
        with torch.no_grad():
            scores, boxes = self.net.forward(images)

        boxes = boxes[0]
        scores = scores[0]

        scores_single_class, labels = torch.max(scores[:, 1:], dim=1)
        scores_topk, topk_indices = torch.topk(scores_single_class, self.candidate_size)
        boxes_topk = boxes[topk_indices]
        labels_topk = labels[topk_indices]

        mask = scores_topk > self.score_threshold
        boxes_thres = boxes_topk[mask]
        scores_thres = scores_topk[mask]
        labels = labels_topk[mask]

        selected_indices = torchvision.ops.batched_nms(
            boxes_thres[:, :4], scores_thres, labels, self.iou_threshold
        )

        selected_boxes = boxes_thres[selected_indices, :4]
        selected_boxes_prob = scores_thres[selected_indices]
        selected_labels = labels[selected_indices] + 1

        return selected_boxes, selected_labels, selected_boxes_prob


class Predictor(nn.Module):
    def __init__(
        self,
        net,
        size,
        mean=0.0,
        std=1.0,
        nms_method=None,
        iou_threshold=0.45,
        filter_threshold=0.01,
        candidate_size=200,
        sigma=0.5,
        device=None,
        score_threshold=0.0
    ):
        super().__init__()
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.core = PredictorCore(self.net, candidate_size, iou_threshold, score_threshold)

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(
                box_probs,
                self.nms_method,
                score_threshold=prob_threshold,
                iou_threshold=self.iou_threshold,
                sigma=self.sigma,
                top_k=top_k,
                candidate_size=self.candidate_size,
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return (
            picked_box_probs[:, :4],
            torch.tensor(picked_labels),
            picked_box_probs[:, 4],
        )

    def forward(self, image):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)

        selected_boxes, selected_labels, selected_boxes_prob = self.core(images)
        selected_boxes[:, 0] *= width
        selected_boxes[:, 1] *= height
        selected_boxes[:, 2] *= width
        selected_boxes[:, 3] *= height
        return selected_boxes, selected_labels, selected_boxes_prob
