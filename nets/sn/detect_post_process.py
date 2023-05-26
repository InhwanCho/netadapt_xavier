import torch

from torch import nn
from torchvision.ops import nms


class DetectPostProcess(nn.Module):
    def __init__(self,
                 anchor,
                 th_conf=0.5,
                 th_iou=0.5,
                 logistic='softmax',
                 max_num_detection=None):
        super().__init__()

        self.anchor = anchor

        self.th_conf = th_conf
        self.th_iou = th_iou

        if logistic == 'softmax':
            self.regression = nn.Softmax(dim=2)
        elif logistic == 'sigmoid':
            self.regression = nn.Sigmoid()
        else:
            self.regression = None

        self.max_num_detection = max_num_detection

    def forward(self, x):
        conf, loc = x

        num_cls = conf.size(2)

        if self.regression:
            conf = self.regression(conf)

        return self.nms(self.anchor, num_cls, conf, loc)

    def nms(self, anchor, num_cls, score, loc):
        th_conf = self.th_conf
        th_iou = self.th_iou

        has_bg = 1 if isinstance(self.regression, nn.Softmax) else 0
        batch_size = score.size(0)

        box = anchor.decode(loc)

        batches = []
        for b in range(0, batch_size):
            classes = []

            for i in range(has_bg, num_cls):
                mask = score[b][:, i] >= th_conf

                _box = box[b][mask]
                _score = score[b][mask, i]

                if self.max_num_detection is not None:
                    topk = min(self.max_num_detection, _score.numel())

                    _, selected = _score.topk(topk)

                    _box = _box[selected]
                    _score = _score[selected]

                idx = nms(_box, _score, th_iou)
                objs = torch.cat((_box[idx], _score[idx].unsqueeze(1)), 1)

                classes.append(objs.tolist())

            batches.append(classes)

        return batches

