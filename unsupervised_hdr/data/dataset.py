from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils import data


def exposure(img: torch.Tensor, v: float) -> torch.Tensor:
    return (img * (2**v) ** (1 / 2.2)).clamp(0, 255).to(torch.uint8).to(torch.float32)


class LDRDataset(data.Dataset):
    def __init__(
        self, video_path: str, train: bool = False, image_size: Tuple[int, int] = None
    ) -> None:
        self.video_path = video_path
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError
        self.cap_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.train = train
        self.image_size = image_size

    def __len__(self) -> int:
        return self.cap_frame_num

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if self.image_size is not None:
            frame = cv2.resize(frame, dsize=self.image_size)
        frame = torch.as_tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
        if self.train:
            s = np.random.rand() * 0.25
        else:
            s = 0
        frame = exposure(frame, s)
        frame_high = exposure(frame, 2 + s)
        return {"Ib": frame / 255, "Ih": frame_high / 255}
