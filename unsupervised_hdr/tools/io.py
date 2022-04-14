import os
import sys
from typing import List

import cv2
import numpy as np
from tqdm.auto import tqdm


def write_hdr_images(dir_path: str, image_list: List[np.ndarray]) -> None:
    os.makedirs(dir_path, exist_ok=True)
    for i, image in enumerate(tqdm(image_list)):
        cv2.imwrite(
            os.path.join(dir_path, str(i).zfill(len(str(len(image_list)))) + ".hdr"),
            image,
        )


def write_exr_images(dir_path: str, image_list: List[np.ndarray]) -> None:
    os.makedirs(dir_path, exist_ok=True)
    raise NotImplementedError


def write_hdr_to_mp4(video_path: str, image_list: List[np.ndarray], fps: float) -> None:
    image_size = image_list[0].shape[:2][::-1]
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(video_path, fourcc, fps, image_size)
    if not video.isOpened():
        print("can't be opened video file")
        sys.exit()
    for image in tqdm(image_list):
        image = (image * 255).clip(0, 255).astype(np.uint8)
        video.write(image)
    video.release()
