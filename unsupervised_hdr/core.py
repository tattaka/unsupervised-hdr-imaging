import math
import os
import random
import re
import warnings
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .data import LDRDataset
from .models import EncoderDecoderModel, ImageSpaceLoss
from .tools import helper_functions

warnings.simplefilter("ignore")
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, int_classes, string_classes
else:
    import collections.abc as container_abcs

    int_classes = int
    string_classes = str


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UnsupervisedHDRModel:
    def __init__(
        self,
        video_path: str,
        checkpoint_path: str = None,
        encoder: str = "SimpleEncoder",
        decoder: str = "SimpleDecoder",
        encoder_pretrained: bool = None,
        encoder_lr: float = 1e-4,
        decoder_lr: float = 1e-4,
        num_worker: int = 4,
        device_ids: Union[str, int, List[int]] = None,
        output_dir: str = "./",
        seed: int = 0,
    ) -> None:
        self.video_path = video_path
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.num_worker = num_worker
        self.output_dir = output_dir
        self.seed = seed
        self.max_epoch = None
        self.iterator = 0
        self.epoch = 0

        seed = 0

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        g = torch.Generator()
        g.manual_seed(seed)

        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device_ids == "cpu":
                self.device_ids = list()
                self.device = torch.device("cpu")
            else:
                self.device_ids = helper_functions.input2list(device_ids)
                self.device = torch.device(f"cuda:{self.device_ids[0]}")
        self.build_model(encoder, decoder, encoder_pretrained)
        self.configure_optimizers()
        self.train_dataloader = None
        self.predict_dataloader = None
        self.initialize_logger()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def build_model(
        self, encoder: str, decoder: str, encoder_pretrained: bool = False
    ) -> None:
        self.model = EncoderDecoderModel(encoder, decoder, encoder_pretrained)
        self.mse_loss = nn.MSELoss()
        self.image_space_loss = ImageSpaceLoss()

    def build_dataset(
        self,
        rank: int,
        world_size: int,
        train: bool,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        # TODO: dataloader
        dataset = LDRDataset(self.video_path, train=train, image_size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(rank == -1 and train),
            sampler=None
            if rank == -1
            else DistributedSampler(dataset, rank=rank, num_replicas=world_size),
            num_workers=self.num_worker,
            worker_init_fn=seed_worker,
            pin_memory=True,
            drop_last=train,
        )
        return dataloader

    def initialize_logger(self) -> None:
        self.train_results = {}
        self.lr = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(os.path.join(self.output_dir, "checkpoints")):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"))
        # TODO: save hyper parameters
        self.train_results["best_loss"] = float("inf")
        self.result_info = ""

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns logits."""
        outputs = {}
        outputs["pred_delta"] = self.model(image)
        outputs["pred_Ib"] = image * outputs["pred_delta"]
        return outputs

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        losses["loss_delta"] = self.mse_loss(
            (batch["Ib"] + 1e-6) / (batch["Ih"] + 1e-6), outputs["pred_delta"]
        )
        losses["loss_image"] = self.image_space_loss(batch["Ib"], outputs["pred_Ib"])
        losses["loss"] = losses["loss_delta"] + losses["loss_image"]
        return losses

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int = None
    ) -> Dict[str, torch.Tensor]:
        step_output = {}
        outputs = self.forward(batch["Ih"])
        train_loss = self.loss(outputs, batch)
        step_output.update(train_loss)
        train_loss["loss"].backward()
        self.optimizer.step()
        return step_output

    def training_one_epoch(self, rank=-1, device=None) -> Dict[str, float]:
        self.model.train()
        losses = {}
        with tqdm(
            self.train_dataloader,
            position=0,
            leave=True,
            ascii=" ##",
            dynamic_ncols=True,
            disable=rank > 0,
        ) as t:
            for batch_idx, batch_data in enumerate(t):
                batch_data = self.cuda(batch_data, device=device if rank < 0 else rank)
                step_output = self.training_step(batch_data)
                t.set_description("Epoch %i Training" % self.epoch)
                print_losses = {}
                for key in step_output:
                    print_losses[key] = step_output[key].item()
                    if key in losses.keys():
                        losses[key].append(step_output[key].item())
                    else:
                        losses[key] = [step_output[key].item()]
                t.set_postfix(ordered_dict=dict(**print_losses))
                self.iterator += 1
        if rank <= 0:
            for key in losses:
                losses[key] = sum(losses[key]) / len(losses[key])
            if len(losses.values()) > 1 and not ("loss" in losses.keys()):
                losses["loss"] = sum(losses.values())
            for key in losses:
                self.train_results[key] = losses[key]
            if self.train_results["best_loss"] >= self.train_results["loss"]:
                self.train_results["best_loss"] = self.train_results["loss"]
                self.save_checkpoint(metric="loss")
            if self.epoch != 0:
                self.save_checkpoint()
            self.lr = {}
            self.lr["lr"] = [group["lr"] for group in self.optimizer.param_groups][0]
            self.result_info = ""
            for result_key, result_value in zip(
                self.train_results.keys(), self.train_results.values()
            ):
                self.result_info = (
                    self.result_info
                    + result_key
                    + ":"
                    + str(round(result_value, 4))
                    + " "
                )
            for lr_key, lr_value in zip(self.lr.keys(), self.lr.values()):
                self.result_info = (
                    self.result_info + lr_key + ":" + str(round(lr_value, 4)) + " "
                )
            print("Epoch %i" % self.epoch, self.result_info)
        self.epoch += 1
        return losses

    def fit_single(
        self, rank: int, max_epoch: int, world_size: int, batch_size: int = 1
    ) -> None:
        # TODO: early stopping
        self.train_dataloader = self.build_dataset(
            rank=rank,
            world_size=world_size,
            train=True,
            batch_size=batch_size,
        )
        use_ddp = world_size > 1
        if use_ddp:
            setup(rank, world_size)
            self.model = self.model.to(rank)
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = self.model.to(self.device)
        self.max_epoch = max_epoch
        for _ in range(max_epoch):
            self.training_one_epoch(rank=rank, device=self.device)
        if use_ddp:
            cleanup()

    def fit(self, max_epoch: int, batch_size: int = 1) -> None:
        if len(self.device_ids) > 1:
            mp.spawn(
                self.fit_single,
                args=(max_epoch, len(self.device_ids), batch_size),
                nprocs=len(self.device_ids),
                join=True,
            )
        else:
            self.fit_single(-1, max_epoch, len(self.device_ids), batch_size)

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int = None
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            delta = self.model(batch["Ib"])  # (bs, 3, h, w)
            Il_2 = batch["Ib"] * delta
            delta_2l = self.model(Il_2)
            Il_4 = Il_2 * delta_2l
            Ih_2 = batch["Ib"] / delta
            delta_2h = self.model(Ih_2)
            Ih_4 = Ih_2 / delta_2h
        exposure_list = [
            Il_4,
            Il_2,
            batch["Ib"],
            Ih_2,
            Ih_4,
        ]
        exposure_list = np.stack(
            [
                (img.clone().detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                for img in exposure_list
            ]
        ).transpose(
            1, 0, 3, 4, 2
        )  # (bs, 5, h, w, 3)

        merge_mertens = cv2.createMergeMertens()
        hdr_image = [merge_mertens.process(img_set) for img_set in exposure_list]
        output = {"exposure_list": exposure_list, "hdr_image": hdr_image}
        return output  # BGR

    def predict(
        self,
        frame_idx: Union[int, List[int]] = None,
        batch_size: int = 1,
        image_size=None,
    ) -> Dict[str, np.ndarray]:
        self.predict_dataloader = self.build_dataset(
            rank=-1,
            world_size=1,
            train=False,
            batch_size=batch_size,
            image_size=image_size,
        )
        best_checkpoint = os.path.join(self.output_dir, "checkpoints", "best_loss.pth")
        print(f"Start loading best checkpoint from {best_checkpoint}")
        self.load_checkpoint(best_checkpoint)
        self.cuda(self.model, device=self.device)
        self.model.eval()
        print("Finish loading!")
        output = {"exposure_list": [], "hdr_image": []}
        if frame_idx is None:
            for batch in tqdm(self.predict_dataloader):
                batch = self.cuda(batch, self.device)
                output_batch = self.predict_step(batch)
                output["exposure_list"].append(output_batch["exposure_list"])
                output["hdr_image"].append(output_batch["hdr_image"])
        else:
            frame_idx = helper_functions.input2list(frame_idx)
            for i in tqdm(
                range(math.ceil(len(frame_idx) / self.predict_dataloader.batch_size))
            ):
                batch = []
                for f in frame_idx[
                    i
                    * self.predict_dataloader.batch_size : (i + 1)
                    * self.predict_dataloader.batch_size
                ]:
                    batch.append(self.predict_dataloader.dataset[f])
                batch = self.cuda(batch, self.device)
                if len(batch) == 1:
                    batch = {k: b.unsqueeze(0) for k, b in batch[0].items()}
                else:
                    batch = helper_functions.concat_data(batch)
                output_batch = self.predict_step(batch)
                output["exposure_list"].append(output_batch["exposure_list"])
                output["hdr_image"].append(output_batch["hdr_image"])
        output["exposure_list"] = np.concatenate(output["exposure_list"])
        output["hdr_image"] = np.concatenate(output["hdr_image"])
        return output

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
            )["model_state_dict"]
        )

    def save_checkpoint(self, metric=None):
        checkpoint = {}
        if metric is None:
            file_path = os.path.join(self.output_dir, "checkpoints", "last.pth")
        else:
            file_path = os.path.join(
                self.output_dir, "checkpoints", "best_" + metric + ".pth"
            )
            checkpoint = {
                "best_epoch": self.epoch,
                "best_" + metric: self.train_results["best_" + metric],
            }
        checkpoint["model_state_dict"] = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        torch.save(checkpoint, file_path)

    def configure_optimizers(self) -> optim.Optimizer:

        self.optimizer = optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.model.decoder.parameters(), "lr": self.decoder_lr},
            ],
            lr=self.decoder_lr,
            weight_decay=0.0001,
        )

    def cuda(self, x, device=None):
        np_str_obj_array_pattern = re.compile(r"[SaUO]")
        if torch.cuda.is_available():
            if isinstance(x, torch.Tensor):
                x = x.cuda(non_blocking=True, device=device)
                return x
            elif isinstance(x, nn.Module):
                x = x.cuda(device=device)
                return x
            elif isinstance(x, np.ndarray):
                if x.shape == ():
                    if np_str_obj_array_pattern.search(x.dtype.str) is not None:
                        return x
                    return self.cuda(torch.as_tensor(x), device=device)
                return self.cuda(torch.from_numpy(x), device=device)
            elif isinstance(x, float):
                return self.cuda(torch.tensor(x, dtype=torch.float64), device=device)
            elif isinstance(x, int_classes):
                return self.cuda(torch.tensor(x), device=device)
            elif isinstance(x, string_classes):
                return x
            elif isinstance(x, container_abcs.Mapping):
                return {key: self.cuda(x[key], device=device) for key in x}
            elif isinstance(x, container_abcs.Sequence):
                return [
                    self.cuda(np.array(xi), device=device)
                    if isinstance(xi, container_abcs.Sequence)
                    else self.cuda(xi, device=device)
                    for xi in x
                ]

    def to_cpu(self, x):
        if isinstance(x, torch.Tensor) and x.device != "cpu":
            return x.clone().detach().cpu()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, container_abcs.Mapping):
            return {key: self.to_cpu(x[key]) for key in x}
        elif isinstance(x, container_abcs.Sequence):
            return [self.to_cpu(xi) for xi in x]
        else:
            return x
