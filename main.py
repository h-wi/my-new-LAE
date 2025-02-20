# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from absl import logging, flags
from timm.utils.model_ema import ModelEmaV2
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from libml.core import App, Conf, DataModule, Module, ModuleConf
from libml.data import build_imagenet_transform
from libml.model.pet import Adapter, STAdapter, Conv2dAdapter, KVLoRA, Prefix
from libml.model.utils import freeze, unfreeze
from libml.model.backbone.swin import SwinTransformer
from libml.model.backbone.convnext import ConvNeXt


@dataclass
class MyModuleConf(ModuleConf):
    adapt_blocks: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pet_cls: str = "Adapter"  # enum: Adapter, LoRA, Prefix
    pet_type : List[str] = field(default_factory=lambda: ["Adapter", "Conv2dAdapter"])
    pet_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    num_emas: int = 1
    ema_decay: float = 0.9999
    num_freeze_epochs: int = 3
    eval_only_emas: bool = False


@dataclass
class MyConf(Conf):
    module: MyModuleConf = MyModuleConf()


class MyDataModule(DataModule):
    def train_transform(self):
        return build_imagenet_transform(train=True, norm=False)

    def val_transform(self):
        return build_imagenet_transform(train=False, norm=False)


class MyModule(Module):
    pets: nn.Module  # online pet modules
    pets_emas: nn.ModuleList  # offline pet modules
    original_vit: nn.Module
    train_acc: Accuracy
    loss_fn: nn.Module

    def setup_model(self):
        super().setup_model()

        if getattr(self, "pets_emas", None) is None:
            freeze(self.model.backbone)
            self.pets_emas = nn.ModuleList([]) # offline (학습 X, 지식 축적 대상)
            self.pets = self.create_pets() # onilne, adapter, LoRA, Prefix 등이 n개 구성
            logging.info(f"==> pets:\n{self.pets}")
        elif len(self.pets_emas) < self.cfg.num_emas: # init EMA for knowledge Accumulation
            idx = len(self.pets_emas)
            ema = ModelEmaV2(self.pets, decay=self.cfg.ema_decay)
            self.pets_emas.append(ema)

        self.train_acc = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

        self.attach_pets(self.pets) # ViT에 PET 붙이기 (e.g. Adapter를 n번째 블럭까지)

    def create_pets_swin(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        kwargs = self.cfg.pet_kwargs
        adapters = [
            Adapter(blocks[idx].dim, **kwargs) for idx in self.cfg.adapt_blocks
        ]
        return nn.ModuleList(adapters)

    def create_pets_convnext(self):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [list(stage.blocks.children()) for stage in stages.children()]
        blocks = sum(blocks, [])
        adapters = []
        for idx in self.cfg.adapt_blocks:
            dim = blocks[idx].conv_dw.in_channels
            adapter = Conv2dAdapter(dim, dim, **self.cfg.pet_kwargs)
            adapters.append(adapter)
        return nn.ModuleList(adapters)

    def create_pets_vit(self):
        assert self.cfg.pet_cls in [ "Adapter", "LoRA", "Conv2dAdapter", "STAdapter", "myadapter", "Prefix"]

        n = len(self.cfg.adapt_blocks)
        embed_dim = self.model.backbone.embed_dim
        moe = self.cfg.model.backbone_kwargs.moe
        expert_num = self.cfg.model.backbone_kwargs.expert

        kwargs = dict(**self.cfg.pet_kwargs)
        if self.cfg.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            if moe:
                adapter_list = []
                for k in range(n):
                    tmp = nn.ModuleList()
                    for _ in range(expert_num):
                        tmp.append(Adapter(**kwargs, moe=moe)) # initializes one adapter module
                    adapter_list.append(tmp)
                return nn.ModuleList([ i for i in adapter_list])
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "STAdapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([STAdapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            return nn.ModuleList([KVLoRA(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == 'Conv2dAdapter':
            return nn.ModuleList([Conv2dAdapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == 'myadapter':
            using = self.cfg.pet_type
            pet = nn.ModuleList()
            for name in using:
                if name == 'Adapter':
                    kwargs["embed_dim"] = embed_dim
                    pet.append(nn.ModuleList([Adapter(**kwargs) for _ in range(n)]))
                elif name == 'STAdapter':
                    kwargs["embed_dim"] = embed_dim
                    pet.append(nn.ModuleList([STAdapter(**kwargs) for _ in range(n)]))
                elif name == 'Conv2dAdapter':
                    kwargs["embed_dim"] = embed_dim
                    pet.append(nn.ModuleList([Conv2dAdapter(**kwargs) for _ in range(n)]))
                else:
                    raise NotImplementedError
            return pet

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])

    def create_pets(self):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.create_pets_swin()

        if isinstance(self.model.backbone, ConvNeXt):
            return self.create_pets_convnext()

        return self.create_pets_vit()

    def attach_pets_swin(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        blocks = []
        for layer in self.model.backbone.layers:
            blocks += list(layer.blocks.children())

        for i, b in enumerate(self.cfg.adapt_blocks):
            blocks[b].attach_adapter(attn=pets[i])

    def attach_pets_convnext(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls == "Adapter", "Not implemented PET"

        n = len(self.cfg.adapt_blocks)
        stages = self.model.backbone.stages
        blocks = [
            [(idx, stage) for idx, _ in enumerate(stage.blocks.children())]
            for stage in stages.children()
        ]
        blocks = sum(blocks, [])
        for i, b in enumerate(self.cfg.adapt_blocks):
            idx, stage = blocks[b]
            stage.attach_adapter(**{f"blocks.{idx}": pets[i]})

    def attach_pets_vit(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls in ["myadapter", "Adapter", "STAdapter", "LoRA", "Conv2dAdapter", "Conv2dLoRA", "Prefix"]

        if self.cfg.pet_cls == "Adapter" or self.cfg.pet_cls == "Conv2dAdapter":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attach_adapter(attn=pets[i])
            return

        if self.cfg.pet_cls == "STAdapter":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attach_adapter(attn=pets[i])
            return

        if self.cfg.pet_cls == "LoRA" or self.cfg.pet_cls == "Conv2dLoRA":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_adapter(qkv=pets[i])
            return

        if self.cfg.pet_cls == 'myadapter':
            for i, b in enumerate(self.cfg.adapt_blocks):
                # self.cfg.pet_type = [Adapter, Conv2dAdapter]를 가정 *순서*
                self.model.backbone.blocks[b].attach_adapter(mlp=pets[0][i]) # @ MLP
                self.model.backbone.blocks[b].attach_adapter(attn=pets[1][i]) # @ MSA
            return

        for i, b in enumerate(self.cfg.adapt_blocks):
            self.model.backbone.blocks[b].attn.attach_prefix(pets[i])

    def attach_pets(self, pets: nn.ModuleList):
        if isinstance(self.model.backbone, SwinTransformer):
            return self.attach_pets_swin(pets)

        if isinstance(self.model.backbone, ConvNeXt):
            return self.attach_pets_convnext(pets)

        return self.attach_pets_vit(pets)

    def filter_state_dict(self, state_dict):
        ps = ("model_wrap.", "model.backbone.")
        return {k: v for k, v in state_dict.items() if not k.startswith(ps)}

    def forward(self, input):
        return super().forward(input)

    def pre_train_epoch(self): # Current Epoch(hyperparameter)에 따라서, Online PET를 freeze하거나 unfreeze한다.
        self.train_acc.reset()

        if self.current_task == 0 or self.cfg.num_freeze_epochs < 1:
            return

        if self.current_epoch == 0:
            freeze(self.pets)
            logging.info("===> Freeze pets")

        if self.current_epoch == self.cfg.num_freeze_epochs:
            unfreeze(self.pets)
            logging.info("===> Unfreeze pets")

    def train_step(
        self, batch, batch_idx
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_old_classes = self.datamodule.num_old_classes
        output = self(batch[0])
        self.train_acc.update(output, batch[1])
        output = output[:, num_old_classes:]
        target = batch[1] - num_old_classes
        loss = self.loss_fn(output, target)
        loss_dict = {"loss": loss, "acc": self.train_acc.compute() * 100}
        return loss, loss_dict

    def post_train_step(self): # EMA 방식으로 PET를 업데이트 -> Accumulation of multi-task knowledge
        for idx, ema in enumerate(reversed(self.pets_emas)):
            if idx == 0:  # the last one
                ema.update(self.pets)
            else:
                ema.update(self.pets_emas[idx - 1])

    def eval_epoch(self, loader):
        task_ranges = []
        n_tasks = self.current_task + 1
        for t in range(n_tasks):
            s = task_ranges[-1][-1] + 1 if task_ranges else 0
            e = s + self.datamodule.num_classes_of(t)
            task_ranges.append(list(range(s, e)))

        pred_on, pred_off, pred_ens, gts = [], [], [], []
        for _, batch in enumerate(loader): # batch[0] = images, [1] = labels, [2] = masking
            input, target = batch[:2]
            output, bs = self(input), input.shape[0]
            pred_on.append(output.argmax(dim=1))

            output_emas = [output.softmax(dim=1)] # 맨 처음 output은 pred_online
            for ema in self.pets_emas: # 순서대로 PET를 attach 후, 각각 inference, offline
                self.attach_pets(ema.module)
                output_emas.append(self(input).softmax(dim=1))
            for oe in output_emas[1:]: # Task 1 이후 생긴다
                pred_off.append(oe.argmax(dim=1))
                break

            if self.cfg.eval_only_emas and len(output_emas) > 1:
                output_emas = output_emas[1:]
            self.attach_pets(self.pets)

            output = torch.stack(output_emas, dim=-1).max(dim=-1)[0]
            # [B, Class], pred_on과 pred_off를 모두 포함해서 큰 Output probability를 갖는 Prediction을 선택

            self.val_acc.update(output, target) # Global Accuracy
            # import pdb; pdb.set_trace()
            for t in batch[2].long().unique().tolist():
                sel = batch[2] == t
                self.val_task_accs[t].update(output[sel], target[sel]) # masking
                t_range = task_ranges[t]
                output_local = output[sel][:, t_range]
                target_local = target[sel] - t_range[0]
                self.val_task_local_accs[t].update(output_local, target_local) # Task 당 정확도

            pred_ens.append(output.argmax(dim=1))
            gts.append(target)

        return pred_on, pred_off, pred_ens, gts

    def post_eval_epoch(self, result):
        super().post_eval_epoch()
        if self.current_task == 0 or not flags.FLAGS.debug:
            return

        pred_on, pred_off, pred_ens, gts = result
        pred_on = torch.cat(pred_on)
        pred_off = torch.cat(pred_off) if len(pred_off) else pred_on[:0]
        pred_ens = torch.cat(pred_ens)
        gts = torch.cat(gts)
        # torch.save(pred_on, f=f"{self.trainer.ckpt_dir}/pred_on.pt")
        # torch.save(pred_off, f=f"{self.trainer.ckpt_dir}/pred_off.pt")
        # torch.save(pred_ens, f=f"{self.trainer.ckpt_dir}/pred_ens.pt")
        # torch.save(gts, f=f"{self.trainer.ckpt_dir}/gts.pt")

    def configure_optimizer(self, *_):
        return super().configure_optimizer(self.model.head, self.pets)


if __name__ == "__main__":
    kwargs = dict(
        datamodule_cls=MyDataModule, module_cls=MyModule, config_cls=MyConf
    )
    App(**kwargs).run()
