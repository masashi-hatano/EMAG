import torch
from timm.scheduler import CosineLRScheduler


class Warmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        init_lr_ratio: float = 0.0,
        num_epochs: int = 5,
        last_epoch: int = -1,
        iters_per_epoch: int = None,
    ):
        self.base_scheduler = scheduler
        self.warmup_iters = max(num_epochs * iters_per_epoch, 1)
        if self.warmup_iters > 1:
            self.init_lr_ratio = init_lr_ratio
        else:
            self.init_lr_ratio = 1.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        assert self.last_epoch < self.warmup_iters
        return [el * (self.init_lr_ratio + (1 - self.init_lr_ratio) * (float(self.last_epoch) / self.warmup_iters)) for el in self.base_lrs]

    def step(self, *args, **kwargs):
        if self.last_epoch < (self.warmup_iters - 1):
            super().step(*args, **kwargs)
        else:
            self.base_scheduler.step(*args, **kwargs)


def get_optimizer(
    backbone_rgb,
    backbone_flow,
    model,
    optimizer,
    scheduler,
    lr,
    weight_decay,
    momentum,
    max_epochs,
    warmup_epochs,
    warmup_lr_init,
    iters_per_epoch
):
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for m in backbone_rgb.modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if is_bn:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)
    for m in backbone_flow.modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if is_bn:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)
    vae_params = [
        p
        for p_name, p in model.named_parameters()
        if ("vae" in p_name or "head" in p_name) and p.requires_grad
    ]
    other_params = [
        p
        for p_name, p in model.named_parameters()
        if ("vae" not in p_name and "head" not in p_name) and p.requires_grad
    ]

    print(f"Using {optimizer} optimizer!")
    optim_params = [
        {"params": bn_params, "weight_decay": 0.0},
        {"params": non_bn_parameters, "weight_decay": 1e-3},
        {"params": vae_params, "weight_decay": 1e-3},
        {"params": other_params, "weight_decay": 1e-3}
    ]
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=0.0,
            nesterov=True,
        )
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=lr,
            weight_decay=weight_decay
        )

    scheduler = CosineLRScheduler(
        optimizer, t_initial=max_epochs * iters_per_epoch, lr_min=0,
        warmup_t=warmup_epochs * iters_per_epoch, warmup_lr_init=warmup_lr_init, warmup_prefix=True)

    scheduler = {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1
    }

    return optimizer, scheduler
