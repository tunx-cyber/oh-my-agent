import math
import os
import random
import torch
import torch.distributed as dist
import numpy as np
def is_main_process():
    return not dist.is_initialized or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(cur_step, total_step, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * cur_step / total_step)))

def init_distribute_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

