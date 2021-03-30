import torch

def get_device(model):
    return next(model.parameters()).device

def spread_device(job_num = None):
    if job_num != None:
        gpu_id = job_num % torch.cuda.device_count()
    else:
        gpu_id = torch.cuda.default_stream().device.index
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return device
