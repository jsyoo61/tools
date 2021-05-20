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

def multiprocessing_device(gpu_id = None):
    '''
    if gpu_id
    '''
    if gpu_id == -1 or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        if gpu_id == None:
            p=multiprocessing.current_process()
            try:
                worker_num = int(p.name.split('-')[-1]) # Fails if it's not integer
                gpu_id = worker_num % torch.cuda.device_count()
                return torch.device(f"cuda:{gpu_id}")
            except ValueError: # Parent process
                return torch.cuda.default_stream().device
