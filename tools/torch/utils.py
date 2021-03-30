
def get_device(model):
    return next(model.parameters()).device
