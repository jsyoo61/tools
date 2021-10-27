
import matplotlib.pyplot as plt

# %%
def imshow(tensor, **kwargs):
    '''
    tensor of (Channel, width, height)
    '''
    tensor=tensor.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(tensor.transpose(1,2,0))

    return fig, ax
