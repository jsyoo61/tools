# %%
import torch
import torch.nn as nn

# %%
def aggregation(model_source, model_target, weight = None):
    '''
    model_source: List of nn.Model instances
    model_target: Single nn.Model instance
    weights: default = None
        List of numbers. Must match the number of model_source.
        If None, then weight is set as 1/len(model_source)
    '''
    if weight == None:
        weight = [1/len(model_source)] * len(model_source)
    assert len(model_source) == len(weight), "length of model_source(%s) and weight(%s) does not match"%(len(model_source), len(weight))

    for parameters in zip(model_target.parameters(), *[model.parameters() for model in model_source]):
        p_trg = parameters[0]
        p_src_tuple = parameters[1:]
        # model_target's device
        device = p_trg.device

        # 1. Reset
        p_trg.data[:] = 0

        # 2. Add weighted sum
        for p_src, w in zip(p_src_tuple, weight):
            p_trg.data += (w * p_src.data).to(device)

def distribution(model_source, model_target):
    '''
    model_source: Single nn.Model instance
    model_target: List of nn.Model instances
    '''
    for parameters in zip(model_source.parameters(), *[model.parameters() for model in model_target]):
        p_src = parameters[0]
        p_trg_tuple = parameters[1:]

        for p_trg in p_trg_tuple:
            device = p_trg.device
            p_trg.data[:] = p_src.data.to(device)

def aggregate_state_dict(model_list, device):
    '''
    aggregate model values which are not parameters(weight&bias) but are updated
    '''
    state_dict = {}
    state_dict_list = []
    for model in model_list:
        print(next(model.parameters()).device)
        for key, value in model.state_dict().items():
            if 'weight' in key or 'bias' in key:
                continue
            if key not in state_dict:
                state_dict[key] = value.clone().to(device)
            else:
                state_dict[key] += value.to(device)

    n_model = len(model_list)
    for key in state_dict:
        state_dict[key] /= n_model

    return state_dict

def attentive_aggregation(model_source_list, model_target, step_size = 0.01, p = 2, optimizer = None):
    '''
    perform attentive aggregation from source -> target.
    model_source_list: List of nn.Model instances
    model_target: Single nn.Model instance
    '''
    for parameters in zip(model_target.parameters(), *[model.parameters() for model in model_source]):
        p_trg = parameters[0]
        p_src_tuple = parameters[1:]
        # model_target's device
        device = p_trg.device

        # 1. s = | w_server - w_cleint |
        delta = []
        for p_src in p_src_tuple:
            delta.append(p_trg - p_src.to(device))
        delta = torch.as_tensor(delta).to(device)
        assert len(delta) == len(p_src_tuple)

        s_k = torch.norm(delta, p=p, dim = tuple(range(1, len(p_trg.shape)+1 ) ) ) # dim = (1,2) or (1,2,3) or (1,2,3,...)
        assert len(s_k) == len(p_src_tuple)

        # 2. attention = softmax(s, dim = client)
        attention = torch.softmax(s_k, dim = 0)

        # 3. store gradient
        gradient = torch.sum(attention.expand(delta.shape[::-1]).T * delta, dim = 0)
        p_trg.grad[:] = gradient

        # 4. apply gradient
        p_trg.data -= step_size * p_trg.grad
        # (possible to use optimizer here?)


        # # 1. s = | w_server - w_cleint |
        # s_k = []
        # for p_src in p_src_tuple:
        #     s_k.append(torch.norm(p_trg - p_src.to(device), p=p)) # matrix norm of the difference of layer weights
        # s_k = torch.as_tensor(s_k).to(device)
        # assert len(s_k) == len(p_src_tuple)
        #
        # # 2. attention = softmax(s, dim = client)
        # attention = torch.softmax(s_k, dim = 0)
        #
        # # 3. store gradient
        # p_trg
        # gradient = []
        # for a, p_src in zip(attention, p_src_tuple):
        #     gradient.append(a * (p_trg - p_src.to(device) )


        # 4. apply gradient (either by adding or )


# %%
# class m(nn.Module):
#     def __init__(self):
#         super(m, self).__init__()
#         self.x = nn.Linear(10,10)
# # %%
# model_source = [m().cuda() for i in range(5)]
# model_target = m()
# for i, model in enumerate(model_source):
#     model.x.weight.data[:] = i
# # %%
# for model in model_source:
#     print(model.x.weight)
# print(model_target.x.weight.data)
# # %%
# aggregation(model_source, model_target)
# # %%
# print(model_target.x.weight.data)
# # %%
# distribution(model_target, model_source)
# # %%
#
# y.x.weight.data
# weight = None
# for i in zip([t.parameters() for t in model_source]):
#     print(i)
# torch.cuda.device_count()
