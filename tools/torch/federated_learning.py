# %%
import torch
import torch.nn as nn

# TODO: migrate these functions to hideandseek

# %%
def transfer(model_source, model_target):
    '''
    model_source: Single nn.Model instance
    model_target: Single nn.Model instance

    for multiple source or targets, refer to aggregate() or distribute()
    '''
    for p_trg, p_src in zip(model_target.parameters(), model_source.parameters()):
        # model_target's device
        device = p_trg.device

        # Clone data. Becareful not to copy hard link! (never copy pointers)
        p_trg.data = torch.clone(p_src.data).to(device)

        # # 1. Reset
        # data = torch.zeros_like(p_trg).to(device)
        # # p_trg.data[:] = 0
        #
        # # 2. Add weighted sum
        # data += p_src.data.to(device)
        # p_trg.data += p_src.data.to(device)

        '''# TODO: p_trg.data = torch.clone(p_src.data)'''

def aggregate(model_source, model_target, weight = None):
    '''
    Aggregate parameters of the model

    model_source: List of nn.Model instances
    model_target: Single nn.Model instance
    weights: default = None
        List of numbers. Must match the number of model_source.
        If None, then weight is set as 1/len(model_source)
    '''
    if weight is None:
        weight = [1/len(model_source)] * len(model_source)
    assert len(model_source) == len(weight), "length of model_source(%s) and weight(%s) does not match"%(len(model_source), len(weight))

    for parameters in zip(model_target.parameters(), *[model.parameters() for model in model_source]):
        p_trg = parameters[0]
        p_src_tuple = parameters[1:]
        # model_target's device
        device = p_trg.device

        # 1. Reset
        data = torch.zeros_like(p_trg)
        # p_trg.data[:] = 0

        # 2. Add weighted sum
        for p_src, w in zip(p_src_tuple, weight):
            # p_trg.data += (w * p_src.data).to(device)
            data += (w * p_src.data).to(device)

        p_trg.data = data

def distribute(model_source, model_target):
    '''
    Distribute parameters of the model

    model_source: Single nn.Model instance
    model_target: List of nn.Model instances
    '''
    for parameters in zip(model_source.parameters(), *[model.parameters() for model in model_target]):
        p_src = parameters[0]
        p_trg_tuple = parameters[1:]

        for p_trg in p_trg_tuple:
            device = p_trg.device
            # p_trg.data[:] = p_src.data.to(device)
            p_trg.data = torch.clone(p_src.data).to(device)

def aggregate_grad(model_source, model_target):
    '''
    model_source: List of nn.Model instances
    model_target: Single nn.Model instance
    '''
    for parameters in zip(model_target.parameters(), *[model.parameters() for model in model_source]):
        p_trg = parameters[0]
        p_src_tuple = parameters[1:]
        # model_target's device
        device = p_trg.device

        for p_src in p_src_tuple:
            p_trg.grad += p_src.grad.to(device)

def aggregate_all (model_source, model_target):
    '''
    Aggregate parameters and states of the model.
    States

    model_source: List of nn.Model instances
    model_target: Single nn.Model instance
    '''
    device = next(model.parameters()).device
    state_dict = {}
    state_dict_list = []
    for model in model_source:
        for key, value in model.state_dict().items():
            if 'weight' in key or 'bias' in key:
                continue
            if key not in state_dict:
                state_dict[key] = value.clone().to(device)
            else:
                state_dict[key] += value.to(device)

    n_model = len(model_source)
    for key in state_dict:
        state_dict[key] /= n_model

    model_target.load_state_dict(state_dict)

def distribute_all(model_source, model_target):
    '''
    model_source: Single nn.Model instance
    model_target: List of nn.Model instances
    '''
    for model_target_ in model_target:
        model_target_.load_state_dict(model_source.state_dict())

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

def attentive_aggregate(model_source_list, model_target, step_size = 0.01, p = 2, optimizer = None):
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
if __name__ == '__main__':

    class A(nn.Module):
        def __init__(self, i=1):
            super().__init__()
            self.x = nn.Linear(5,5)
            self.x.weight.data = torch.full_like(self.x.weight, i)
            self.x.bias.data = torch.full_like(self.x.bias, i)

        def forward(self, x):
            return

    # %%
    m1 = A(1)
    m2 = A(2)
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    # Only the data is copied
    m1.x.weight.data[:] = m2.x.weight.data
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    m1.x.weight.data += 1
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    # %%
    # Change together. Linked
    m1.x.weight.data = m2.x.weight.data
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    m1.x.weight.data += 1
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    # Not linked
    m1.x.weight.data = torch.clone(m2.x.weight.data)
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    m1.x.weight.data += 1
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    # %%
    # Not Linked
    transfer(m1,m2)
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    m1.x.weight.data += 1
    print(m1.x.weight)
    print(m2.x.weight)
    id(m1.x.weight)
    id(m2.x.weight)
    id(m1.x.weight.data)
    id(m2.x.weight.data)

    # %%
    # Aggregate
    m1=A(1)
    m2=A(2)
    m3=A(3)
    aggregate([m1,m2], m3)
    print(m1.x.weight)
    print(m2.x.weight)
    print(m3.x.weight)

    # %%
    # distribute
    distribute(m3, [m1,m2])
    print(m1.x.weight)
    print(m2.x.weight)
    print(m3.x.weight)
    m1.x.weight.data += 1
    m2.x.weight.data += 2
    print(m1.x.weight)
    print(m2.x.weight)
    print(m3.x.weight)

# from model import DNN
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# n_input = 5
# n_output = 10
# n_hidden_list = [10,20,30,20]
# n_model = 4
# activation = [nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(0.03), nn.Tanh(), nn.Identity()]
# model_list = [DNN(n_input, n_output, n_hidden_list, activation) for i in range(n_model)]
#
# model_source = model_list[1:]
# model_target = model_list[0]
# model_target.cuda()
#
# x = torch.randn(20,5)
# for model in model_list:
#     device = next(model.parameters()).device
#     y_hat = model(x.to(device))
#     loss = torch.sum(y_hat)
#     loss.backward()
#
# p=list(model_target.parameters())
# p[0].grad
# next(model_source[0].parameters()).grad
#
# g = torch.zeros(10,5)
# for model in model_list:
#     g += next(model.parameters()).grad.cpu()
# print('cumsum of gradient:\n',g)


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
