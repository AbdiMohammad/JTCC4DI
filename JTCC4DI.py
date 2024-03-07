#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import torchvision.datasets
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pathlib
import math
import torch.distributions
import sklearn
import sklearn.cluster
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Callable
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# from torchvision.models import resnet18, resnet152
# import torch_pruning
from torch.autograd import Variable
from IPython.display import display
from ipywidgets import interactive
import json
import sys
from codebook_output import CodebookOutput
import cifar_resnet
from imagenet_resnet import imagenet_resnet18, imagenet_resnet34, imagenet_resnet50, imagenet_resnet101, imagenet_resnet152

import codebook, drop_layer
from codebook import Codebook
from utils import op_counter
from utils import ptflops
import torch_pruning as tp
import copy

def LinearCoefficient(start_beta, end_beta):
    def res(epoch, n_epochs):
        return start_beta + (end_beta - start_beta) * epoch / n_epochs
    return res

def ConstantCoefficient(beta):
    def res(epoch, n_epochs):
        return beta
    return res

def categorical_entropy(dist):
    return torch.distributions.Categorical(dist).entropy().mean(dim=-1)

# @dataclass
# class CodebookOutput:
#     original_tensor: torch.Tensor
#     codebook_outputs: list

class ChannelPruner(nn.Module):

    def __init__(self, n_channels):
        super().__init__()

        conv_weights = torch.eye(n_channels).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('conv_weights', conv_weights)
        self.pruned_indices = []
        self.n_channels = n_channels
    
    def set_pruned_indices(self, indices):

        # indices = torch.LongTensor(indices).to(self.conv_weights.device)
        self.pruned_indices = indices
        conv_weights = torch.eye(self.n_channels).unsqueeze(-1).unsqueeze(-1).to(self.conv_weights.device)
        conv_weights[indices, :, :, :] = 0
        self.conv_weights.set_(conv_weights)
    
    def forward(self, x):
        if type(x) == CodebookOutput:
            original_res = F.conv2d(x.original_tensor, self.conv_weights)
            codebook_outputs = []
            for codebook_x, dist, codebook in x.codebook_outputs:
                codebook_outputs.append([F.conv2d(codebook_x, self.conv_weights), dist, codebook])
            return CodebookOutput(original_res, codebook_outputs)
        else:
            return F.conv2d(x, self.conv_weights)

class StopCompute(nn.Module):

    def __init__(self, inner):
        super().__init__()

        self.inner = inner
    
    def forward(self, x):
        res = self.inner(x)
        raise Exception(res)


# def load_pretrained_resnet():
#     model_path = pathlib.Path(__file__).parent / 'resnet20-12fca82f.th'
#     model_dict = torch.load(model_path)
#     device = torch.device('cuda')
#     rn = cifar_resnet20()
#     rn = rn.to(device)
#     def remove_module_prefix(d):
#         res = dict()
#         for key in d.keys():
#             res[key[len('module.'):]] = d[key]
#         return res
#     rn.load_state_dict(remove_module_prefix(model_dict['state_dict']))
#     return rn

def get_physically_pruned_model(model, channel_prune_args):
    physically_pruned_model = copy.deepcopy(model).to(model_device(model))
    prune_dict = dict()
    for channel_pruning_data in channel_prune_args:
        try:
            new_layer_name = get_layer_name_after_pruning(channel_pruning_data.layer_name, True)
            layer = eval('physically_pruned_model.' + new_layer_name)
        except:
            new_layer_name = get_layer_name_after_pruning(channel_pruning_data.layer_name, False)
            layer = eval('physically_pruned_model.' + new_layer_name)
        exec(f'physically_pruned_model.{new_layer_name} = layer[0]')
        prune_dict[eval(f'physically_pruned_model.{new_layer_name}')] = get_pruner_submodule(layer).pruned_indices
    DG = tp.DependencyGraph()
    DG.register_customized_layer(
        codebook.Codebook, 
        codebook.CodebookPruner())
    DG.register_customized_layer(
        drop_layer.DropLayer, 
        drop_layer.DropLayerPruner())
    DG.build_dependency(physically_pruned_model, example_inputs=next(iter(valid_dl))[0].to(model_device(physically_pruned_model)))
    for target_module, indices in prune_dict.items():
        pruning_group = DG.get_pruning_group(target_module, tp.prune_conv_out_channels, idxs=indices)
        if DG.check_pruning_group(pruning_group):
            pruning_group.prune()
    
    return physically_pruned_model
    
def cifar_resnet_loader_generator(model_name, pretrained_weights_path=None):
    def load_pretrained_resnet():
        device = torch.device('cuda')
        # rn = cifar_resnet20()
        rn = eval(f'cifar_resnet.cifar_{model_name}')()
        rn = rn.to(device)
        def remove_module_prefix(d):
            res = dict()
            for key in d.keys():
                res[key[len('module.'):]] = d[key]
            return res
        if pretrained_weights_path is not None:
            model_dict = torch.load(pretrained_weights_path)
            rn.load_state_dict(remove_module_prefix(model_dict['state_dict']))
        return rn
    return load_pretrained_resnet

def imagenet_resnet_loader_generator(model_name):
    def load_pretrained_imagenet():
        device = torch.device('cuda')
        rn = eval('imagenet_' + model_name)()
        print("loaded model modules size:", len(list(rn.modules())))
        rn = rn.to(device)
        return rn
    return load_pretrained_imagenet

def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    n_all = 0
    n_correct = 0
    for xs, labels in tqdm(dataloader):

        xs = xs.to(device)
        labels = labels.to(device)
        out = model(xs)
        n_correct += (out.argmax(dim=-1) == labels).sum().item()
        n_all += len(xs)
    return n_correct / n_all

def evaluate_codebook_model(model, dataloader, codebook_index=-1):
    device = next(model.parameters()).device
    n_all = 0
    n_correct = 0
    for xs, labels in tqdm(dataloader):

        xs = xs.to(device)
        labels = labels.to(device)
        out = model(xs)
        if type(out) == CodebookOutput:
            if codebook_index == -1:
                out = out.original_tensor
            else:
                out = out.codebook_outputs[codebook_index][0]
        n_correct += (out.argmax(dim=-1) == labels).sum().item()
        n_all += len(xs)
    return n_correct / n_all

def model_device(model):
    return next(model.parameters()).device

def resnet_vib_loss(model_output: CodebookOutput, labels, norm_regularizers, epoch, n_epochs):
    if type(model_output) != CodebookOutput:
        dummy = CodebookOutput(model_output, [])
        return resnet_vib_loss(dummy, labels)

    original_model_outputs = model_output.original_tensor
    metrics = dict()
    original_model_loss = F.cross_entropy(original_model_outputs, labels)
    original_model_acc = (original_model_outputs.argmax(dim=-1) == labels).float().mean().item()
    loss = original_model_loss
    metrics["original acc"] = original_model_acc

    # FIXME: Remove norm regulizers for more stability
    for norm_regularizer in norm_regularizers:
        loss += norm_regularizer[0](epoch, n_epochs) * norm_regularizer[1]

    for codebook_output, dist, codebook in model_output.codebook_outputs:
        distortion_loss = F.cross_entropy(codebook_output, labels)
        codebook_acc = (codebook_output.argmax(dim=-1) == labels).float().mean().item()
        codebook_entropy = categorical_entropy(dist).mean().item()
        codebook_loss = distortion_loss + codebook.beta(epoch, n_epochs) * codebook_entropy
        loss += codebook_loss
        # metrics.append(codebook_acc)
        metrics["codebook at " + codebook.train_data.layer_name + " acc"] = codebook_acc
    return loss, metrics

def get_pruner_submodule(module):
    if type(module) == ChannelPruner:
        return module

    for submodule in module.modules():
        if submodule == module:
            continue
        submodule_result = get_pruner_submodule(submodule)
        if not (submodule_result is None):
            return submodule_result
        
    return None

def get_layer_name_after_pruning(old_layer_name, has_codebook=True):
    layer_name_parts = old_layer_name.split('.')
    if has_codebook:
        new_layer_name = '.'.join(layer_name_parts[:-2] + [layer_name_parts[-2] + '[0]'] + [layer_name_parts[-1]])
    else:
        new_layer_name = '.'.join(layer_name_parts[:-2] + [layer_name_parts[-2]] + [layer_name_parts[-1]])
    return new_layer_name

def module_name2class_name(module_name):
    module_name_parts = module_name.split('.')
    if len(module_name_parts) > 2:
        return f'{module_name_parts[-3]}[{module_name_parts[-2]}].{module_name_parts[-1]}'
    else:
        return None

def train_model(n_epochs, model, loss_fn, train_dataloader, valid_dataloader, optimizer=None, codebook_training_datas=[], channel_prune_args=[]):
    device = model_device(model)
    train_losses = []
    train_metrics = defaultdict(list)
    valid_losses = []
    valid_metrics = defaultdict(list)

    writer = SummaryWriter()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        for channel_prune_data in channel_prune_args:
            try:
                prune_index = channel_prune_data.prune_epochs.index(epoch)
            except ValueError:
                prune_index = -1

            if prune_index != -1:
                channel_indices_to_prune = get_channels_to_prune(get_weight_norms(model, channel_prune_data.layer_name).detach().cpu().numpy(), channel_prune_data.prune_values[prune_index], channel_prune_data.prune_values_type)
                
                print('_' * 80)
                print('Pruning channels before {} at epoch {}, removing {} channels ({})'.format(channel_prune_data.layer_name, epoch,\
                                                                                                 len(channel_indices_to_prune), channel_prune_data.prune_values_type))
                # print('Valid accuracy before pruning: ', evaluate_codebook_model(model, valid_dataloader, -1))

                try:
                    new_layer_name = get_layer_name_after_pruning(channel_prune_data.layer_name, True)
                    layer = eval('model.' + new_layer_name)
                except:
                    new_layer_name = get_layer_name_after_pruning(channel_prune_data.layer_name, False)
                    layer = eval('model.' + new_layer_name)

                pruner_module = get_pruner_submodule(layer)
                pruner_module.set_pruned_indices(channel_indices_to_prune)
                # eval('model.' + codebook_training_data.layer_name)[1].set_pruned_indices(codebook_training_data.prune_channels_ns[prune_index])
                # prune_and_replace_codebook(model, train_dataloader, i, codebook_training_data.layer_name, codebook_training_data.prune_values[prune_index])
                # print('Valid accuracy after pruning: ', evaluate_codebook_model(model, valid_dataloader, -1))
                print('_' * 80)
        for i, codebook_training_data in enumerate(codebook_training_datas):
            try:
                prune_index = codebook_training_data.prune_epochs.index(epoch)
            except ValueError:
                prune_index = -1

            if prune_index != -1:
                print('_' * 80)
                print('Pruning codebook {} at epoch {}, removing {} codewords ({})'.format(codebook_training_data.layer_name, epoch,\
                                                                                          codebook_training_data.prune_values[prune_index], codebook_training_data.prune_values_type))
                # print('Valid accuracy before pruning: ', evaluate_codebook_model(model, valid_dataloader, i))
                prune_and_replace_codebook(model, train_dataloader, i, codebook_training_data.layer_name,\
                                           codebook_training_data.prune_values[prune_index], codebook_training_data.prune_values_type)
                # print('Valid accuracy after pruning: ', evaluate_codebook_model(model, valid_dataloader, i))
                print('_' * 80)

        for xs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch}/{n_epochs}'):

            xs = xs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model_output = model(xs)
            # loss = F.cross_entropy(pred, labels)
            norm_regularizers = []
            for channel_prune_data in channel_prune_args:
                weight_norms = get_weight_norms(model, channel_prune_data.layer_name)
                norm_regularizers.append([channel_prune_data.gamma, weight_norms.mean()])
            loss, metrics = loss_fn(model_output, labels, norm_regularizers, epoch, n_epochs)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar('train loss', loss.item(), epoch)
            for metric_name, metric in metrics.items():
                train_metrics[metric_name + '/train'].append(metric)
                writer.add_scalar(metric_name + '/train', metric, epoch)


        with torch.no_grad():
            for xs, labels in tqdm(valid_dataloader, desc=f'Epoch {epoch}/{n_epochs}'):
                xs = xs.to(device)
                labels = labels.to(device)
                model_output = model(xs)
                norm_regularizers = []
                for channel_prune_data in channel_prune_args:
                    weight_norms = get_weight_norms(model, channel_prune_data.layer_name)
                    norm_regularizers.append([channel_prune_data.gamma, weight_norms.mean()])
                loss, metrics = loss_fn(model_output, labels, norm_regularizers, epoch, n_epochs)
                valid_losses.append(loss.item())
                writer.add_scalar('valid loss', loss.item(), epoch)
                for metric_name, metric in metrics.items():
                    valid_metrics[metric_name + '/valid'].append(metric)
                    writer.add_scalar(metric_name + '/valid', metric, epoch)

        # plt.subplot(321, title='train original acc')
        # plt.plot(train_metrics[0])
        # plt.subplot(322, title='train codebook acc')
        # plt.plot(train_metrics[1])
        # plt.subplot(323, title='train loss')
        # plt.plot(train_losses)
        # plt.subplot(324, title='valid original acc')
        # plt.plot(valid_metrics[0])
        # plt.subplot(325, title='valid codebook acc')
        # plt.plot(valid_metrics[1])
        # plt.subplot(326, title='valid loss')
        # plt.plot(valid_losses)
        # plt.show()
        
    writer.close()
    return train_metrics, train_losses, valid_metrics, valid_losses

@dataclass
class ChannelPruneData:
    layer_name: str
    gamma: Callable[[int, int], float]
    prune_epochs: List[int]
    prune_values: List[float]
    prune_values_type: str

@dataclass
class CodebookTrainData:
    layer_name: str
    hidden_dim: int
    codebook_size: int
    beta: Callable[[int, int], float]
    prune_epochs: List[int]
    prune_values: List[float]
    prune_values_type: str
    

def create_resnet_with_pruner_and_codebook(loader_fn, dataloader, PSNR, codebooks: List[CodebookTrainData], channels_to_prune: List[ChannelPruneData] = []):
    unmodified_resnet = loader_fn()
    pretrained_resnet = loader_fn()

    for channel_prune_data in channels_to_prune:
        target_module = eval('pretrained_resnet.' + channel_prune_data.layer_name)
        n_channels = get_layer_output_shape(unmodified_resnet, dataloader, channel_prune_data.layer_name)[1]
        new_module = nn.Sequential(
            target_module,
            ChannelPruner(n_channels).to(model_device(pretrained_resnet))
        )
        exec('pretrained_resnet.' + channel_prune_data.layer_name + '= new_module')

    for codebook in codebooks:
        target_module = eval('pretrained_resnet.' + codebook.layer_name)

        n_channels = get_layer_output_shape(unmodified_resnet, dataloader, codebook.layer_name)[1]
        
        new_module = nn.Sequential(
            target_module,
            #FIXME: Determine input dimension of codebook either based on config file or based on last layer's number of channels
            Codebook(n_channels, codebook.codebook_size, codebook.beta, codebook, PSNR=PSNR).to(model_device(pretrained_resnet))
        )
        exec('pretrained_resnet.' + codebook.layer_name + '= new_module')
        weights = get_initial_weights(unmodified_resnet, dataloader, codebook.layer_name, codebook.hidden_dim, codebook.codebook_size)
        exec('pretrained_resnet.' + codebook.layer_name + '[-1].embedding.data = torch.Tensor(weights).to(model_device(pretrained_resnet))')
    return pretrained_resnet

def get_initial_weights(model, dataloader, layer, codebook_dim, codebook_size):
    embeddings_list = []
    exec('model.' + layer + '= StopCompute(model.' + layer + ')')

    for xs, labels in tqdm(dataloader):
        xs = xs.to(model_device(model))
        labels = labels.to(model_device(model))
        try:
            _ = model(xs)
        except Exception as e:
            embeddings_list.append(e.args[0].detach())

    embeddings = torch.cat(embeddings_list).cpu().numpy().reshape(-1, codebook_dim)
    k_means = sklearn.cluster.MiniBatchKMeans(n_clusters=codebook_size, n_init='auto')
    k_means.fit(embeddings)

    exec('model.' + layer + '= model.' + layer + '.inner')
    return k_means.cluster_centers_

def get_layer_output_shape(model, dataloader, layer):
    exec('model.' + layer + '= StopCompute(model.' + layer + ')')

    try:
        _ = model(next(iter(dataloader))[0].to(model_device(model)))
    except Exception as e:
        if type(e.args[0]) == CodebookOutput:
            return e.args[0].original_tensor.shape
        else:
            return e.args[0].detach().shape
    finally:
        exec('model.' + layer + '= model.' + layer + '.inner')

def get_codebook_params_and_ids(model):

    ids = []
    params = []

    for module in model.modules():
        if type(module) == Codebook:
            for param in module.parameters():
                ids.append(id(param))
                params.append(param)
    return params, ids

def get_codebook_usage_data(model, dataloader, codebook_index):
    with torch.no_grad():
        sample_output = model(next(iter(dataloader))[0].to(model_device(model)))
        codebook_dim = sample_output.codebook_outputs[codebook_index][1].shape[1]
        counts = torch.zeros(codebook_dim, dtype=torch.int64).to(model_device(model))

        for xs, _ in tqdm(dataloader):
            xs = xs.to(model_device(model))
            output = model(xs)
            dist = output.codebook_outputs[codebook_index][1]
            batch_indices, batch_counts = dist.argmax(dim=1).unique(return_counts=True)
            counts[batch_indices] += batch_counts
        return counts
    

def prune_codebook(model, dataloader, codebook_index, value_to_prune, value_type):
    sample_output = model(next(iter(dataloader))[0].to(model_device(model)))
    codebook = sample_output.codebook_outputs[codebook_index][2]

    counts = get_codebook_usage_data(model, dataloader, codebook_index)
    if value_type == 'number':
        unpruned_indices = counts.sort()[1][value_to_prune:]
    elif value_type == 'percentage':
        unpruned_indices = counts.sort()[1][round(value_to_prune/100.0*len(counts)):]
        # unpruned_indices = (counts > (value_to_prune / 100.0 * counts.max())).nonzero(as_tuple=True)[0]

    new_codebook = Codebook(codebook.latent_dim, codebook.n_embeddings - (len(counts) - len(unpruned_indices)), codebook.beta, codebook.train_data).to(model_device(model))
    new_codebook.embedding.data = codebook.embedding.data[unpruned_indices]
    print(f'{len(counts) - len(unpruned_indices)} codewords removed')
    # codebook = model.layer1[0][1]
    # codebook.embedding.data[indices_to_prune] = 0
    return new_codebook

def prune_and_replace_codebook(model, dataloader, index, layer_name, value_to_prune, value_type='number'):
    codebook_layer_name = layer_name + '[-1]'
    exec('model.' + codebook_layer_name + '= prune_codebook(model, dataloader, index, value_to_prune, value_type)')

def get_channels_to_prune(weight_norms, value_to_prune, value_type='number'):
    if value_type == 'number':
        indices_to_prune = weight_norms.argsort()[:value_to_prune]
    elif value_type == 'percentage':
        indices_to_prune = weight_norms.argsort()[:round(value_to_prune/100.0*len(weight_norms))]
        # indices_to_prune = (weight_norms < (value_to_prune / 100.0 * weight_norms.max())).nonzero()[0]
    return list(indices_to_prune)

def get_weight_norms(model, layer_name):
    try:
        weights = eval('model.' + layer_name).weight
    except:
        try:
            layer_parts = layer_name.split('.')
            layer_parts = layer_parts[:-2] + [layer_parts[-2] + '[0]'] + [layer_parts[-1] + '[0]']
            new_layer_name = '.'.join(layer_parts)
            weights = eval('model.' + new_layer_name).weight
        except:
            layer_parts = layer_name.split('.')
            layer_parts = layer_parts[:-2] + [layer_parts[-2]] + [layer_parts[-1] + '[0]']
            new_layer_name = '.'.join(layer_parts)
            weights = eval('model.' + new_layer_name).weight
        # weight = eval('model.' + layer_name + '[0]')
    return weights.flatten(1).norm(dim=1)

#%%
# pretrained_resnet = load_pretrained_resnet()

if __name__ == '__main__':

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    dataset = json_data["dataset"]
    BATCH_SIZE = json_data["batch_size"]
    model_name = json_data["model"]

    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10': 
        cifar10_train = torchvision.datasets.CIFAR10('dataset/cifar10', train=True, download=True, transform=transform)
        cifar10_test = torchvision.datasets.CIFAR10('dataset/cifar10', train=False, download=True, transform=transform)
        cifar10_train_dataloader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, shuffle=True)
        cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=BATCH_SIZE)

        cifar10_train_small = torch.utils.data.Subset(cifar10_train, range(0, 1024 * 4))
        cifar10_train_small_dataloader = DataLoader(cifar10_train_small, batch_size=BATCH_SIZE, shuffle=True)
        train_dl = cifar10_train_dataloader
        valid_dl = cifar10_test_dataloader

    if dataset == 'imagenet':
        # todo: load actual imagenet
        imagenet_train = TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
        imagenet_test = TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
        imagenet_train_dataloader = DataLoader(imagenet_train, batch_size=BATCH_SIZE, shuffle=True) 
        imagenet_test_dataloader = DataLoader(imagenet_test, batch_size=BATCH_SIZE)

        imagenet_train_small = torch.utils.data.Subset(imagenet_train, range(0, 1024))
        imagenet_train_small_dataloader = DataLoader(imagenet_train_small, batch_size=BATCH_SIZE, shuffle=True)
        train_dl = imagenet_train_dataloader
        valid_dl = imagenet_test_dataloader


    
    PSNR = json_data['PSNR']
    pretrained_weights_path = json_data['pretrained_weights_path']
    save_weights_after_train_path = json_data['save_weights_after_train_path']
    n_epochs = json_data['n_epochs']
    codebook_lr = json_data['codebook_lr']
    non_codebook_lr = json_data['non_codebook_lr']

    output_folder = json_data['output_folder']
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    reference_pretrained_weights_path = json_data["reference_pretrained_weights_path"]

    codebook_training_data = []
    channel_pruning_data = []

    if 'funnel' in json_data.keys():
        funnel = json_data['funnel']
        reference_model = cifar_resnet_loader_generator(model_name)()
        head_conv_modules = []
        for name, module in reference_model.named_modules():
            if isinstance(module, nn.Conv2d):
                name = module_name2class_name(name)
                if name is None:
                    continue
                head_conv_modules.append((name, module))
                if name == funnel['layer']:
                    break
        for conv_index, (name, module) in enumerate(head_conv_modules):
            channel_pruning_data.append(
                ChannelPruneData(
                    name,
                    gamma=LinearCoefficient(funnel['gamma'] / 10.0 * conv_index / (len(head_conv_modules) - 1), funnel['gamma'] * conv_index / (len(head_conv_modules) - 1)),
                    prune_epochs=funnel['prune_epochs'],
                    prune_values=[(i + 1) * funnel['prune_value'] / len(funnel['prune_epochs']) * conv_index / (len(head_conv_modules) - 1) for i, _ in enumerate(funnel['prune_epochs'])],
                    prune_values_type=funnel['prune_value_type']
                )
            )
        del reference_model
    elif 'prune_channels' in json_data.keys():
        for channel_pruning_data_json in json_data['prune_channels']:
            channel_pruning_data.append(
                ChannelPruneData(
                    channel_pruning_data_json['layer'],
                    gamma=eval(channel_pruning_data_json['gamma']),
                    prune_epochs=channel_pruning_data_json['prune_epochs'],
                    prune_values=[(i + 1) * channel_pruning_data_json['prune_value'] / len(channel_pruning_data_json['prune_epochs']) for i, _ in enumerate(channel_pruning_data_json['prune_epochs'])],
                    prune_values_type=channel_pruning_data_json['prune_value_type']
                )
            )

    for codebook_training_data_json in json_data['codebooks']:
        codebook_training_data.append(
            CodebookTrainData(
                codebook_training_data_json['layer'],
                hidden_dim=codebook_training_data_json['hidden_dim'],
                codebook_size=codebook_training_data_json['codebook_size'],
                beta=eval(codebook_training_data_json['beta']),
                prune_epochs=codebook_training_data_json['prune_epochs'],
                prune_values=[codebook_training_data_json['prune_value'] / len(codebook_training_data_json['prune_epochs'])] * len(codebook_training_data_json['prune_epochs']),
                prune_values_type=codebook_training_data_json['prune_value_type']
            )
        )

    if dataset == 'cifar10':
        resnet_with_codebook = create_resnet_with_pruner_and_codebook(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path), train_dl, PSNR, codebook_training_data, channel_pruning_data)
    else:
        resnet_with_codebook = create_resnet_with_pruner_and_codebook(imagenet_resnet_loader_generator(model_name), imagenet_train_small_dataloader, PSNR, codebook_training_data, channel_pruning_data)

    if not pretrained_weights_path is None:
        resnet_with_codebook.load_state_dict(torch.load(pretrained_weights_path))

    all_params = list(resnet_with_codebook.parameters())

    codebook_params, codebook_ids = get_codebook_params_and_ids(resnet_with_codebook)
    non_codebook_params = [p for p in all_params if id(p) not in codebook_ids]

    optimizer = torch.optim.Adam([
        {'params': non_codebook_params, 'lr': non_codebook_lr},
        {'params': codebook_params, 'lr': codebook_lr},
    ])
    
    train_metrics, train_losses, valid_metrics, valid_losses = train_model(n_epochs,
                                                                            resnet_with_codebook,
                                                                            resnet_vib_loss,
                                                                            train_dl,
                                                                            valid_dl,
                                                                            optimizer,
                                                                            codebook_training_data,
                                                                            channel_pruning_data)

    tail_modules = [tail_module_name for tail_module_name, _ in resnet_with_codebook.named_modules()\
            if 'layer2' in tail_module_name or\
            'layer3' in tail_module_name or\
            'linear' in tail_module_name
            ]
    def return_example_input(input_res=(3, 32, 32)):
        return next(iter(train_dl))[0].to('cuda')
    def count_ops_and_params(model, ignore_list=[]):
        # return op_counter.count_ops_and_params(model, example_inputs=return_example_input(), ignore_list=ignore_list)
        return ptflops.get_model_complexity_info(model, input_res=tuple(next(iter(train_dl))[0].shape)[1:], print_per_layer_stat=False, as_strings=False, input_constructor=return_example_input, verbose=False, ignore_list=ignore_list)

    flops_total, params_total = count_ops_and_params(resnet_with_codebook)
    flops_head, params_head = count_ops_and_params(resnet_with_codebook, ignore_list=tail_modules)
    val_ori_acc, val_codebook_acc = evaluate_codebook_model(resnet_with_codebook, valid_dl, -1), evaluate_codebook_model(resnet_with_codebook, valid_dl, 0)

    physically_pruned_model = get_physically_pruned_model(resnet_with_codebook, channel_prune_args=channel_pruning_data)
    pruned_flops_total, pruned_params_total = count_ops_and_params(physically_pruned_model)
    pruned_flops_head, pruned_params_head = count_ops_and_params(physically_pruned_model, ignore_list=tail_modules)
    pruned_val_ori_acc, pruned_val_codebook_acc = evaluate_codebook_model(physically_pruned_model, valid_dl, -1), evaluate_codebook_model(physically_pruned_model, valid_dl, 0)

    unmodified_model = cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path)()
    unmodified_flops_total, unmodified_params_total = count_ops_and_params(unmodified_model)
    unmodified_flops_head, unmodified_params_head = count_ops_and_params(unmodified_model, ignore_list=tail_modules)
    unmodified_val_ori_acc = evaluate_codebook_model(unmodified_model, valid_dl, -1)

    acceleration_total, acceleration_head = flops_total / pruned_flops_total, flops_head / pruned_flops_head
    compression_total, compression_head = params_total / pruned_params_total, params_head / pruned_params_head

    print(10 * '*' + 'Virtually Pruned' + 10 * '*')
    print(f'Total:\tFLOPs: {flops_total}, Params: {params_total}')
    print(f'Head:\tFLOPs: {flops_head}, Params: {params_head}')
    print(f'Valid Original Accuracy:{val_ori_acc}\tValid Codebook Accuracy:{val_codebook_acc}')

    print(10 * '*' + 'Physically Pruned' + 10 * '*')
    print(f'Total:\tFLOPs: {pruned_flops_total}, Params: {pruned_params_total}')
    print(f'Head:\tFLOPs: {pruned_flops_head}, Params: {pruned_params_head}')
    print(f'Valid Original Accuracy:{pruned_val_ori_acc}\tValid Codebook Accuracy:{pruned_val_codebook_acc}')

    print(f'Acceleration:\tTotal:{acceleration_total * 100.0}%, Head:{acceleration_head * 100.0}%')
    print(f'Compression:\tTotal:{compression_total * 100.0}%, Head:{compression_head * 100.0}%')
    
    print(10 * '*' + 'Unmodified Model' + 10 * '*')
    print(f'Total:\tFLOPs: {unmodified_flops_total}, Params: {unmodified_params_total}')
    print(f'Head:\tFLOPs: {unmodified_flops_head}, Params: {unmodified_params_head}')
    print(f'Valid Original Accuracy:{unmodified_val_ori_acc}')

    training_data = {
        'train_metrics': train_metrics,
        'train_losses': train_losses,
        'valid_metrics': valid_metrics,
        'valid_losses': valid_losses
    }

    with open(output_folder + '/train_metrics.json', 'w') as outfile:
        json.dump(training_data, outfile)

    if save_weights_after_train_path:
        torch.save(resnet_with_codebook.state_dict(), f'{output_folder}/model.pth')
        torch.save(physically_pruned_model.state_dict(), f'{output_folder}/physically_pruned_model.pth')
    
    print(physically_pruned_model)
