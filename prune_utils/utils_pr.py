from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import copy
import yaml
import numpy as np

import datetime
import operator
import random

def prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Multi level admm arguments')
    admm_args.add_argument('--sp-load-frozen-weights',
        type=str, help='the weights that are frozen '
        'throughout the pruning process')


def canonical_name(name):
    # if the model is running in parallel, the name may start
    # with "module.", but if hte model is running in a single
    # GPU, it may not, we always filter the name to be the version
    # without "module.",
    # names in the config should not start with "module."
    if "module." in name:
        return name.replace("module.", "")
    else:
        return name


def get_frozen_weights(model, filename,
                       prune_ratios, frozen_ratios,
                       all_params=False):
    assert(filename)
    loaded_weights = torch.load(filename)
    frozen_weights = {}
    # import pdb; pdb.set_trace()
    state_dict = model.state_dict()
    for name in state_dict:
        W = state_dict[name]
        cname = canonical_name(name)
        if (frozen_ratios is not None and name not in frozen_ratios) or \
           (frozen_ratios is None and name not in prune_ratios):
            continue
        if name in loaded_weights:
            frozen_weight = loaded_weights[name]
        elif cname in loaded_weights:
            frozen_weight = loaded_weights[cname]
        else:
            continue
        if frozen_ratios:
            prune_ratio = frozen_ratios[name]
        else:
            prune_ratio = prune_ratios[name]
        frozen_weights[name] = {}
        # check identical
        if (False):
            non_zeros = ((frozen_weight).cpu().float() != 0).float()
            W_frozen = W.detach().cpu().float() * non_zeros
            assert (torch.sum(W_frozen != frozen_weight.detach().cpu().float()) == 0)
            print("{} weight contains frozen_weight".format(name))

        weight = torch.tensor(frozen_weight).cuda()
        frozen_weights[name]['weight'] = weight.type(W.dtype)
        # mask for other weight, not the frozen weight
        assert weight.shape == W.shape
        if prune_ratio < 0.001:
            frozen_weights[name]['mask'] = torch.zeros(weight.shape).cuda().type(W.dtype)
        else:
            frozen_weights[name]['mask'] = (weight == 0).cuda().type(W.dtype)
        assert W.shape == frozen_weights[name]['mask'].shape

        if 'running' in name:
            frozen_weights[name]['weight'] = loaded_weights[name]
            frozen_weights[name]['mask'] = True

    return frozen_weights


def update_one_frozen_weight(W, mask, weight):
    return (W * mask.type(W.dtype)) + weight.type(W.dtype)


def _collect_dir_keys(configs, dir):
    if not isinstance(configs, dict):
        return

    for name in configs:
        if name not in dir:
            dir[name] = []
        dir[name].append(configs)
    for name in configs:
        _collect_dir_keys(configs[name], dir)


def _canonicalize_names(configs, model, logger):
    dir = {}
    collected_keys = _collect_dir_keys(configs, dir)
    for name in model.state_dict():
        cname = canonical_name(name)
        if cname == name:
            continue
        if name in dir:
            assert cname not in dir
            for parent in dir[name]:
                assert cname not in parent
                parent[cname] = parent[name]
                del parent[name]
            logger.info("Updating parameter from {} to {}".format(name, cname))


def load_configs(model, filename, logger):
    assert filename is not None, \
            "Config file must be specified"

    with open(filename, "r") as stream:
        try:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    _canonicalize_names(configs, model, logger)

    if "prune_ratios" in configs:
        config_prune_ratios = configs["prune_ratios"]

        count = 0
        prune_ratios = {}
        for name in model.state_dict():
            W = model.state_dict()[name]
            cname = canonical_name(name)

            if cname not in config_prune_ratios:
                continue
            count = count + 1
            prune_ratios[name] = config_prune_ratios[cname]
            if name != cname:
                logger.info("Map weight config name from {} to {}".\
                    format(cname, name))

        if len(prune_ratios) != len(config_prune_ratios):
            extra_weights = set(config_prune_ratios) - set(prune_ratios)
            for name in extra_weights:
                logger.warning("{} in config file cannot be found".\
                    format(name))


    return configs, prune_ratios


def apply_masks(model, frozen_weights, all_params=False):
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if name in frozen_weights:
                dtype = W.dtype
                W.data = update_one_frozen_weight(W,
                        frozen_weights[name]['mask'],
                        frozen_weights[name]['weight']).type(dtype)
                pass
        # debug only
        if all_params:
            for key in model.state_dict():
                if 'running' in key and frozen_weights[key]['mask'] is True:
                    model.state_dict()[key][:] = frozen_weights[key]['weight']

def check_gather_scatter_valid(model, num_buckets=16, num_ele_per_row=1, layers=None):
    if layers is None:
        layers = []
        for name, W in model.named_parameters():
            if 'weight' in name and 'bn' not in name and len(W.detach().cpu().numpy().shape)>1:
                layers.append(name)

    valid = True
    for name, W in model.named_parameters():
        if name in layers:
            this_valid = True

            np_W = W.detach().cpu().numpy()

            if len(np_W.shape) == 2:
                pass

            elif len(np_W.shape) == 3:
                co, ci, kl = np_W.shape
                np_W = np.moveaxis(np_W, 1, -1)
                np_W = np_W.reshape([co, ci * kl])

            elif len(np_W.shape) == 4:
                co, ci, kh, kw = np_W.shape
                np_W = np_W.reshape([co, ci, kh * kw])
                # convert from CoCiKhKw to CoKhKwCi format
                np_W = np.moveaxis(np_W, 1, -1)
                # merge Ci, Kh, Kw dimension
                np_W = np_W.reshape([co, ci * kh * kw])

            else:
                assert False, "Weight matrix not 2,3,4 dimensions!"

            assert len(np_W.shape) == 2, "Weight matrix is not 2-D, but {}".format(np_W.shape)
            sparsity = float(np.sum(np_W == 0))/np.size(np_W)
            #print(np_W.shape, name, sparsity)
            if sparsity < 0.5: # check validity onlly if sparsity > 0.5
                continue

            assert num_buckets % num_ele_per_row == 0, "#buckets not divisible by #element per row!"

            num_rows = num_buckets / num_ele_per_row
            start = 0
            end = start + num_rows
            while end < np_W.shape[0]:
                sub_array = np_W[int(start):int(end)]
                #print(sub_array.shape)
                # first check if # elements per row is the same
                non_zero_per_row = np.sum(sub_array != 0,axis=1)
                #print(non_zero_per_row.shape, non_zero_per_row)
                if np.max(non_zero_per_row) != np.min(non_zero_per_row):
                    print("Error: {} is not a vaild gather_scatter pattern: #element per row is not the same within a subarray! max:{},min:{},start:{},end:{}".format(name,np.max(non_zero_per_row),np.min(non_zero_per_row),start,end))
                    valid = False
                    this_valid = False
                    #break
                # Then check if modulo is the same
                [non_zeros_row_idx, non_zeros_col_idx] = np.nonzero(sub_array)

                non_zeros_col_idx = non_zeros_col_idx % num_buckets
                #print(non_zeros_col_idx)
                col_idx_cnt = np.histogram(non_zeros_col_idx, bins=list(x - 0.5 for x in range(num_buckets+1)))[0]
                #print(col_idx_cnt)
                if np.max(col_idx_cnt) != np.min(col_idx_cnt):
                    print("Error: {} is not a vaild gather_scatter pattern: #element in each bucket is not the same within a subarray!max:{},min:{}".format(name,np.max(col_idx_cnt),np.min(col_idx_cnt)))
                    this_valid = False
                    #break
                #input("?")
                start = end
                end = start + num_rows
            if this_valid:
                print('Layer {}, shape {}, sparsity {} is a valid gather-scatter pattern.'.format(name, W.detach().cpu().numpy().shape, sparsity))
    return valid

def check_hierarchy(dicts, layer_names=[], logger=None):
    assert len(dicts) == 2, 'Number of models not equal to 2'
    # model[0] is larger than model[1]

    if len(layer_names) == 0:
        for name in dicts[1]:
            if 'weight' in name and 'bn' not in name and 'downsample.1' not in name:
                layer_names.append(name)
    if logger:
        p = logger.info
    else:
        p = print

    # generate mask for model[1]
    masks = {}
    for name, W in dicts[1].items():
        if name in layer_names:
            non_zeros = W != 0
            masks[name] = non_zeros

            np_large = dicts[0][name]
            np_small = dicts[1][name]

            diff = np.abs(np_large[masks[name]] - np_small[masks[name]])
            sum_diff = np.sum(diff)

            if sum_diff > 1e-6:
                p("Model[0] is not a superset of Model[1]")
                return False
            else:
                p("model[0] is superset of model[1] in layer {}".format(name))
    return True

def generate_seq_gap_yaml(layer_list_file=None, output_folder='test/'):
    if layer_list_file is None:
        return
    # read txt files
    # layers in each steps are lists, such as
    # 0.9 # sparsity
    #[ [ layer1, layer2 ],
    #  [ layer3  ],
    #  [ layer4, layer5, layer6]]
    L = ""
    with open(layer_list_file,'r') as f:
        lines = f.readlines()
        sparsity = float(lines[0])
        for line in lines[1:]:
            if line[0] != '#':
                line = line.strip()
                L += line

    L = L.split('],[')
    for i in range(len(L)):
        L[i] = L[i].replace('[','')
        L[i] = L[i].replace(']','')
        L[i] = L[i].strip()
        L[i] = L[i].split(',')
        while('' in L[i]) :
            L[i].remove('')

    #print(L)
    #input("?")
    os.system('mkdir -p {}'.format(output_folder))
    # generate yaml files
    # step 0, first block dense, all others sparse, no mask
    step_0_yaml = output_folder + "/step_0_{}.yaml".format(sparsity)
    with open(step_0_yaml, 'w') as f:
        f.write("prune_ratios:\n")
        for ll in L[1:]:
            for str in ll:
                f.write("  "+str+": {}\n".format(sparsity))
        f.write("\nrho: 0.001\n")
        f.write("masked_layers:\n")
        for ll in L[1:]:
            for str in ll:
                f.write("  "+str+": {}\n".format(sparsity))
    # step 1 to n
    for i in range(1,len(L)):
        step_i_yaml = output_folder + "/step_{}_{}.yaml".format(i,sparsity)
        with open(step_i_yaml, 'w') as f:
            f.write("prune_ratios:\n")
            for str in L[i-1]:
                f.write("  "+str+": {}\n".format(sparsity))
            f.write("\nrho: 0.001\n\n")

            f.write("masked_layers:\n")
            for k in range(len(L)):
                if k == i:
                    continue
                for str in L[k]:
                    f.write("  "+str+": {}\n".format(sparsity))

    # last step, grow first block, prune last block, mask all but first block
    step_last_yaml = output_folder + "/step_{}_{}.yaml".format(len(L),sparsity)
    with open(step_last_yaml, 'w') as f:
        f.write("prune_ratios:\n")
        for str in L[-1]:
            f.write("  "+str+": {}\n".format(sparsity))

        f.write("\nrho: 0.001\n\n")

        f.write("masked_layers:\n")
        for k in range(len(L)):
            if k == 0:
                continue
            for str in L[k]:
                f.write("  "+str+": {}\n".format(sparsity))

def prune_MLP(old_model, new_model):
    cfg_mask = []
    cfg = []
    for m in old_model.modules():
        if isinstance(m, nn.Linear):
            out_channels = m.weight.data.shape[0]
            weight_copy = m.weight.data.abs().cpu().numpy()
            norm = np.sum(weight_copy, axis=(1))
            index = np.where(norm == 0)[0]
            mask = torch.ones(out_channels)
            mask[index.tolist()] = 0
            cfg_mask.append(mask)
            cfg.append(out_channels - len(index.tolist()))
            continue
    cfg = cfg[:-1]
    newmodel = new_model
    idx = 0
    for [m0, m1] in zip(old_model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Linear):
            mask = cfg_mask[idx]
            index = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if index.size == 1:
                index = np.resize(index, (1,))
            if idx == 0:
                w = m0.weight.data[index,:].clone()
                m1.weight.data = w.clone()
                b = m0.bias.data[index].clone()
                m1.bias.data = b.clone()
            else:
                prev_mask = cfg_mask[idx-1]
                prev_index = np.squeeze(np.argwhere(np.asarray(prev_mask.cpu().numpy())))
                if prev_index.size == 1:
                    prev_index = np.resize(prev_index, (1,))
                tmp = m0.weight.data[index,:].clone()
                w = tmp[:, prev_index].clone()
                m1.weight.data = w.clone()
                b = m0.bias.data[index].clone()
                m1.bias.data = b.clone()
            idx += 1
            continue
    return newmodel
