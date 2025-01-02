
import torch
import logging
import sys
import os
import numpy as np
import argparse
import time
import random
import copy
from . import utils_pr
from .admm import weight_growing, weight_pruning, ADMM

def prune_parse_arguments(parser):
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight',
                    help="retrain mask pattern")
    parser.add_argument('--sp-update-init-method', type=str, default='zero',
                        help="mask update initialization method")
    parser.add_argument('--sp-mask-update-freq', type=int, default=5,
                        help="how many epochs to update sparse mask")
    parser.add_argument('--retrain-mask-sparsity', type=float, default=-1.0,
                    help="sparsity of a retrain mask, used when retrain-mask-pattern is set to NOT being 'weight' ")
    parser.add_argument('--retrain-mask-seed', type=int, default=None,
                    help="seed to generate a random mask")
    parser.add_argument('--sp-prune-before-retrain', action='store_true',
                        help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--output-compressed-format', action='store_true',
                        help="output compressed format")
    parser.add_argument("--sp-grad-update", action="store_true",
                        help="enable grad update when training in random GaP")
    parser.add_argument("--sp-grad-decay", type=float, default=0.98,
                        help="The decay number for gradient")
    parser.add_argument("--sp-grad-restore-threshold", type=float, default=-1,
                        help="When the decay")
    parser.add_argument("--sp-global-magnitude", action="store_true",
                        help="Use global magnitude to prune models")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None,
                        help="using another sparse model to init sparse mask")
    parser.add_argument("--sp-restore-blk", action="store_true",
                        help="Use previous checkpoint to grow block, continue train")


class Retrain(object):
    def __init__(self, args, model, logger=None, pre_defined_mask=None, seed=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask # as model's state_dict
        self.sparsity = self.args.retrain_mask_sparsity
        self.seed = self.args.retrain_mask_seed
        self.sp_mask_update_freq = self.args.sp_mask_update_freq
        self.update_init_method = self.args.sp_update_init_method

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.masks = {}
        self.masked_layers = {}
        self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger)

        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None


        if "fixed_layers" in self.configs:
            self.fixed_layers = self.configs['fixed_layers']
        else:
            self.fixed_layers = None
        self.fixed_layers_save = {}

        if "upper_bound" in self.configs:
            self.upper_bound = self.configs['upper_bound']
        else:
            self.upper_bound = None
        if "lower_bound" in self.configs:
            self.lower_bound = self.configs['lower_bound']
        else:
            self.lower_bound = None
        if "mask_update_decay_epoch" in self.configs:
            self.mask_update_decay_epoch = self.configs['mask_update_decay_epoch']
        else:
            self.mask_update_decay_epoch = None

        if "seq_gap_layer_indices" in self.configs:
            self.seq_gap_layer_indices = self.configs['seq_gap_layer_indices']
            self.all_part_name_list = []
        else:
            self.seq_gap_layer_indices = None

        if "weight_mutate_epoch" in self.configs:
            self.weight_mutate_epoch = self.configs['weight_mutate_epoch']
        else:
            self.weight_mutate_epoch = None
        if "mutation_ratio" in self.configs:
            self.mutation_ratio = self.configs["mutation_ratio"]


        self.init()

    def init(self):

        self.generate_mask(self.pre_defined_mask)


    def apply_masks(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    W.mul_((self.masks[name] != 0).type(dtype))
                    # W.data = (W * (self.masks[name] != 0).type(dtype)).type(dtype)
                    pass

    def apply_masks_on_grads(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
                    pass

    def show_masks(self, debug=False):
        with torch.no_grad():
            if debug:
                name = 'module.layer1.0.conv1.weight'
                np_mask = self.masks[name].cpu().numpy()
                np.set_printoptions(threshold=sys.maxsize)
                print(np.squeeze(np_mask)[0], name)
                return
            for name, W in self.model.named_parameters():
                if name in self.masks:
                    np_mask = self.masks[name].cpu().numpy()
                    np.set_printoptions(threshold=sys.maxsize)
                    print(np.squeeze(np_mask)[0], name)


    def update_mask(self, epoch, batch_idx):
        # a hacky way to differenate random GaP and others
        if not self.mask_update_decay_epoch:
            return
        if batch_idx != 0:
            return

        freq = self.sp_mask_update_freq

        bound_index = 0

        try: # if mask_update_decay_epoch has only one entry
            int(self.mask_update_decay_epoch)
            freq_decay_epoch = int(self.mask_update_decay_epoch)
            try: # if upper/lower bound have only one entry
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError: # if upper/lower bound have multiple entries
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity
                if epoch >= freq_decay_epoch:
                    freq *= 1
                    bound_index += 1
        except ValueError: # if mask_update_decay_epoch has multiple entries
            freq_decay_epoch = self.mask_update_decay_epoch.split('-')
            for i in range(len(freq_decay_epoch)):
                freq_decay_epoch[i] = int(freq_decay_epoch[i])

            try:
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError:
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity

                if len(freq_decay_epoch) + 1 <= len(upper_bound): # upper/lower bound num entries enough for all update
                    for decay in freq_decay_epoch:
                        if epoch >= decay:
                            freq *= 1
                            bound_index += 1
                else: # upper/lower bound num entries less than update needs, use the last entry to do rest updates
                    for idx, _ in enumerate(upper_bound):
                        if epoch >= freq_decay_epoch[idx] and idx != len(upper_bound) - 1:
                            freq *= 1
                            bound_index += 1

        lower_bound_value = float(lower_bound[bound_index])
        upper_bound_value = float(upper_bound[bound_index])

        if epoch % freq == 0:
            '''
            calculate prune_part and grow_part for sequential GaP, if no seq_gap_layer_indices specified in yaml file,
            set prune_part and grow_part to all layer specified in yaml file as random GaP do.
            '''
            prune_part, grow_part = self.seq_gap_partition()

            # load restore checkpoint to grow
            blk_state = None
            if self.args.sp_restore_blk:
                print("loading previous checkpoint to restore grow partition")
                restore_path = self.args.checkpoint_dir + "checkpoint-{}.pth.tar".format(epoch - 13)
                if os.path.isfile(restore_path):
                    blk_checkpoint = torch.load(restore_path)
                    blk_state = blk_checkpoint['state_dict'] if "state_dict" in blk_checkpoint else blk_checkpoint
                else:
                    print("\n * Resume block model not exist!\n")
                    time.sleep(5)

            with torch.no_grad():
                sorted_to_prune = None
                if self.args.sp_global_magnitude:
                    total_size = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        total_size += W.data.numel()
                    to_prune = np.zeros(total_size)
                    index = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        size = W.data.numel()
                        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
                        index += size
                    sorted_to_prune = np.sort(to_prune)

                # import pdb; pdb.set_trace()
                for name, W in (self.model.named_parameters()):
                    if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                        continue

                    weight = W.cpu().detach().numpy()
                    weight_current_copy = copy.copy(weight)


                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    np_orig_mask = self.masks[name].cpu().detach().numpy()

                    print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,
                                                                    str(num_nonzeros),
                                                                    str(total_num),
                                                                    str(sparsity))))

                    ############## pruning #############
                    pruned_weight_np = None
                    if name in prune_part:
                        sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type)
                        sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+")
                        for i in range(len(sparsity_type_list)):
                            sparsity_type = sparsity_type_list[i]
                            print("* sparsity type {} is {}".format(i, sparsity_type))
                            self.args.sp_admm_sparsity_type = sparsity_type

                            pruned_mask, pruned_weight = weight_pruning(self.args,
                                                                        self.configs,
                                                                        name,
                                                                        W,
                                                                        lower_bound_value)
                            self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                            # pruned_mask_np = pruned_mask.cpu().detach().numpy()
                            pruned_weight_np = pruned_weight.cpu().detach().numpy()

                            W.mul_(pruned_mask.cuda())


                            non_zeros_prune = pruned_weight_np != 0
                            num_nonzeros_prune = np.count_nonzero(non_zeros_prune.astype(np.float32))
                            print(("==> PRUNE: {}: {}, {}, {}".format(name,
                                                             str(num_nonzeros_prune),
                                                             str(total_num),
                                                             str(1 - (num_nonzeros_prune * 1.0) / total_num))))

                            self.masks[name] = pruned_mask.cuda()


                    ############## growing #############
                    if name in grow_part:
                        if pruned_weight_np is None: # use in seq gap
                            pruned_weight_np = weight_current_copy

                        if self.args.sp_restore_blk and blk_state is not None:
                            print("restoring previous dense block to continue training")
                            l_name = name
                            l_weight = blk_state[l_name]
                            self.model.state_dict()[l_name].data.copy_(l_weight)

                            print("==> migrated layer {} from {}".format(l_name, restore_path))

                        updated_mask = weight_growing(self.args,
                                                      name,
                                                      pruned_weight_np,
                                                      lower_bound_value,
                                                      upper_bound_value,
                                                      self.update_init_method)



                        # np_updated_mask = np_updated_zero_one_mask
                        # # calculate grad mask
                        # if self.args.sp_grad_update:
                        #     # import pdb; pdb.set_trace()
                        #     decay_factor = self.args.sp_grad_decay
                        #     np_updated_mask = (pruned_mask_np * decay_factor) + \
                        #         (np_updated_zero_one_mask * (pruned_mask_np == 0))
                        #     if self.args.sp_grad_restore_threshold > 0:
                        #         restore_mask = (np_updated_mask != 0).astype(np.float32) * \
                        #             (np_updated_mask < self.args.sp_grad_restore_threshold).astype(np.float32)
                        #         np_updated_mask = (np_updated_mask != 0).astype(np.float32) * \
                        #             (restore_mask.astype(np.float32) + (1-restore_mask) * np_updated_mask)
                        # updated_mask = torch.from_numpy(np_updated_mask).cuda()
                        self.masks[name] = updated_mask
                        pass

    def weight_mutate(self, epoch, batch_idx):
        '''
        NOTE:
            This part is only for mutation, not for growing to lower sparsity

        '''

        if self.weight_mutate_epoch == None:
            return
        if batch_idx != 0:
            return

        try: # if weight_mutate_epoch has only one entry
            int(self.weight_mutate_epoch)
            weight_mutate_epoch = [int(self.weight_mutate_epoch)]
        except ValueError: # if weight_mutate_epoch has multiple entries
            weight_mutate_epoch = self.weight_mutate_epoch.split('-')
            for i in range(len(weight_mutate_epoch)):
                weight_mutate_epoch[i] = int(weight_mutate_epoch[i])


        if epoch in weight_mutate_epoch and epoch != 0:  # epoch 0 before weight updating, no gradient
            with torch.no_grad():
                if "pattern" in self.args.sp_admm_sparsity_type:
                    pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
                    pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
                    pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
                    pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

                    pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
                    pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
                    pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
                    pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53

                    patterns_dict = {1: pattern1,
                                     2: pattern2,
                                     3: pattern3,
                                     4: pattern4,
                                     5: pattern5,
                                     6: pattern6,
                                     7: pattern7,
                                     8: pattern8
                                     }

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                            continue

                        weight_np = W.cpu().detach().numpy()

                        conv_kernel_indicate = np.sum(weight_np, axis=(2, 3))
                        conv_kernel_indicate = conv_kernel_indicate != 0
                        conv_kernel_indicate = conv_kernel_indicate.astype(np.float32)
                        conv_kernel_indicate_shape = conv_kernel_indicate.shape
                        conv_kernel_indicate_1d = conv_kernel_indicate.reshape(1, -1)[0]

                        ones_indices = np.where(conv_kernel_indicate_1d == 1)[0]

                        num_mutate_kernel = int(self.mutation_ratio * np.size(ones_indices))

                        indices = np.random.choice(ones_indices,
                                                   num_mutate_kernel,
                                                   replace=False)

                        conv_kernel_indicate_1d[indices] = -1
                        conv_kernel_indicate = conv_kernel_indicate_1d.reshape(conv_kernel_indicate_shape)

                        c = np.where(conv_kernel_indicate == -1)

                        for idx in range(len(c[0])):
                            target_kernel = weight_np[c[0][idx], c[1][idx], :, :]
                            print(target_kernel)
                            target_kernel_1d = target_kernel.reshape(1, -1)
                            non_zero_values = target_kernel_1d[np.where(target_kernel_1d != 0)]
                            np.random.shuffle(non_zero_values)

                            pick_pattern = random.choice(list(patterns_dict.values()))

                            mutate_kernel = np.ones_like(target_kernel)

                            for index in pick_pattern:
                                mutate_kernel[index[0], index[1]] = 0

                            shape_k = mutate_kernel.shape
                            mutate_kernel_1d = mutate_kernel.reshape(1, -1)
                            idx = 0
                            for i in range(len(mutate_kernel_1d[0])):
                                if mutate_kernel_1d[0][i] == 1:
                                    mutate_kernel_1d[0][i] = non_zero_values[idx]
                                    idx += 1

                            mutate_kernel = mutate_kernel_1d.reshape(shape_k)
                            weight_np[c[0][idx], c[1][idx], :, :] = 0
                            weight_np[c[0][idx], c[1][idx], :, :] += mutate_kernel

                        non_zeros_updated = weight_np != 0
                        non_zeros_updated = non_zeros_updated.astype(np.float32)
                        np_updated_mask = non_zeros_updated
                        updated_mask = torch.from_numpy(np_updated_mask).cuda()
                        self.masks[name] = updated_mask

                        cuda_pruned_weights = torch.from_numpy(weight_np)
                        W.data = cuda_pruned_weights.cuda().type(W.dtype)

                elif "irregular" in self.args.sp_admm_sparsity_type:
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                            continue

                        weight_np = W.cpu().detach().numpy()
                        grad_np = (W.grad).cpu().detach().numpy()

                        shape = np.shape(weight_np)

                        mask_before_update = weight_np != 0
                        mask_before_update = mask_before_update.astype(np.float32)

                        mask_updated = copy.copy(mask_before_update)
                        mask_updated_1d = mask_updated.reshape(1, -1)[0]
                        # print(np.sum(mask_updated_1d))

                        mask_complement = weight_np == 0
                        mask_complement = mask_complement.astype(np.float32)

                        grad_on_zero_weight = np.abs(grad_np * mask_complement)
                        # imp_on_nonzero_weight = np.power(weight_np * grad_np, 2)
                        imp_on_nonzero_weight = np.abs(weight_np)

                        # add from zeros
                        grad_on_zero_weight_1d = grad_on_zero_weight.reshape(1, -1)[0]
                        num_added_zeros = int(self.mutation_ratio * np.size(grad_on_zero_weight_1d))

                        # idx_added = np.argpartition(grad_on_zero_weight_1d, -num_added_zeros)[-num_added_zeros:]
                        zeros_indices = np.where(grad_on_zero_weight_1d != 0)[0]
                        idx_added = np.random.choice(zeros_indices,
                                                   num_added_zeros,
                                                   replace=False)


                        mask_updated_1d[idx_added] = 1
                        # print(np.sum(mask_updated_1d))

                        # remove least important from non-zero weights
                        imp_on_nonzero_weight_1d = imp_on_nonzero_weight.reshape(1, -1)[0]
                        non_zeros_idx = np.nonzero(imp_on_nonzero_weight_1d)
                        extract_nonzero_imp = np.array(imp_on_nonzero_weight_1d[non_zeros_idx])
                        idx_removed = np.argpartition(extract_nonzero_imp, num_added_zeros)[:num_added_zeros]
                        extract_nonzero_imp[idx_removed] = 0
                        imp_on_nonzero_weight_1d[non_zeros_idx] = extract_nonzero_imp

                        imp_on_nonzero_weight_1d = imp_on_nonzero_weight_1d != 0
                        imp_on_nonzero_weight_1d = imp_on_nonzero_weight_1d.astype(np.float32)
                        temp1 = np.logical_xor(mask_updated_1d, imp_on_nonzero_weight_1d)
                        temp2 = (mask_before_update.reshape(1, -1)[0]) * temp1
                        temp3 = temp1 - temp2
                        mask_updated_1d = temp3 + imp_on_nonzero_weight_1d
                        mask_updated_1d = mask_updated_1d != 0
                        mask_updated_1d = mask_updated_1d.astype(np.float32)
                        # print(np.sum(mask_updated_1d))
                        # time.sleep(300)

                        updated_mask = mask_updated_1d.reshape(shape)

                        updated_mask_cuda = torch.from_numpy(updated_mask).cuda()
                        self.masks[name] = updated_mask_cuda

                        weight_np_update = weight_np * updated_mask
                        cuda_pruned_weights = torch.from_numpy(weight_np_update)
                        W.data = cuda_pruned_weights.cuda().type(W.dtype)

                        # double check mask sparsity
                        non_zeros_before = mask_before_update != 0
                        non_zeros_before = non_zeros_before.astype(np.float32)
                        num_nonzeros_before = np.count_nonzero(non_zeros_before)
                        total_num_before = non_zeros_before.size
                        sparsity_before = 1 - (num_nonzeros_before * 1.0) / total_num_before

                        non_zeros_after = updated_mask != 0
                        non_zeros_after = non_zeros_after.astype(np.float32)
                        num_nonzeros_after = np.count_nonzero(non_zeros_after)
                        total_num_after = non_zeros_after.size
                        sparsity_after = 1 - (num_nonzeros_after * 1.0) / total_num_after
                        print("\n==> {}, BEFORE MUTATE: {}, AFTER MUTATE: {}".format(name,
                                                                                     sparsity_before,
                                                                                     sparsity_after))





    def cut_all_partitions(self, all_update_layer_name):
        # calculate the number of partitions and range
        temp1 = str(self.seq_gap_layer_indices)
        temp1 = (temp1).split('-')
        num_partition = len(temp1) + 1
        head = 0
        end = len(all_update_layer_name)
        all_range = []

        for i, indice in enumerate(temp1):
            assert int(indice) < end, "\n\n * Error, seq_gap_layer_indices must within range [0, {}]".format(end - 1)
        assert len(temp1) == len(set(temp1)), "\n\n * Error, seq_gap_layer_indices can not have duplicate element"

        for i in range(0, num_partition):
            if i == 0:
                range_i = (head, int(temp1[i]))
            elif i == num_partition - 1:
                range_i = (int(temp1[i - 1]), end)
            else:
                range_i = (int(temp1[i - 1]), int(temp1[i]))
            print(range_i)
            all_range.append(range_i)

        for j in range(num_partition):
            range_j = all_range[j]
            self.all_part_name_list.append(all_update_layer_name[range_j[0]:range_j[1]])

    def seq_gap_partition(self):
        prune_part = []
        grow_part = []

        if self.seq_gap_layer_indices is None: # Random Gap: add all layer name in prune part and grow part list
            for name, _ in self.model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                prune_part.append(name)
                grow_part.append(name)
        else: # Sequential gap One-run: partition model
            all_update_layer_name = []
            for name, _ in self.model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                all_update_layer_name.append(name)
            if not self.all_part_name_list:
                self.cut_all_partitions(all_update_layer_name) # get all partitions by name in self.all_part_name_list

            to_grow = (self.all_part_name_list).pop(0)
            to_prune = self.all_part_name_list

            for layer in to_grow:
                grow_part.append(layer)
            for part in to_prune:
                for layer in part:
                    prune_part.append(layer)

            (self.all_part_name_list).append(to_grow)

        return prune_part, grow_part


    def update_grad(self, opt):
        if not self.args.sp_grad_update:
            return
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    W.grad.mul_(self.masks[name])
                    pass


    def fix_layer_weight_save(self):
        if self.fixed_layers == None:
            return
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if (utils_pr.canonical_name(name) not in self.fixed_layers) and (name not in self.fixed_layers):
                    continue
                W_cpu = W.cpu().detach().numpy()
                self.fixed_layers_save[name] = torch.from_numpy(W_cpu).float().cuda()

        #
        # grad_masks = {}
        # for name, W in (self.model.named_parameters()):
        #     if (utils_pr.canonical_name(name) not in self.fixed_layers) and (name not in self.fixed_layers):
        #         continue
        #     weight = W.cpu().detach().numpy()
        #     non_zeros = np.zeros_like(weight)
        #     zero_mask = torch.from_numpy(non_zeros).cuda()
        #     grad_masks[name] = zero_mask
        #
        # with torch.no_grad():
        #     for name, W in (self.model.named_parameters()):
        #         if name in grad_masks:
        #             W.grad *= grad_masks[name]

    def fix_layer_weight_restore(self):
        if self.fixed_layers == None:
            return

        with torch.no_grad():
            for name, W in self.model.named_parameters():
                if name in self.fixed_layers_save:
                    W.copy_(self.fixed_layers_save[name])


    def generate_mask(self, pre_defined_mask=None):
        masks = {}
        # import pdb; pdb.set_trace()
        if self.pattern == 'weight':


            with torch.no_grad():
                for name, W in (self.model.named_parameters()):

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

            #for name in masks:
            #    print("Current mask includes:", name)
                    #if 'weight' in name:
                    #    print(name, (np.sum(non_zeros) + 0.0) / np.size(non_zeros) )
                #exit()



        elif self.pattern == 'random':
            if self.seed is not None:
                print("Setting the random mask seed as {}".format(self.seed))
                np.random.seed(self.seed)

            with torch.no_grad():
                # self.sparsity (args.retrain_mask_sparsity) will override prune ratio config file
                if self.sparsity > 0:
                    sparsity = self.sparsity

                    for name, W in (self.model.named_parameters()):
                        if 'weight' in name and 'bn' not in name:
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        self.masks[name] = zero_mask

                else: #self.sparsity < 0

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        if name in self.prune_ratios:
                            # Use prune_ratio[] to indicate which layers to random masked
                            sparsity = self.prune_ratios[name]
                            '''
                            if sparsity < 0.001:
                                continue
                            '''
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()

                        self.masks[name] = zero_mask

                # # DEBUG:
                DEBUG = False
                if DEBUG:
                    for name, W in (self.model.named_parameters()):
                        m = self.masks[name].detach().cpu().numpy()
                        total_ones = np.sum(m)
                        total_size = np.size(m)
                        print( name, m.shape, (total_ones+0.0)/total_size)

                #exit()
        # TO DO
        elif self.pattern == 'regular':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    if 'weight' in name and 'bn' not in name:

                        ouputSize, inputSize = W.data.shape[0], W.data.shape[1]
                        non_zeros = np.zeros(W.data.shape)
                        non_zeros = np.squeeze(non_zeros)

                        if 'sa1.conv_blocks.0.0.weight' in name or 'sa1.conv_blocks.1.0.weight' in name or 'sa1.conv_blocks.2.0.weight' in name:
                            non_zeros[::self.args.mask_sample_rate,::] = 1

                        else:
                            non_zeros[::self.args.mask_sample_rate,::self.args.mask_sample_rate] = 1

                        non_zeros = np.reshape(non_zeros, W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()

                    else:
                        non_zeros = 1 - np.zeros(W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask
        elif self.pattern == 'global_weight':
            with torch.no_grad():
                all_w = []
                all_name = []
                print('Concatenating all weights...')
                for name, W in self.model.named_parameters():
                    if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                        continue
                    all_w.append(W.detach().cpu().numpy().flatten())
                    all_name.append(name)
                np_w = all_w[0]
                for i in range(1,len(all_w)):
                    np_w = np.append(np_w, all_w[i])

                #print(np_w.shape)
                print("All weights concatenated!")
                print("Start sorting all the weights...")
                np_w = np.sort(np.abs(np_w))
                print("Sort done!")
                L = len(np_w)
                #print(np_w)
                if self.args.retrain_mask_sparsity >= 0.0:
                    thr = np_w[int(L * self.args.retrain_mask_sparsity)]

                    for name, W in self.model.named_parameters():
                        if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                            continue


                        np_mask = np.abs(W.detach().cpu().numpy())  > thr
                        print(name, np.size(np_mask), np.sum(np_mask), float(np.sum(np_mask))/np.size(np_mask) )

                        self.masks[name] = torch.from_numpy(np_mask).cuda()

                    total_non_zero = 0
                    total_size = 0
                    with open('gw_sparsity.txt','w') as f:
                        for name, W in sorted(self.model.named_parameters()):
                            if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                                continue
                            np_mask = self.masks[name].detach().cpu().numpy()
                            sparsity = 1.0 - float(np.sum(np_mask))/np.size(np_mask)
                            if sparsity < 0.5:
                                sparsity = 0.0

                            if sparsity < 0.5:
                                total_non_zero += np.size(np_mask)
                            else:
                                total_non_zero += np.sum(np_mask)
                            total_size += np.size(np_mask)

                            f.write("{}: {}\n".format(name,sparsity))
                    print("Thr:{}".format(thr))
                    print("{},{},{}".format(total_non_zero, total_size, float(total_non_zero)/total_size))
                    exit()



        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        elif self.pattern == "pre_defined":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.001:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        elif self.pattern == "pre_defined_ratio":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)

                    non_zeros_1d = non_zeros.flatten()
                    np.random.shuffle(non_zeros_1d)
                    non_zeros = np.reshape(non_zeros_1d, weight.shape)

                    # weight_training = self.model.state_dict()[name]
                    # weight_1d = (weight_training.cpu().detach().numpy()).flatten()
                    #
                    # p = np.random.permutation(len(non_zeros_1d))
                    # mask_shuffle = non_zeros_1d[p]
                    # weight_shuffle = weight_1d[p]

                    # non_zeros = np.reshape(mask_shuffle, weight.shape)
                    # weight_shuffle = np.reshape(weight_shuffle, weight.shape)
                    # weight_cuda = torch.from_numpy(weight_shuffle).cuda()
                    # (self.model.state_dict()[name]).data = weight_cuda.type(weight_training.dtype)

                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        #self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask


        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
