import collections.abc
from itertools import repeat
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)          # (H, W)
        patch_size = to_2tuple(patch_size)      # (P, P)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])       # N = (H // P) * (W // P)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # (B, C, H, W) -> (B, D, (H//P), (W//P)) -> (B, D, N) -> (B, N, D)
        # D=embed_dim=768, N=num_patches=(H//P)*(W//P)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def generate_cross_modal_mask(ehr_cls_index, cxr_cls_index, note_cls_index, total_lens):
    mask = torch.ones(total_lens, total_lens)
    # task
    mask[0, :] = 0

    # m1+m2+m3
    mask[1, 1] = 0
    mask[1, ehr_cls_index + 1: cxr_cls_index] = 0
    mask[1, cxr_cls_index + 1: note_cls_index] = 0
    mask[1, note_cls_index + 1: total_lens] = 0

    # m1+m2 mask
    mask[2, 2] = 0
    mask[2, ehr_cls_index + 1: cxr_cls_index] = 0
    mask[2, cxr_cls_index + 1: note_cls_index] = 0

    # m1+m3 mask
    mask[3, 3] = 0
    mask[3, ehr_cls_index + 1: cxr_cls_index] = 0
    mask[3, note_cls_index + 1: total_lens] = 0

    # m2+m3 mask
    mask[4, 4] = 0
    mask[4, cxr_cls_index + 1: note_cls_index] = 0
    mask[4, note_cls_index + 1: total_lens] = 0

    # m1
    mask[ehr_cls_index, ehr_cls_index: cxr_cls_index] = 0
    mask[ehr_cls_index+1: cxr_cls_index, ehr_cls_index+1: cxr_cls_index] = 0

    # m2
    mask[cxr_cls_index, cxr_cls_index: note_cls_index] = 0
    mask[cxr_cls_index+1: note_cls_index, cxr_cls_index+1: note_cls_index] = 0

    # m3
    mask[note_cls_index, note_cls_index: total_lens] = 0
    mask[note_cls_index+1: total_lens, note_cls_index+1: total_lens] = 0

#     mask = mask.triu() + mask.t().tril()
#     mask = mask - torch.diag_embed(mask.diagonal())

    return mask


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.relu(out)
        return out


class MoE(nn.Module):
    def __init__(self, query_size, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.task_gate = nn.Parameter(torch.zeros(query_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, task, x, train, noise_epsilon=1e-2):
        modal_logits = x @ self.w_gate
        task_logits = task @ self.task_gate
        clean_logits = modal_logits + task_logits
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, query, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(query, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss