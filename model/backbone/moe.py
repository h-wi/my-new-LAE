from collections import OrderedDict
from typing import Tuple, Union

import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from collections import Counter

from ..pet_mixin import AdapterMixin
from .block import Mlp, Attention

_logger = logging.getLogger(__name__)

global_taskid = 0
global_is_train=True

NAME_SEP = "/"
def get_submodule(
    module: nn.Module,
    name: str,
    default: nn.Module = nn.Identity(),
):
    names = name.split(NAME_SEP)
    while names:
        module = getattr(module, names.pop(0), default)
    return module

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor. -> input 에 대한 mini-batch를 만들고, 각 Expert에 대해 통합된 Output을 산출.

    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    -> 들어온 input tensor에 대해 각 expert한테 먹일 input tensor를 만든다.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    -> gates에 저장된 중요도에 따라서 element-wise summation.

    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.

    Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)

    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

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
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        # import pdb; pdb.set_trace()

        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weighted summation

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module, AdapterMixin):
    def __init__(self, d_model: int, n_head: int, expert: int,attn_mask: torch.Tensor = None): #d_model: dimension of model
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        # block modules of previous one
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.ln_2 = LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(
            d_model,
            num_heads=n_head,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * 4.0)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.attn_mask = attn_mask
        self.is_train = global_is_train

        self.step = 1 # router의 개수?
        self.top_k = 2
        self.experts_num = expert
        self.noisy_gating = True

        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()

        self.choose_map_image = torch.zeros([self.experts_num]) # image를 학습한 각 Adapter에 대한 Weights
        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()

        for i in range(self.step):
            self.router_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)) # d_model : width of transformer, pre-trained 모델의 width, hidden dimension인가?
            self.w_noise_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)) # noisy gating? 대충 noise 추가해서 그냥 Router랑 비교하겠다는 듯.

        #  self.taskid = None
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2): # for sparsely-gated MoE
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x)) # TODO: MSA layer와 parallel 해야 함

        if global_taskid is not None and hasattr(self, 'adapters'): # adapter가 달린 block에 대해서만 진행
            B, C, D = x.shape
            x_re = x[:, 0, :] # [batch, 197, 768]
            gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router_list[global_taskid],
                                                  self.w_noise_list[global_taskid])
            importance = gates.sum(0)

            nonzero_indices = torch.nonzero(gates)
            counter = Counter(nonzero_indices[:, 1].tolist())
            for number, count in counter.items():
                    self.choose_map_image[number] = self.choose_map_image[number] + count

            # gates: expert끼리 softmax한 결과
            dispatcher = SparseDispatcher(self.experts_num, gates)

            # import pdb;
            # pdb.set_trace()

            # 여러 개의 Expert에 대한 input을 만든다. -> 각 batch(sample)에 따라 어느 Expert에 보낼지 나눔
            # expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))
            expert_inputs = dispatcher.dispatch(x.view(B, -1)) # [24, 197, 768] -> [24, 197*768]

            # expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
            #                                                               x.shape[0], x.shape[2]).to(x), add_residual=False)
            #                   for i in range(self.experts_num)]
            expert_outputs = [self.adapters['attn'][i](get_submodule(self, 'attn'),
                                                       expert_inputs[i].view(expert_inputs[i].shape[0], C, D).to(x))
                              for i in range(self.experts_num)]
            # TODO: adapter.forward(module, input) 형태 ->  지금은 mlp랑 parallel, msa랑도 해줘야..

            i = 0
            while i < len(expert_outputs):
                if expert_outputs[i].shape[0] == 0: # 할당된 sample이 없으면 NAGA
                    expert_outputs.pop(i)
                else:
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1) # [expert's batch, 197*768] ??
                    i += 1
            y = dispatcher.combine(expert_outputs)
            y = y.view(B, C, D)
            x = x + self.mlp(self.norm2(x)) + y # MLP Layer와 이미 Parallel
        else:
            x = x + self.mlp(self.norm2(x))
        return x