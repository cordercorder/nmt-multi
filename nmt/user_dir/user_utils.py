import torch
import torch.nn as nn
import logging

from argparse import Namespace


logger = logging.getLogger(__name__)


def get_tgt_langs(args: Namespace):
    extra_lang_pairs = [p for _, v in args.extra_lang_pairs.items() for p in v.split(",")] if args.extra_lang_pairs else []
    lang_pairs = args.lang_pairs.split(",") if isinstance(args.lang_pairs, str) else args.lang_pairs + extra_lang_pairs
    # sort to guarantee order
    tgt_langs = sorted(set(lang_pair.split("-")[1] for lang_pair in lang_pairs))
    return tgt_langs


def get_lang_embedding(num_langs, embedding_dim):
    m = nn.Embedding(num_langs, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def apply_language_specific_layer_norm(
    input: torch.Tensor, 
    tgt_lang_id: torch.Tensor,
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
    eps: float = 1e-5
):
    # input is T x B x C
    # T x B x 1
    var, mean = torch.var_mean(input, dim=-1, unbiased=False, keepdim=True)
    # B x C
    scale = layer_norm_weight.index_select(dim=0, index=tgt_lang_id)
    offset = layer_norm_bias.index_select(dim=0, index=tgt_lang_id)
    return (input - mean) * torch.rsqrt(var + eps) * scale + offset


def apply_language_specific_linear_transformation(
    input: torch.Tensor, 
    tgt_lang_id: torch.Tensor,
    lang_aware_linear_transformation: torch.Tensor
):
    # input is T x B x C
    # B x C x C
    transformation = lang_aware_linear_transformation.index_select(dim=0, index=tgt_lang_id)
    # T x B x C -> B x T x C
    input = input.transpose(0, 1)
    return input.matmul(transformation).transpose(0, 1)


def get_routing_prob(epoch: int, max_epoch: int, num_languages: int, device: str):
    if epoch >= max_epoch:
            return None

    start_prob = 1.0 / num_languages
    end_prob = 1.0
    increased_prob_per_epoch = (end_prob - start_prob) / (max_epoch - 1)
    current_langs_prob = start_prob + (epoch - 1) * increased_prob_per_epoch
    others_prob = (1.0 - current_langs_prob) / (num_languages - 1)
    prob = torch.empty(num_languages, num_languages, device=device).fill_(others_prob)
    prob.diagonal()[:] = current_langs_prob
    return prob
