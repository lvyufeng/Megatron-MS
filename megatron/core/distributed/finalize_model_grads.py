# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import List

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (
            parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (
            parallel_state.is_rank_in_position_embedding_group()
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
            and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_word_embedding_grads_mm(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (
            parallel_state._EMBEDDING_GLOBAL_RANKS
            and parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        if hasattr(model_module.module, "language_model"):
            model_module = model_module.module.language_model
            if (model_module.pre_process or model_module.post_process) \
                    and model_module.share_embeddings_and_output_weights:
                weight = model_module.shared_embedding_or_output_weight()
                grad = weight.main_grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads_mm(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    is_parallel_state = parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS \
                        and parallel_state.is_rank_in_position_embedding_group() \
                        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    is_split_rank = config.pipeline_model_parallel_split_rank is not None
    if (
         is_parallel_state and is_split_rank
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    if not config.multimodal:
        _allreduce_word_embedding_grads(model, config)
        _allreduce_position_embedding_grads(model, config)
    else:
        _allreduce_word_embedding_grads_mm(model, config)
        _allreduce_position_embedding_grads_mm(model, config)


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
            config.sequence_parallel or config.qk_layernorm
    ):
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if (
                        param.requires_grad
                        and getattr(param, 'sequence_parallel', False)
                        or 'q_layernorm' in name
                        or 'k_layernorm' in name
                ):
                    grad = param.main_grad
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


def _allreduce_duplicate_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce duplicate param grads .
    """

    # All-reduce duplicate parameters across model parallel nodes
    grads = []
    for model_chunk in model:
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
            if "vision_model.Qformer" in name or \
                    "vision_model.query_tokens" in name or \
                    "vision_model.norm" in name or \
                    "projection" in name or \
                    "vision_model.c_abstractor" in name:
                grad = param.main_grad
                grads.append(grad.data)
    if grads:
        data_tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        for item in grads:
            item /= data_tensor_parallel_world_size
            torch.distributed.all_reduce(item, group=parallel_state.get_tensor_model_parallel_group())


def finalize_model_grads(model: List[torch.nn.Module]):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied).
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # For Multimodal: all-reduce duplicate grads if needed.
    if config.timers is not None:
        config.timers('duplicate-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time)
    _allreduce_duplicate_grads(model, config)
    if config.timers is not None:
        config.timers('duplicate-grads-all-reduce').stop()
