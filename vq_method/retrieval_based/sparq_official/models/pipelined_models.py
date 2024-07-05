# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Any, Optional, Tuple, Union, cast

import torch
from torch import nn
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXLayer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from .. import utility


class PipelinedLlamaDecoderLayer(LlamaDecoderLayer):  # type:ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Any,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        layer_device = self.input_layernorm.weight.device
        if hidden_states.device != layer_device:
            hidden_states = hidden_states.to(layer_device)
        if attention_mask.device != layer_device:
            attention_mask = attention_mask.to(layer_device)
        if position_ids.device != layer_device:
            position_ids = position_ids.to(layer_device)
        return super().forward(  # type:ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )


class PipelinedGPTNeoXLayer(GPTNeoXLayer):  # type:ignore[misc]
    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Any:
        layer_device = self.input_layernorm.weight.device
        if hidden_states is not None and hidden_states.device != layer_device:
            hidden_states = hidden_states.to(layer_device)
        if attention_mask is not None and attention_mask.device != layer_device:
            attention_mask = attention_mask.to(layer_device)
        if position_ids is not None and position_ids.device != layer_device:
            position_ids = cast(torch.LongTensor, position_ids.to(layer_device))
        if head_mask is not None and head_mask.device != layer_device:
            head_mask = head_mask.to(layer_device)
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            layer_past=layer_past,
            output_attentions=output_attentions,
        )


def pipeline_model(
    model: Union[GPTNeoXForCausalLM, LlamaForCausalLM], num_stages: int
) -> Union[GPTNeoXForCausalLM, LlamaForCausalLM]:
    """Convert a GPTNeoX or Llama model to use pipelining."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXLayer):
            return PipelinedGPTNeoXLayer(model.config)
        if isinstance(m, LlamaDecoderLayer):
            return PipelinedLlamaDecoderLayer(model.config)

    model = utility.convert_module(model, _replace)
    num_hidden_layers = cast(int, model.config.num_hidden_layers)
    partition_len = ((num_hidden_layers - 1) // num_stages) + 1
    gpu_allocation = [i // partition_len for i in range(num_hidden_layers)]

    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens.cuda(0)
        trunk = cast(nn.Module, model.model.layers)
        for idx, (_, layer) in enumerate(trunk.named_children()):
            layer.cuda(gpu_allocation[idx])
        model.model.norm.cuda(num_stages - 1)
        model.lm_head.cuda(num_stages - 1)
        return model
    elif isinstance(model, GPTNeoXForCausalLM):
        model.gpt_neox.embed_in.cuda(0)
        model.gpt_neox.emb_dropout.cuda(0)
        trunk = cast(nn.Module, model.gpt_neox.layers)
        for idx, (_, layer) in enumerate(trunk.named_children()):
            layer.cuda(gpu_allocation[idx])
        model.gpt_neox.final_layer_norm.cuda(num_stages - 1)
        model.embed_out.cuda(num_stages - 1)
        return model
    else:
        raise ValueError(f"Model class {type(model)} isn't supported")
