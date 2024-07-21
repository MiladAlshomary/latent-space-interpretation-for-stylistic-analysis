import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, PreTrainedModel

from .config import LUARConfig

# Adapted LucidRains impl. of Memory Efficient Attention
# https://github.com/lucidrains/memory-efficient-attention-pytorch


def exists(val):
    return val is not None


def summarize_qkv_chunk(q, k, v, mask):
    """Dot-Product Attention for a chunk of queries, keys, and values."""
    weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)

    if exists(mask):
        # HuggingFace masks have to be added:
        weight += mask

    weight_max = weight.amax(dim=-1, keepdim=True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    weighted_value = torch.einsum("b h i j, b h j d -> b h i d", exp_weight, v)

    return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, "... 1 -> ...")


checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)


def memory_efficient_attention(
    q, k, v, mask=None, q_bucket_size=512, k_bucket_size=1024, eps=1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = (
        checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk
    )

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = (
        mask.split(k_bucket_size, dim=-1) if exists(mask) else ((None,) * len(k_chunks))
    )

    # loop through all chunks and accumulate
    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(
            zip(k_chunks, v_chunks, mask_chunks)
        ):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        exp_weights = torch.stack(exp_weights, dim=-1)
        weighted_values = torch.stack(weighted_values, dim=-1)
        weight_maxes = torch.stack(weight_maxes, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, "... c -> ... 1 c")

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (rearrange(all_weights, "... -> ... 1") + eps)
        out.append(normalized_values)

    return torch.cat(out, dim=-2)


class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need"."""

    def __init__(
        self,
        memory_efficient_attention=False,
        q_bucket_size=512,
        k_bucket_size=1024,
    ):
        super(SelfAttention, self).__init__()
        self.use_memory_efficient_attention = memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(self, k, q, v):

        if self.use_memory_efficient_attention:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=12), (q, k, v)
            )

            out = memory_efficient_attention(
                q,
                k,
                v,
                q_bucket_size=self.q_bucket_size,
                k_bucket_size=self.k_bucket_size,
            )
            out = rearrange(out, "b h n d -> b n (h d)")
            return out
        else:
            d_k = q.size(-1)
            scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
            p_attn = F.softmax(scores, dim=-1)
            return torch.matmul(p_attn, v)


class LUAR(PreTrainedModel):
    """Defines the LUAR model."""

    config_class = LUARConfig

    def __init__(self, config):
        super().__init__(config)
        self.create_transformer()
        self.attn_fn = SelfAttention(
            config.use_memory_efficient_attention,
            config.q_bucket_size,
            config.k_bucket_size,
        )
        self.linear = nn.Linear(self.hidden_size, config.embedding_size)

    def create_transformer(self):
        """Creates the Transformer backbone."""
        self.transformer = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper."""
        input_mask_expanded = repeat(
            attention_mask, "b l -> b l d", d=self.hidden_size
        ).type(token_embeddings.type())
        sum_embeddings = reduce(
            token_embeddings * input_mask_expanded, "b l d -> b d", "sum"
        )
        sum_mask = torch.clamp(
            reduce(input_mask_expanded, "b l d -> b d", "sum"), min=1e-9
        )
        return sum_embeddings / sum_mask

    def get_episode_embeddings(
        self, input_ids, attention_mask, output_attentions=False, document_batch_size=0
    ):
        """Computes the Author Embedding."""
        B, E, _ = attention_mask.shape

        input_ids = rearrange(input_ids, "b e l -> (b e) l")
        attention_mask = rearrange(attention_mask, "b e l -> (b e) l")

        if document_batch_size > 0:
            outputs = {"last_hidden_state": [], "attentions": []}
            for i in range(0, len(input_ids), document_batch_size):
                out = self.transformer(
                    input_ids=input_ids[i : i + document_batch_size],
                    attention_mask=attention_mask[i : i + document_batch_size],
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=output_attentions,
                )
                outputs["last_hidden_state"].append(out["last_hidden_state"])
                if output_attentions:
                    outputs["attentions"].append(out["attentions"])
            outputs["last_hidden_state"] = torch.cat(
                outputs["last_hidden_state"], dim=0
            )
            if output_attentions:
                outputs["attentions"] = tuple(
                    [
                        torch.cat([x[i] for x in outputs["attentions"]], dim=0)
                        for i in range(len(outputs["attentions"][0]))
                    ]
                )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=output_attentions,
            )

        # at this point, we're embedding individual "comments"
        comment_embeddings = self.mean_pooling(
            outputs["last_hidden_state"], attention_mask
        )
        comment_embeddings = rearrange(comment_embeddings, "(b e) l -> b e l", b=B, e=E)

        # aggregate individual comments embeddings into episode embeddings
        episode_embeddings = self.attn_fn(
            comment_embeddings, comment_embeddings, comment_embeddings
        )
        episode_embeddings = reduce(episode_embeddings, "b e l -> b l", "max")

        episode_embeddings = self.linear(episode_embeddings)

        if output_attentions:
            return episode_embeddings, outputs["attentions"]

        return episode_embeddings

    def forward(
        self, input_ids, attention_mask, output_attentions=False, document_batch_size=0
    ):
        """Calculates a fixed-length feature vector for a batch of episode samples."""
        output = self.get_episode_embeddings(
            input_ids, attention_mask, output_attentions, document_batch_size
        )

        return output
