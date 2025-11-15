"""Multimodal encoder: CXR (ViT) + Vitals (1D-CNN + Transformer) + Notes (BioLinkBERT)

Produces a fused representation via cross-attention (Q=text, K/V=image+vitals)
and exposes attention maps for visualization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "microsoft/BioLinkBERT-base",
        vit_hidden: int = 768,
        vitals_hidden: int = 128,
        attn_dim: int = 768,
        n_heads: int = 8,
    ):
        super().__init__()
        # Image encoder (ViT)
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # Text encoder (BioLinkBERT or other transformer)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Vitals encoder: 1D-CNN to reduce noise + Transformer encoder for temporal context
        # Input vitals shape: (batch, seq_len, n_features)
        self.vitals_conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=vitals_hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(vitals_hidden),
        )

        # Transformer encoder for vitals (batch_first=True)
        vit_layer = nn.TransformerEncoderLayer(d_model=vitals_hidden, nhead=4, batch_first=True)
        self.vitals_transformer = nn.TransformerEncoder(vit_layer, num_layers=2)

        # project vitals to attention dimension
        self.vitals_proj = nn.Linear(vitals_hidden, attn_dim)

        # project image features to attn_dim if needed (ViT hidden size usually matches)
        if vit_hidden != attn_dim:
            self.image_proj = nn.Linear(vit_hidden, attn_dim)
        else:
            self.image_proj = nn.Identity()

        # project text pooled output to attn_dim (if different)
        self.text_proj = nn.Identity() if attn_dim == 768 else nn.Linear(768, attn_dim)

        # Cross-attention: queries from text, keys/values from image+vitals
        self.cross_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, batch_first=True)

        self.layernorm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(0.1)

        # store last attention weights for visualization
        self._last_attn: torch.Tensor | None = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        vitals: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: (batch, 3, H, W) preprocessed for ViT
            vitals: (batch, seq_len, n_features) e.g., (B,1440,5)
            input_ids, attention_mask: for the text encoder

        Returns:
            fused: (batch, attn_dim) fused representation
            attn_weights: (batch, query_len=1, key_len) attention maps
        """
        device = pixel_values.device

        # Image encoding
        img_out = self.vit(pixel_values=pixel_values)
        # ViTModel returns `pooler_output` for pooled representation; fallback to cls token
        img_feats = getattr(img_out, "pooler_output", None)
        if img_feats is None:
            # last_hidden_state: (batch, seq_len, hidden)
            img_feats = img_out.last_hidden_state[:, 0, :]
        img_feats = self.image_proj(img_feats)  # (batch, attn_dim)

        # Text encoding (use pooled output if available, else mean pool)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_pooled = getattr(txt_out, "pooler_output", None)
        if txt_pooled is None:
            # mean pool over tokens
            last = txt_out.last_hidden_state  # (batch, seq_len, hidden)
            mask = attention_mask.unsqueeze(-1).to(last.dtype) if attention_mask is not None else 1.0
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-5)
            txt_pooled = summed / denom
        txt_emb = self.text_proj(txt_pooled)  # (batch, attn_dim)

        # Vitals encoding
        # vitals: (batch, seq_len, features) -> conv expects (batch, features, seq_len)
        vitals_in = vitals.permute(0, 2, 1)
        vit_feats = self.vitals_conv(vitals_in)  # (batch, vitals_hidden, seq_len)
        vit_feats = vit_feats.permute(0, 2, 1)  # (batch, seq_len, vitals_hidden)
        vit_feats = self.vitals_transformer(vit_feats)  # (batch, seq_len, vitals_hidden)
        vit_feats = self.vitals_proj(vit_feats)  # (batch, seq_len, attn_dim)

        # Build key/value sequence: prepend image as first token
        img_token = img_feats.unsqueeze(1)  # (batch, 1, attn_dim)
        kv = torch.cat([img_token, vit_feats], dim=1)  # (batch, kv_len, attn_dim)

        # Query is text embedding; expand as sequence length 1
        q = txt_emb.unsqueeze(1)  # (batch, 1, attn_dim)

        # Cross-attention: query=q, key=kv, value=kv
        attn_output, attn_weights = self.cross_attn(q, kv, kv, need_weights=True)
        # attn_output: (batch, 1, attn_dim)
        fused = attn_output.squeeze(1)
        fused = self.layernorm(self.dropout(fused))

        # attn_weights: (batch, query_len=1, key_len)
        self._last_attn = attn_weights.detach() if attn_weights is not None else None
        return fused, attn_weights

    def get_last_attention(self) -> torch.Tensor | None:
        """Return the most recent attention weights (or None)."""
        return self._last_attn
