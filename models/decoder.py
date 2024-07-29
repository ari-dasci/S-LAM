import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerDecoder
import pytorch_lightning as pl

from attn import AttentionLayer
from embed import PositionalEncoding
from base_module import Transformer_BaseModule
import pickle
import os

    
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer class for the Transformer model.

    This module represents a single layer of the Transformer decoder. It includes multi-head self-attention,
    multi-head cross-attention, feedforward neural networks, and residual connections with layer normalization.

    Parameters
    ----------
    d_model : int
        The dimensionality of the model, representing the number of features or channels in
        the input and output embeddings.
    nhead : int
        The number of attention heads in the multi-head self-attention and cross-attention mechanisms.
    dim_feedforward : int, optional
        The dimensionality of the feedforward neural networks, by default 16.
    dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.

    Attributes
    ----------
    self_attn : nn.MultiheadAttention
        Multi-head self-attention mechanism.
    multihead_attn : nn.MultiheadAttention
        Multi-head cross-attention mechanism.
    linear1 : nn.Linear
        Linear transformation for the feedforward neural network.
    dropout : nn.Dropout
        Dropout layer applied to the input and output of the attention mechanisms and feedforward network.
    linear2 : nn.Linear
        Linear transformation for the second part of the feedforward network.
    dropout1 : nn.Dropout
        Dropout layer applied to the output of the self-attention mechanism.
    dropout2 : nn.Dropout
        Dropout layer applied to the output of the cross-attention mechanism.
    dropout3 : nn.Dropout
        Dropout layer applied to the output of the second part of the feedforward network.
    activation : nn.LeakyReLU
        Activation function used in the feedforward network.

    Methods
    -------
    forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
            tgt_is_causal=None, memory_is_causal=None)
        Forward pass through the Transformer Decoder Layer.

    Examples
    --------
    Create a TransformerDecoderLayer instance:

    >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
    >>> target_data = torch.rand(32, 100, 512)
    >>> memory_data = torch.rand(32, 100, 512)
    >>> output_data = decoder_layer(target_data, memory_data)

    Notes
    -----
    The Transformer Decoder Layer is a building block in the Transformer model, responsible for
    processing the target sequence and capturing dependencies between tokens.

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).

    """
    def __init__(self, d_model, nhead, attn, attn_params, w_size, w_size_out=None, d_model_out=None, dim_feedforward=None, dropout=0,
                l1=nn.Linear, l1_args={"dtype":torch.float64}, 
                l2=nn.Linear, l2_args={"dtype":torch.float64}):
        super(TransformerDecoderLayer, self).__init__()
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model
        self.attn = attn(**attn_params)
        self.self_attn = AttentionLayer(self.attn, w_size, d_model, nhead)

        self.m_attn = attn(**attn_params)
        self.multihead_attn = AttentionLayer(self.attn, w_size, d_model, nhead)
        #self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, dtype=torch.float64)
        self.l1 = l1(d_model, self.dim_feedforward, **l1_args)
        #self.linear1 = nn.Linear(d_model, self.dim_feedforward, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.l2 = l2(self.dim_feedforward, d_model, **l2_args)
        #self.linear2 = nn.Linear(self.dim_feedforward, d_model, dtype=torch.float64)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)
        self.normalization1 = nn.LayerNorm(d_model)
        self.normalization2 = nn.LayerNorm(d_model)

        """ self.projection = None
        if w_size_out is not None and d_model_out is not None:
            self.projection = nn.Linear(d_model*w_size, d_model_out*w_size_out, dtype=torch.float64)
            self.w_size_out = w_size_out
            self.d_model_out = d_model_out """
        self.w_size_out = w_size_out
        self.d_model_out = d_model_out
        self.d_model = d_model
        self.w_size = w_size
        self.reshape = self.w_size_out is not None and self.d_model_out is not None
        self.proj1, self.proj2 = None, None
        #self.cont=0
        if self.reshape:
            self.proj1 = nn.Linear(self.d_model, self.d_model_out, dtype=torch.float64)
            self.proj2 = nn.Linear(self.w_size, self.w_size_out, dtype=torch.float64)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=None):
        """
        Apply forward pass through the Transformer Decoder Layer.

        Parameters
        ----------
        tgt : torch.Tensor
            The input tensor representing the target sequence.
        memory : torch.Tensor
            The input tensor representing the source sequence (memory).
        tgt_mask : torch.Tensor, optional
            Mask to prevent attention to certain positions in the target sequence, by default None.
        memory_mask : torch.Tensor, optional
            Mask to prevent attention to certain positions in the source sequence, by default None.
        tgt_key_padding_mask : torch.Tensor, optional
            Mask to prevent attention to padding positions in the target sequence, by default None.
        memory_key_padding_mask : torch.Tensor, optional
            Mask to prevent attention to padding positions in the source sequence, by default None.
        tgt_is_causal : bool, optional
            A flag to control the causality of the self-attention in the target sequence, by default None.
        memory_is_causal : bool, optional
            A flag to control the causality of the cross-attention in the source sequence, by default None.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the layer.

        """
        tgt2, attn_w1 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, attn_w2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.normalization1(tgt)
        residual_tgt = tgt
        tgt2 = self.l2(self.dropout(self.activation(self.l1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # Add residual connection and layer normalization
        tgt = tgt + residual_tgt
        tgt = self.normalization2(tgt)
        
        if self.reshape:
            tgt = self.proj1(tgt)
            # permute to have the window size as the last dimension
            tgt = tgt.permute(0, 2, 1)
            tgt = self.proj2(tgt)
            # permute back to the original shape
            tgt = tgt.permute(0, 2, 1)
        return tgt