import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        assert self.head_dim * num_heads == input_size, "input_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.out_linear = nn.Linear(input_size, input_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        batch_size = 1
        seq_length = x.size(0)
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        
        # Output projection
        output = self.out_linear(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output.squeeze(0), attn_weights.squeeze(0)

class TemporalAttention(nn.Module):
    def __init__(self, input_size, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.conv1d = nn.Conv1d(input_size, input_size, window_size, padding=window_size//2)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # Apply temporal convolution
        x_t = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
        conv_out = self.conv1d(x_t)
        conv_out = conv_out.transpose(1, 2).squeeze(0)  # [T, C]
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + conv_out)
        return output

class EnhancedAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(input_size, num_heads, dropout)
        self.temporal_attn = TemporalAttention(input_size)
        self.fusion = nn.Linear(input_size * 2, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head self-attention
        mha_out, attn_weights = self.mha(x)
        
        # Temporal attention
        temp_out = self.temporal_attn(x)
        
        # Fusion of both attention mechanisms
        combined = torch.cat([mha_out, temp_out], dim=-1)
        output = self.fusion(combined)
        output = self.dropout(output)
        
        return output, attn_weights 