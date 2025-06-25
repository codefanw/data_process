import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: 输入特征维度（d_model）
            num_heads: 注意力头的总数
            num_groups: Key/Value的分组数（必须能被num_heads整除）
            dropout: Dropout概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        assert num_heads % num_groups == 0, "num_heads必须能被num_groups整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups  # 每组包含的注意力头数

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Query投影（每个头独立）
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups)  # Key投影（按分组共享）
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups)  # Value投影（按分组共享）
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            输出序列 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影计算Q/K/V
        q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch, seq_len, head_dim * num_groups]
        v = self.v_proj(x)  # [batch, seq_len, head_dim * num_groups]
        
        # 2. 调整形状以分组处理
        # Query: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Key/Value: [batch, seq_len, num_groups, head_dim]
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim)
        
        # 3. 广播Key/Value到每个组内的所有头
        # Key: [batch, seq_len, num_heads, head_dim]
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)
        
        # 4. 计算注意力分数
        # Q/K转置后矩阵乘法: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ​**​ 0.5)
        
        # 5. 应用掩码（如需要）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 6. Softmax和Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. 加权求和Value
        # 输出: [batch, seq_len, num_heads, head_dim]
        output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # 8. 合并多头并投影
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output
