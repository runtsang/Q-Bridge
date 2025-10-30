import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=.1):
        super().__init__()
        if embed_dim % num_heads: raise ValueError
        self.embed_dim=embed_dim; self.num_heads=num_heads; self.d_k=embed_dim//num_heads; self.dropout=nn.Dropout(dropout)
    def separate_heads(self,x):
        return x.view(x.size(0),-1,self.num_heads,self.d_k).transpose(1,2)
    def attention(self,q,k,v,mask=None):
        scores=torch.bmm(q,k.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None: scores=scores.masked_fill(mask.unsqueeze(1)==0,-1e9)
        return F.softmax(scores,dim=-1)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self,embed_dim,num_heads,dropout=.1):
        super().__init__(embed_dim,num_heads,dropout)
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout,batch_first=True)
    def forward(self,x,mask=None):
        out,_=self.attn(x,x,x,key_padding_mask=mask)
        return out
    def combine_heads(self,x):
        return x.transpose(1,2).contiguous().view(x.size(0),-1,self.embed_dim)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self,embed_dim,num_heads,dropout=.1):
        super().__init__(embed_dim,num_heads,dropout)
    def forward(self,x,mask=None):
        return MultiHeadAttentionClassical(self.embed_dim,self.num_heads,self.dropout.p)(x,mask)

class FeedForwardBase(nn.Module):
    def __init__(self,embed_dim,ffn_dim,dropout=.1):
        super().__init__()
        self.embed_dim=embed_dim; self.ffn_dim=ffn_dim; self.dropout=nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    def __init__(self,embed_dim,ffn_dim,dropout=.1):
        super().__init__(embed_dim,ffn_dim,dropout)
        self.l1=nn.Linear(embed_dim,ffn_dim)
        self.l2=nn.Linear(ffn_dim,embed_dim)
    def forward(self,x): return self.l2(self.dropout(F.relu(self.l1(x))))

class FeedForwardQuantum(FeedForwardBase):
    def __init__(self,embed_dim,ffn_dim,dropout=.1):
        super().__init__(embed_dim,ffn_dim,dropout)
    def forward(self,x): return FeedForwardClassical(self.embed_dim,self.ffn_dim,self.dropout.p)(x)

class TransformerBlockBase(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=.1):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        self.dropout=nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self,embed_dim,num_heads,ffn_dim,dropout=.1):
        super().__init__(embed_dim,num_heads,dropout)
        self.attn=MultiHeadAttentionClassical(embed_dim,num_heads,dropout)
        self.ffn=FeedForwardClassical(embed_dim,ffn_dim,dropout)
    def forward(self,x):
        a=self.attn(x); x=self.norm1(x+self.dropout(a))
        f=self.ffn(x); return self.norm2(x+self.dropout(f))

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,embed_dim,num_heads,ffn_dim,dropout=.1):
        super().__init__(embed_dim,num_heads,dropout)
        self.attn=MultiHeadAttentionQuantum(embed_dim,num_heads,dropout)
        self.ffn=FeedForwardQuantum(embed_dim,ffn_dim,dropout)
    def forward(self,x):
        a=self.attn(x); x=self.norm1(x+self.dropout(a))
        f=self.ffn(x); return self.norm2(x+self.dropout(f))

class PositionalEncoder(nn.Module):
    def __init__(self,embed_dim,max_len=5000):
        super().__init__()
        pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,embed_dim,2)*(-math.log(10000.0)/embed_dim))
        pe=torch.zeros(max_len,embed_dim)
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class HybridAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=.1):
        super().__init__()
        self.cl=MultiHeadAttentionClassical(embed_dim,num_heads,dropout)
        self.qm=MultiHeadAttentionQuantum(embed_dim,num_heads,dropout)
        self.gate=nn.Parameter(torch.full((num_heads,),.5))
    def forward(self,x,mask=None):
        c=self.cl(x,mask); q=self.qm(x,mask)
        c_head=self.cl.separate_heads(c); q_head=self.qm.separate_heads(q)
        g=torch.sigmoid(self.gate).view(1,-1,1,1)
        comb=g*q_head+(1-g)*c_head
        return self.cl.combine_heads(comb)

class HybridFeedForward(nn.Module):
    def __init__(self,embed_dim,ffn_dim,dropout=.1):
        super().__init__()
        self.cl=FeedForwardClassical(embed_dim,ffn_dim,dropout)
        self.qm=FeedForwardQuantum(embed_dim,ffn_dim,dropout)
        self.gate=nn.Parameter(torch.tensor(.5))
    def forward(self,x):
        c=self.cl(x); q=self.qm(x)
        g=torch.sigmoid(self.gate)
        return g*q+(1-g)*c

class HybridTransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_dim,dropout=.1,use_quantum=True):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim); self.norm2=nn.LayerNorm(embed_dim); self.dropout=nn.Dropout(dropout)
        if use_quantum:
            self.attn=HybridAttention(embed_dim,num_heads,dropout)
            self.ffn=HybridFeedForward(embed_dim,ffn_dim,dropout)
        else:
            self.attn=MultiHeadAttentionClassical(embed_dim,num_heads,dropout)
            self.ffn=FeedForwardClassical(embed_dim,ffn_dim,dropout)
    def forward(self,x):
        a=self.attn(x); x=self.norm1(x+self.dropout(a))
        f=self.ffn(x); return self.norm2(x+self.dropout(f))

class HybridTextClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_heads,num_blocks,ffn_dim,num_classes,dropout=.1,use_quantum=True):
        super().__init__()
        self.tok=nn.Embedding(vocab_size,embed_dim)
        self.pos=PositionalEncoder(embed_dim)
        self.blocks=nn.Sequential(*[HybridTransformerBlock(embed_dim,num_heads,ffn_dim,dropout,use_quantum) for _ in range(num_blocks)])
        self.dropout=nn.Dropout(dropout)
        self.cls=nn.Linear(embed_dim,num_classes if num_classes>2 else 1)
    def forward(self,x):
        t=self.tok(x); x=self.pos(t); x=self.blocks(x); x=x.mean(dim=1); x=self.dropout(x); return self.cls(x)

__all__=["MultiHeadAttentionBase","MultiHeadAttentionClassical","MultiHeadAttentionQuantum","FeedForwardBase","FeedForwardClassical","FeedForwardQuantum","TransformerBlockBase","TransformerBlockClassical","TransformerBlockQuantum","PositionalEncoder","HybridAttention","HybridFeedForward","HybridTransformerBlock","HybridTextClassifier"]
