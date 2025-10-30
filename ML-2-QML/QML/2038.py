import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

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
    def downstream(self,query,key,value,batch_size,mask=None):
        q=self.separate_heads(query); k=self.separate_heads(key); v=self.separate_heads(value)
        out,self.attn_weights=self.attention(q,k,v,mask)
        return out.transpose(1,2).contiguous().view(batch_size,-1,self.embed_dim)

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
    class _QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires=8
            self.encoder=tq.GeneralEncoder([{"input_idx":[i],"func":"rx","wires":[i]} for i in range(8)])
            self.params=nn.ModuleList([tq.RX(has_params=True,trainable=True) for _ in range(self.n_wires)])
            self.measure=tq.MeasureAll(tq.PauliZ)
        def forward(self,x,qdev):
            self.encoder(qdev,x)
            for w,g in enumerate(self.params): g(qdev,wires=w)
            for w in range(self.n_wires-1): tqf.cnot(qdev,wires=[w,w+1])
            tqf.cnot(qdev,wires=[self.n_wires-1,0])
            return self.measure(qdev)
    def __init__(self,embed_dim,num_heads,dropout=.1,mask=None,use_bias=False,q_device=None):
        super().__init__(embed_dim,num_heads,dropout)
        self.q_layer=self._QLayer()
        self.q_device=q_device
        self.combine_heads=nn.Linear(embed_dim,embed_dim,bias=use_bias)
    def forward(self,x,mask=None):
        b,_,e=x.size()
        if e!=self.embed_dim: raise ValueError
        k=self._quantum_heads(x); q=self._quantum_heads(x); v=self._quantum_heads(x)
        out=self.downstream(q,k,v,b,mask)
        return self.combine_heads(out)
    def _quantum_heads(self,x):
        proj=[]
        for token in x.unbind(dim=1):
            token=token.view(token.size(0),self.num_heads,-1)
            heads=[]
            for head in token.unbind(dim=1):
                qdev=self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires,bsz=head.size(0),device=head.device)
                heads.append(self.q_layer(head,qdev))
            proj.append(torch.stack(heads,dim=1))
        return torch.stack(proj,dim=1)

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
    class _QLayer(tq.QuantumModule):
        def __init__(self,n_qubits):
            super().__init__()
            self.n_wires=n_qubits
            self.encoder=tq.GeneralEncoder([{"input_idx":[i],"func":"rx","wires":[i]} for i in range(n_qubits)])
            self.params=nn.ModuleList([tq.RY(has_params=True,trainable=True) for _ in range(n_qubits)])
            self.measure=tq.MeasureAll(tq.PauliZ)
        def forward(self,x,qdev):
            self.encoder(qdev,x)
            for w,g in enumerate(self.params): g(qdev,wires=w)
            return self.measure(qdev)
    def __init__(self,embed_dim,ffn_dim,n_qubits,dropout=.1):
        super().__init__(embed_dim,ffn_dim,dropout)
        self.q_layer=self._QLayer(n_qubits)
        self.q_device=tq.QuantumDevice(n_wires=n_qubits)
        self.l1=nn.Linear(n_qubits,ffn_dim)
        self.l2=nn.Linear(ffn_dim,embed_dim)
    def forward(self,x):
        outs=[]
        for token in x.unbind(dim=1):
            qdev=self.q_device.copy(bsz=token.size(0),device=token.device)
            outs.append(self.q_layer(token,qdev))
        out=torch.stack(outs,dim=1)
        out=self.l1(self.dropout(out))
        return self.l2(F.relu(out))

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
    def __init__(self,embed_dim,num_heads,ffn_dim,n_qubits_transformer=0,n_qubits_ffn=0,n_qlayers=1,q_device=None,dropout=.1):
        super().__init__(embed_dim,num_heads,dropout)
        self.attn=MultiHeadAttentionQuantum(embed_dim,num_heads,dropout,q_device=q_device)
        self.ffn=FeedForwardQuantum(embed_dim,ffn_dim,n_qubits_ffn,dropout) if n_qubits_ffn>0 else FeedForwardClassical(embed_dim,ffn_dim,dropout)
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
