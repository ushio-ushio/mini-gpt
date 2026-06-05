import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,max_seq_len,d_model,n_heads):
        super(SelfAttention, self).__init__()
        assert d_model%n_heads==0
        self.max_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k=d_model//n_heads

        self.Q=nn.Linear(d_model,d_model)
        self.K=nn.Linear(d_model,d_model)
        self.V=nn.Linear(d_model,d_model)
        self.fc=nn.Linear(d_model,d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("bias", mask)

    def forward(self, x):
        B, T, C = x.shape

        q = self.Q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = self.K(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.V(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        attn=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.d_k)
        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        attn=torch.softmax(attn,dim=-1)

        out=torch.matmul(attn,v)
        out=out.transpose(1,2).contiguous().view(B,T,C)
        out=self.fc(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,max_seq_len):
        super(DecoderBlock, self).__init__()
        self.attn=SelfAttention(max_seq_len,d_model=d_model,n_heads=n_heads)
        self.ff=FeedForward(d_model,d_ff)
        self.n1=nn.LayerNorm(d_model)
        self.n2=nn.LayerNorm(d_model)

    def forward(self,x):
        x=x+self.attn(self.n1(x))
        x=x+self.ff(self.n2(x))
        return x
    
class mini_gpt(nn.Module):
    def __init__(self,vocab_size,d_model,max_seq_len,n_heads,d_ff,num_layers):
        super(mini_gpt, self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.position=nn.Embedding(max_seq_len,d_model)
        
        self.blocks=nn.Sequential(
            *[DecoderBlock(d_model,n_heads,d_ff,max_seq_len) for _ in range(num_layers)]
        )
        self.fn=nn.LayerNorm(d_model)
        self.fc_out=nn.Linear(d_model,vocab_size)

    def forward(self,x,targets=None):
        B,T=x.size()
        pos=torch.arange(0,T,dtype=torch.long,device=x.device)
        x=self.embedding(x)
        x=x+self.position(pos)
        x=self.blocks(x)
        x=self.fn(x)
        
        logits = self.fc_out(x)

        loss=None
        if targets is not None:
            B,T,C=logits.size()
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss

