import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class FeedForward(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.ffn = nn.Sequential(
      nn.Linear(args.n_embed, 4*args.n_embed),
      nn.ReLU(),
      nn.Linear(4*args.n_embed, args.n_embed),
      nn.Dropout(args.dropout),
    )
  def forward(self, x):
    return self.ffn(x)

class Block(nn.Module):
  def __init__(self, args) -> None:
    super().__init__( )
    self.head_size = args.n_embed // args.n_heads
    self.sa_head = Mamba(
      # This module uses roughly 3 * expand * d_model^2 parameters
      d_model=args.n_embed, # Model dimension d_model
      d_state=args.d_state,  # SSM state expansion factor
      d_conv=args.d_conv,    # Local convolution width
      expand=args.expand,    # Block expansion factor
  ).to("cuda")
    self.ffn = FeedForward(args)
    self.ln1 = nn.LayerNorm(args.n_embed)
    self.ln2 = nn.LayerNorm(args.n_embed)

  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x

class MambaAudioModel(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.token_embedding_table = nn.Embedding(args.vocab_size,args.n_embed)
    self.position_embedding_table = nn.Embedding(args.block_size,args.n_embed)
    self.lm_head = nn.Linear(args.n_embed,args.vocab_size)
    self.ffn = FeedForward(args)
    self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layers)])
    
    self.device = args.device

  def forward(self, idx, targets=None):
    # idx = idx[:,-block_size:]
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx) # (B,T, C_e)
    pos_emb = self.position_embedding_table(torch.arange(T,device=self.device)) # (T, C_e)
    x = tok_emb + pos_emb # (B,T,Q, C_e)
    x = self.blocks(x) # (B,T,Q, C_e)
    logits = self.lm_head(x) # (B,T,vocab_size)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      logits = logits.view(B,T,C)
    return logits, loss

