import torch
import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size,embedding_dim,context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding = nn.Embedding(context_length,embedding_dim)

    def forward(self,x):
        token_embeddings = self.token_embedding(x)
        B,T = x.shape
        positions = torch.arange(T,device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        return x





model = TransformerLanguageModel(10,12,4)

x = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])
print(x.shape)

print(model(x))
print(model(x).shape)