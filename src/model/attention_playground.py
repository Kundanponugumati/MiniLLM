import torch
import torch.nn as nn


torch.manual_seed(42)

T = 4
C = 6
HEAD_DIM = 3


x = torch.randn(T, C)

print("X:")
print(x)

print("\nX shape:")
print(x.shape)

W_query = nn.Linear(
    C,
    HEAD_DIM,
    bias=False
)

W_key = nn.Linear(
    C,
    HEAD_DIM,
    bias=False
)

W_value = nn.Linear(
    C,
    HEAD_DIM,
    bias=False
)

Q = W_query(x)
K = W_key(x)
V = W_value(x)


print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


print("\nShapes:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

scores = Q @ K.T
print("\nAttention scores:")
print(scores)

print("\nScores shape:")
print(scores.shape)

from math import sqrt

scores = scores / sqrt(HEAD_DIM)

mask = torch.triu(
    torch.ones(T, T),
    diagonal=1
).bool()


print("\nCausal mask:")
print(mask)

scores = scores.masked_fill(
    mask,
    float("-inf")
)
print("\nMasked scores:")
print(scores)

attention_weights = torch.softmax(
    scores,
    dim=-1
)


print("\nAttention weights:")
print(attention_weights)


print("\nRow sums:")
print(attention_weights.sum(dim=-1))

context = attention_weights @ V
print("\nContext vectors:")
print(context)

print("\nContext shape:")
print(context.shape)