import torch
from torch.utils.data import Dataset,DataLoader

class TextDataset(Dataset):
    def __init__(self,token_ids,context_length):
        self.token_ids = torch.tensor(token_ids,dtype=torch.float32)
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):

        x = self.token_ids[index:index+self.context_length]
        y = self.token_ids[index+1:index+1+self.context_length]
        return x,y

token_ids = [
    1, 2, 3, 3, 4,
    5, 6, 4, 7, 3, 8
]
dataset = TextDataset(token_ids,context_length=4)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

# print(dataset)
x,y = dataset[9]
print(x,y)



