class CharacterTokenizer():
    def __init__(self,text):
        chars = sorted(list(set(text)))
        self.stoi = {chr:i for i,chr in enumerate(chars)}
        self.itos = {i:chr for i,chr in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self,text):
        # ids = []
        # for chr in text:
        #     ids.append(self.stoi[chr])
        # return ids
        return [self.stoi[chr] for chr in text]

    def decode(self,ids):
        return "".join([self.itos[id] for id in ids])


text = "hello"
tokenizer = CharacterTokenizer(text)
print(tokenizer.vocab_size)
print(tokenizer.itos)
encoded = tokenizer.encode("hello")
print(encoded)
print(tokenizer.decode(encoded))
