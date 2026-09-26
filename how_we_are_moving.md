## step-1: downloading dataset
first we took dataset of FineWeb-Edu
since it is very large we used stream the dataset and saved 500MB to our local. 

## step-2: tokenization

once we got the dataset. 
we started BPE. and got token_ids. we have initiated our Vocab_size to 8000.
now we have token_ids and their respective words.

## step-3: processing the dataset.

now we need to process our dataset and match our data to token_ids.

what we are essintally doing here is 
suppose our dataset has 3 documents. 
doc-1
[78,238,832,....]
same for doc-2
same for doc-3 some token_ids. 
now we need to add <endoftext> special token to let neural network know that this is the end of text. so we first add that to each doc.
suppose 
then <endoftext> token_id is 0
doc-1
[78,238,832,....0] we have added 0 in the end.
same for doc-2
same for doc-3 some token_ids. 

now we will combine all of those. 
[78,283,832,......0,.....0,......] -> like this . we will combine all data token_ids and store them in .bin file. 
why we are combining?
because we will eventually do next token prediction. 
for that we know we take a sequence_length and predict next token like 
X
[91, 42, 18, 73, 22, 65, 11, 39]
Y
[42, 18, 73, 22, 65, 11, 39, 84]

since our vocab_size is 8000. we can go with uint16 
like 
uint8 0-255
uint16 0-65,535
uint32 0 → 4,294,967,295

uint16 means -> 16 bites -> 2 bytes 
suppose we ended up with 100,000,000 tokens 
then we get 100M x 2 bytes = 200MB

- general doubt is 
what if a training window happens to cross the boundary:
... last words of Document 1
<|endoftext|>
first words of Document 2 ...

That means GPT could see two unrelated documents inside the same context window.
That's actually okay for this type of pretraining because the <|endoftext|> token explicitly marks the boundary.
So we do not need to pad every document to 256 tokens.

once this is done. 
we get .bin files 
like train.bin and validation.bin files 

### step-4: Dataset + DataLoader

generally when we have tokens.
we go batches right like suppose we have 10 tokens 
and we have context length of 4 
no of batches we takes = 10-4 = 6 
[1,2,3,4][2,3,4,5]....[7,8,9,10] -> total 6. 
but these are overlapping intervals.

since our dataset is huge, it is better to go with non-overlapping intervals..
10/4 ~ 3
[1,2,3,4] [5,6,7,8] [9,10]

now we have 
132,087,284 tokens and context length is 256 -> 5,15,965.953125 ~ 516k training sequences.
~515,966 samples ÷ 8 samples per batch ≈ 64,496 batches per epoch

we use both DataSet and DataLoader. 
we use batch_size = 8

so our input will be (8,256)


### now we freeze the data pipeline. we wont run these files again we just use train.bin and validation.bin .. unless we want to re-tokenize the things.

### now we start our core GPT Architecture.

# token embeddings. 
we decided embeddings size as 384. 

so we will create a random tensor of shape (8000,384)
this means for every token we are initating a embedding.

and when we pass token to our gpt we wont pass it as token_id we will pass it as embedding.

so our x will be (8,256) -> (8,256,384)

# positional embeddings. 

see when we are doing embeddings 
suppose we have a 2 texts
"The dog bites man"
vs
"Man bites the dog"

we have same words -> same tokens 
but the meaning is entirely different 
so we will include positional embeddings so that our neural network sees it differently 

so we will create another tensor/matrix of shape (256,384)
