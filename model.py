import torch
import spacy
import torch.nn as nn
import torch.optim as optim
from torchtext import data

TEXT = data.Field(tokenize="spacy", batch_first=True, include_lengths=True, lower=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [("review", TEXT), (None, None), ("sentiment", LABEL)]

# loading data
training_data = data.TabularDataset(
    path="IMDB_Dataset.csv", format="csv", fields=fields, skip_header=True
)

train_data, valid_data = training_data.split(split_ratio=0.75)

TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# number of unique tokens in text:
print("Size of TEXT vocabulary: ", len(TEXT.vocab))

# number of unique tokens in label:
print("Size of LABEL vocabulary: ", len(LABEL.vocab))

# Word dictionary:
# print(TEXT.vocab.stoi)

# set batch size
BATCH_SIZE = 64

device = torch.device("cpu")

# Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.review),
    sort_within_batch=True,
    device=device,
)


class Classifier(nn.Module):
    # define all the layers used in the module:
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
    ):
        # Constructor:
        super().__init__()

        # Embedding layer:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        # Dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):

        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs


# Define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

# Instantiate the model
model = Classifier(
    size_of_vocab,
    embedding_dim,
    num_hidden_nodes,
    num_output_nodes,
    num_layers,
    bidirectional=True,
    dropout=dropout,
)

# Architecture
print(model)


# No. of trainable parameters:
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

# Initialize the pretrained embedding_dim
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

# Define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# Define metric
def binary_accuracy(preds, y):
    # Round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_accuracy = 0

    # set the model in the training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()
        # get text and number of words
        text, text_lengths = batch.review
        # convert to 1-dimensional tensor
        predictions = model(text, text_lengths).squeeze()
        # compute the loss
        loss = criterion(predictions, batch.sentiment)
        # compute the binary accuracy
        accuracy = binary_accuracy(predictions, batch.sentiment)
        loss.backward()
        # update weights
        optimizer.step()
        # loss and accuracy
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # get text and number of words
            text, text_lengths = batch.text
            # convert to 1-dimensional tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.sentiment)
            accuracy = binary_accuracy(predictions, batch.sentiment)
            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


N_EPOCHS = 1
best_valid_loss = float("inf")

for epoch in range(N_EPOCHS):
    # train the model
    train_loss, train_accuracy = train(model, train_iterator, optimizer, criterion)
    # evaluate the model
    valid_loss, valid_accuracy = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "saved_weights.pt")

    print(f"\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Accuracy: {valid_accuracy*100:.2f}%")


# load weights
path = "/content/saved_weights.pt"
model.load_state_dict(torch.load(path))
model.eval()

# inference
nlp = spacy.load("en")


def predict(model, sentence):
    tokenized = [token.text for token in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1).T
    length_tensor = torch.LongTensor(length)
    prediction = model(tensor, length_tensor)
    return prediction.item()


print("----------DONE!----------")
