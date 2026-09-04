import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("100_Unique_QA_Dataset.csv")


# tokenize the questions and answers
def tokenize(text):
    text = text.lower()
    text = text.replace("?", "")
    text = text.replace("'", "")
    return text.split()


# vocab
vocab = {"<UNK>": 0}


def build_vocab(row: pd.Series) -> pd.Series:
    tokenized_question = tokenize(row["question"])
    tokenized_answer = tokenize(row["answer"])

    merged_tokens = tokenized_question + tokenized_answer

    for token in merged_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return row


df.apply(build_vocab, axis=1)


# convert words to numerical indices
def text_to_indices(text, vocab):
    indexed_text = []

    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab["<UNK>"])

    return indexed_text


class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        question = self.df.iloc[idx]["question"]
        answer = self.df.iloc[idx]["answer"]

        question_indices = text_to_indices(question, self.vocab)
        answer_indices = text_to_indices(answer, self.vocab)

        return torch.tensor(question_indices), torch.tensor(answer_indices)


dataset = QADataset(df, vocab)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        embedded_question = self.embedding(x)
        hidden_states, final = self.rnn(embedded_question)
        output = self.fc(final.squeeze(0))
        return output


x = nn.Embedding(324, embedding_dim=50)
y = nn.RNN(50, 64, batch_first=True)
z = nn.Linear(64, 324)

a = dataset[0][0].reshape(1, 6)
print("shape of a:", a.shape)
b = x(a)
print("shape of b:", b.shape)
c, d = y(b)
print("shape of c:", c.shape)
print("shape of d:", d.shape)

e = z(d.squeeze(0))

print("shape of e:", e.shape)

learning_rate = 0.001
epochs = 20

model = SimpleRNN(
    vocab_size=len(vocab), embedding_dim=50, hidden_dim=64, output_dim=len(vocab)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for question, answer in dataloader:

        optimizer.zero_grad()

        # forward pass
        output = model(question)

        # loss calculation -> output shape: (batch_size, vocab_size), answer shape: (batch_size, seq_len)
        loss = criterion(output, answer[0])

        # gradient calculation
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


def predict(model, question, vocab, threshold=0.5):
    model.eval()
    with torch.no_grad():
        question_indices = text_to_indices(question, vocab)
        question_tensor = torch.tensor(question_indices).unsqueeze(
            0
        )  # Add batch dimension
        output = model(question_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        value, predicted_index = torch.max(probabilities, dim=1)

        if value < threshold:
            return "I don't know the answer to that question."
        return list(vocab.keys())[list(vocab.values()).index(predicted_index.item())]


print(predict(model, "What is the largest planet in our solar system?", vocab))
print(predict(model, "What is RNN?", vocab))
