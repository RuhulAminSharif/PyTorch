import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk

# with open("ptbdataset/ptb.train.txt", "r") as file:
#     document = file.read()

# print(document)
document = """About the Program
What is the course fee for  Data Science Mentorship Program (DSMP 2023)
The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
What is the total duration of the course?
The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
What is the syllabus of the mentorship program?
We will be covering the following modules:
Python Fundamentals
Python libraries for Data Science
Data Analysis
SQL for Data Science
Maths for Machine Learning
ML Algorithms
Practical ML
MLOPs
Case studies
You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339afe4b0615a1bbed390
Will Deep Learning and NLP be a part of this program?
No, NLP and Deep Learning both are not a part of this program’s curriculum.
What if I miss a live session? Will I get a recording of the session?
Yes all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
Where can I find the class schedule?
Checkout this google sheet to see month by month time table of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6ORAeCg4emgexhqqPv3noQPYKU7RJ6ArOzk/edit?usp=sharing.
What is the time duration of all the live sessions?
Roughly, all the sessions last 2 hours.
What is the language spoken by the instructor during the sessions?
Hinglish
How will I be informed about the upcoming class?
You will get a mail from our side before every paid session once you become a paid user.
Can I do this course if I am from a non-tech background?
Yes, absolutely.
I am late, can I join the program in the middle?
Absolutely, you can join the program anytime.
If I join/pay in the middle, will I be able to see all the past lectures?
Yes, once you make the payment you will be able to see all the past content in your dashboard.
Where do I have to submit the task?
You don’t have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
Will we do case studies in the program?
Yes.
Where can we contact you?
You can mail us at nitish.campusx@gmail.com
Payment/Registration related questions
Where do we have to make our payments? Your YouTube channel or website?
You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
Can we pay the entire amount of Rs 5600 all at once?
Unfortunately no, the program follows a monthly subscription model.
What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb
15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don’t have to wait for a month to end.
What if I don’t like the course after making the payment. What is the refund policy?
You get a 7 days refund period from the day you have made the payment.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmail.com
Post registration queries
Till when can I view the paid videos on the website?
This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased subscription on 21st Jan, you will be able to watch all the past paid sessions in the period of 21st Jan to 20th Feb. But after 21st Feb you will have to purchase the subscription again.
But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessions till Aug 2024.
Why lifetime validity is not provided?
Because of the low course fee.
Where can I reach out in case of a doubt after the session?
You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance session
If I join the program late, can I still ask past week doubts?
Yes, just select past week doubt in the doubt clearance google form.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmai.com
Certificate and Placement Assistance related queries
What is the criteria to get the certificate?
There are 2 criterias:
You have to pay the entire fee of Rs 5600
You have to attempt all the course assessments.
I am joining late. How can I pay payment of the earlier months?
You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
I have read that Placement assistance is a part of this program. What comes under Placement assistance?
This is to clarify that Placement assistance does not mean Placement guarantee. So we dont guarantee you any jobs or for that matter even interview calls. So if you are planning to join this course just for placements, I am afraid you will be disappointed. Here is what comes under placement assistance
Portfolio Building sessions
Soft skill sessions
Sessions with industry mentors
Discussion on Job hunting strategies
"""

# Tokenization
nltk.download("punkt")
nltk.download("punkt_tab")

# tokenize the document into words
tokens = word_tokenize(document.lower())


# build vocabulary
vocab = {"<unk>": 0}  # unknown token
for token in Counter(tokens).keys():
    if token not in vocab:
        vocab[token] = len(vocab)

# extract sentences from the document
input_sentences = document.split("\n")


def text_to_indices(text, vocab):
    numerical_indices = []
    for token in text:
        if token in vocab:
            numerical_indices.append(vocab[token])
        else:
            numerical_indices.append(vocab["<unk>"])
    return numerical_indices


input_numerical_sentences = []
for sentence in input_sentences:
    words = word_tokenize(sentence.lower())
    indices = text_to_indices(words, vocab)
    input_numerical_sentences.append(indices)

training_sequences = []
for sentence in input_numerical_sentences:
    for i in range(1, len(sentence)):
        input_seq = sentence[:i]
        target_seq = sentence[i]
        training_sequences.append((input_seq, target_seq))

print(len(training_sequences))
len_list = []
for seq in training_sequences:
    len_list.append(len(seq[0]))

print(max(len_list))

padded_training_sequences = []
max_length = max(len_list)
for seq in training_sequences:
    input_seq = seq[0]
    target_seq = seq[1]
    padded_input_seq = [0] * (max_length - len(input_seq)) + input_seq
    padded_training_sequences.append((padded_input_seq, target_seq))

# X, y
X = []
y = []
for seq in padded_training_sequences:
    X.append(seq[0])
    y.append(seq[1])

# to tensor
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = CustomDataset(X, y)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(
            embedded
        )
        output = self.fc(final_hidden_state.squeeze(0))
        return output


model = LSTMModel(
    vocab_size=len(vocab), embedding_dim=100, hidden_dim=150, output_dim=len(vocab)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 50
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def predict_next_word(model, input_text, vocab, max_length=10):
    model.eval()
    tokens = word_tokenize(input_text.lower())
    indices = text_to_indices(tokens, vocab)
    padded_input = [0] * (max_length - len(indices)) + indices
    input_tensor = (
        torch.tensor(padded_input, dtype=torch.long).unsqueeze(0).to(device)
    )

    output = model(input_tensor)
    value, predicted_index = torch.max(output, dim=1)

    return list(vocab.keys())[predicted_index]


import time

num_predictions = 10
input_text = "The course follows a monthly"
for _ in range(num_predictions):
    next_word = predict_next_word(model, input_text, vocab, max_length)
    input_text += " " + next_word
    print(input_text)
    time.sleep(1)


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, predicted_indices = torch.max(outputs, dim=1)

            correct_predictions += (predicted_indices == batch_y).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


accuracy = calculate_accuracy(model=model, dataloader=dataloader, device=device)
print(f"Model Accuracy: {accuracy:.2f}%")
