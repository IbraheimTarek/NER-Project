import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from utils import load_vocab, process_text_input, load_model

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.bilstm_1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected layers
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # BiLSTM layer
        lstm_out, _ = self.bilstm_1(embedded)

        output = self.fc2(lstm_out)
        return F.log_softmax(output, dim=-1)
    


# -----------------------------------------
# Streamlit Application
# -----------------------------------------
st.title("TOP Decoupled Prediction App")

st.write("Enter a sentence to generate a TOP-decoupled hierarchical representation.")

device = torch.device("cpu")
vocab = load_vocab()

input_dim = len(vocab)
embedding_dim = 128
hidden_dim = 128
output_dim1 = 6
output_dim2 = 22
num_layers = 2
dropout = 0.3

model_1 = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim1, num_layers, dropout).to(device)
model_2 = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim2, num_layers, dropout).to(device)

model_1 = load_model(model_1, "./weights/Bilstm_order_sequence.pt")
model_2 = load_model(model_2, "./weights/Bilstm_model2.pt")

input_text = st.text_area("Enter your text here:", "I would like one large thin crust pizza with hot cheese and pepperoni")

if st.button("Process Text"):
    predicted_json, top_decoupled = process_text_input(input_text, model_1, model_2, vocab, device)
    st.write("**TOP Decoupled:**")
    st.write(top_decoupled)
    st.write("**JSON Output:**")
    st.json(predicted_json)
