import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridClassifierQLSTM(nn.Module):
    """
    Classical implementation of a hybrid classifier that combines a
    convolutional backbone, a binary classifier head, and an LSTM tagger.
    """
    def __init__(self, image_channels=3, lstm_hidden_dim=128, lstm_layers=1,
                 vocab_size=1000, tagset_size=10):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(image_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # binary head

        # LSTM tagger
        self.word_embeddings = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, image: torch.Tensor, sentence: torch.Tensor):
        # Image pathway
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        binary_logits = self.fc3(x)
        binary_prob = torch.sigmoid(binary_logits)

        # Sequence pathway
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        tag_log_probs = F.log_softmax(tag_logits, dim=-1)

        return binary_prob, tag_log_probs

__all__ = ["HybridClassifierQLSTM"]
