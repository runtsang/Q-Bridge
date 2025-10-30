import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalHead(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(x))

class HybridCNN(nn.Module):
    def __init__(self, head: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = head or ClassicalHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

class ClassicalQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return out

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, tagset_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(sentence).unsqueeze(1)
        lstm_out = self.lstm(embeds).squeeze(1)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=1)

class HybridQuantumHybridNet:
    def __init__(self, vocab_size: int, tagset_size: int):
        self.image_classifier = HybridCNN()
        self.text_tagger = LSTMTagger(vocab_size, embedding_dim=128, hidden_dim=256, tagset_size=tagset_size)

    def classify_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.image_classifier(img)

    def tag_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        return self.text_tagger(seq)
