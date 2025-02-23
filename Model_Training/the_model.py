import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, input_size, num_classes, embed_dim=32):
        super(EEGNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.embed = nn.Linear(64, embed_dim)  # Embedding layer for contrastive learning
        self.fc3 = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        embedding = self.embed(x)  # Feature representation
        logits = self.fc3(embedding)

        if return_embedding:
            return logits, embedding
        return logits

class TeacherEEGNet(nn.Module):
    def __init__(self, input_size, num_classes, embed_dim=64):
        super(TeacherEEGNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.embed = nn.Linear(128, embed_dim)  # Embedding layer for contrastive learning
        self.fc3 = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        embedding = self.embed(x)  # Feature representation
        logits = self.fc3(embedding)

        if return_embedding:
            return logits, embedding
        return logits

# Example usage
def get_student_model(input_size=32, num_classes=2, embed_dim=32):
    return EEGNet(input_size, num_classes, embed_dim)

def get_teacher_model(input_size=32, num_classes=2, embed_dim=64):
    return TeacherEEGNet(input_size, num_classes, embed_dim)