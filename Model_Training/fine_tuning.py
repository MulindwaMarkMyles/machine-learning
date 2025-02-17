import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

from Model_Training.the_model import get_student_model, get_teacher_model
from Model_Training.dataset_loader import *
import torch.nn.functional as F

healthy_train_loader = healthy_dataloader
affected_train_loader = affected_dataloader

# Combine both datasets
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([healthy_train_loader.dataset, affected_train_loader.dataset]),
    batch_size=32, shuffle=True
)


# Load the student model
input_size = 19  # Example input size
num_classes = 2  # Example number of classes

student_model_path = "models/model_one/v2.pth"
student_model = get_student_model(input_size=input_size, num_classes=num_classes)
student_model.load_state_dict(torch.load(student_model_path))

# Initialize the teacher model with a larger embedding dimension
teacher_model = get_teacher_model(input_size=input_size, num_classes=num_classes, embed_dim=1024)

# Define a new optimizer and loss function
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Define the contrastive loss function
def contrastive_loss_fn(embeddings, labels, margin=1.0):
    idx_0 = torch.where(labels == 0)[0]
    idx_1 = torch.where(labels == 1)[0]
    min_size = min(len(idx_0), len(idx_1))
    if min_size == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    idx_0 = idx_0[torch.randperm(len(idx_0))[:min_size]]
    idx_1 = idx_1[torch.randperm(len(idx_1))[:min_size]]

    anchor = embeddings[idx_0]
    positive = embeddings[idx_1]
    # Create negative samples by rotating the positive indices
    negative = embeddings[idx_1.roll(1)]  # Each positive sample becomes negative for the next anchor

    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)

# Define the knowledge distillation loss function
def kd_loss_fn(student_logits, teacher_logits, temperature=3.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2/M3
    elif torch_directml.is_available():
        return torch_directml.device()  # Windows DirectML
    else:
        return torch.device("cpu")  # Default to CPU

# Continue training the model
device = get_device()
student_model.to(device)
teacher_model.to(device)


def train_model(student_model, teacher_model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()  # Teacher should be in eval mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Add debug prints
            # print("Input requires grad:", inputs.requires_grad)

            student_outputs, student_embeddings = student_model(inputs, return_embedding=True)
            # print("Student embeddings requires grad:", student_embeddings.requires_grad)

            # with torch.no_grad():  # Teacher shouldn't require gradients
            teacher_outputs, _ = teacher_model(inputs, return_embedding=True)

            ce_loss = criterion(student_outputs, labels)
            # print("CE loss requires grad:", ce_loss.requires_grad)

            contrastive_loss = contrastive_loss_fn(student_embeddings, labels)
            # print("Contrastive loss requires grad:", contrastive_loss.requires_grad)

            kd_loss = kd_loss_fn(student_outputs, teacher_outputs)
            # print("KD loss requires grad:", kd_loss.requires_grad)

            total_loss = ce_loss + contrastive_loss + kd_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    print("Finished training")
# Start training
train_model(student_model, teacher_model, train_loader, optimizer, criterion)

# Save the fine-tuned model
torch.save(student_model.state_dict(), 'models/model_one/v3.pth')