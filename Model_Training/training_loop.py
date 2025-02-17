import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from Model_Training.the_model import *
from Model_Training.dataset_loader import *
import torch.nn.functional as F

# Automatically detect best available device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2/M3
    elif torch_directml.is_available():
        return torch_directml.device()  # Windows DirectML
    else:
        return torch.device("cpu")  # Default to CPU

device = get_device()
print(f"Using device: {device}")

# Contrastive Loss (Triplet Loss)
def contrastive_loss_fn(embeddings, labels, margin=1.0):
    # Get indices of each class
    idx_0 = torch.where(labels == 0)[0]  # Healthy
    idx_1 = torch.where(labels == 1)[0]  # Affected

    # Find the minimum number of samples in both classes
    min_size = min(len(idx_0), len(idx_1))

    if min_size == 0:
        return torch.tensor(0.0, requires_grad=True)  # Return zero loss if one class is missing

    # Randomly select min_size samples from both classes
    idx_0 = idx_0[torch.randperm(len(idx_0))[:min_size]]
    idx_1 = idx_1[torch.randperm(len(idx_1))[:min_size]]

    # Select embeddings
    anchor = embeddings[idx_0]
    positive = embeddings[idx_1]
    negative = embeddings[idx_1]  # Can also randomize negatives differently

    # Compute triplet margin loss
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)

# Knowledge Distillation Loss (KL Divergence)
def kd_loss_fn(student_logits, teacher_logits, temperature=3.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

# Training loop
def train_model(student_model, teacher_model, train_loader, optimizer, criterion, epochs=10):
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()  # Ensure teacher model is in evaluation mode

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            student_outputs, student_embeddings = student_model(inputs, return_embedding=True)
            teacher_outputs, _ = teacher_model(inputs, return_embedding=True)

            # Compute Losses
            ce_loss = criterion(student_outputs, labels)
            contrastive_loss = contrastive_loss_fn(student_embeddings, labels)
            kd_loss = kd_loss_fn(student_outputs, teacher_outputs.detach())

            total_loss = ce_loss + contrastive_loss + kd_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    print("Training complete!")

# Save Model Function
def save_the_model(model, path_to_saved_model):
    torch.save(model.state_dict(), path_to_saved_model)
    print(f"Model saved to {path_to_saved_model}")

# Load Dataset
healthy_train_loader = healthy_dataloader
affected_train_loader = affected_dataloader

# Combine both datasets
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([healthy_train_loader.dataset, affected_train_loader.dataset]),
    batch_size=32, shuffle=True
)

# Initialize Models
input_size = healthy_train_loader.dataset.features.shape[1]  # 19
num_classes = 2  # Healthy (0) vs. Affected (1)


student_model = get_student_model(input_size=input_size, num_classes=num_classes)
teacher_model = get_teacher_model(input_size=input_size, num_classes=num_classes)

# Optimizer & Loss Function
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Start Training
train_model(student_model, teacher_model, train_loader, optimizer, criterion)

# Save the trained model
save_the_model(student_model, './models/student_model.pth')
