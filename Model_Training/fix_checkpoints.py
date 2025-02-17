import torch

# Load the checkpoint
checkpoint = torch.load('./models/model_one.pth', map_location='cpu')

# Print the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# Modify the checkpoint to match your current model
new_checkpoint = {}
for key, value in checkpoint.items():
    if key.startswith('fc4'):
        # Skip keys that don't exist in the current model
        continue
    # Rename or resize keys as needed
    if key == 'fc1.weight':
        new_checkpoint['fc1.weight'] = value[:128, :19]  # Resize to match current model
    elif key == 'fc1.bias':
        new_checkpoint['fc1.bias'] = value[:128]  # Resize to match current model
    else:
        new_checkpoint[key] = value

# Save the modified checkpoint
torch.save(new_checkpoint, './models/modified_model_one.pth')