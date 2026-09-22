import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained ResNet model from Hugging Face.
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# MNIST images are 28x28 grayscale images.
# ResNet expects 224x224 RGB images normalized like ImageNet images.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ]
)

test_dataset = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

correct = 0
total = 0

with torch.inference_mode():
    for batch_number, (images, labels) in enumerate(test_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images)
        predictions = outputs.logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_number % 25 == 0:
            print(f"Processed {total}/{len(test_dataset)} images")

accuracy = 100.0 * correct / total

print(f"Correct predictions: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")