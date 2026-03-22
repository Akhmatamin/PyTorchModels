import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, File
from torchvision.datasets import CIFAR100

dataset = CIFAR100(root='./data', train=True, download=True)
classes = dataset.classes



cifar100_router = APIRouter(prefix="/cifar100", tags=["cifar100"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CifarClassification(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 100),
    )

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

model = CifarClassification()
model.load_state_dict(torch.load("model_files/cifar100_model.pth", map_location=device))
model.to(device)
model.eval()

@cifar100_router.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="File not found")

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_tensor)
            prediction = y_pred.argmax(dim=1).item()
        return {'prediction': classes[prediction]}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))






