import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, File

classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

cifar10_router = APIRouter(prefix="/cifar10", tags=["Cifar10"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CifarClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
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
model.load_state_dict(torch.load('model_files/cifar10_model.pth', map_location=device))
model.to(device)
model.eval()



@cifar10_router.post("/predict/")
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
        raise HTTPException(status_code=500, detail=str(e))
