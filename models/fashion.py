import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, File

fashion_router = APIRouter(prefix="/fashion", tags=["Fashion"])
device_fashion = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16 , kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16*14*14, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.ToTensor(),
])

model = CheckImage()
model.load_state_dict(torch.load('model_files/fashion_model.pth', map_location=device_fashion))
model.to(device_fashion)
model.eval()

@fashion_router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")

        img = Image.open(io.BytesIO(image_data))
        image_tensor = transform(img).unsqueeze(0).to(device_fashion)

        with torch.no_grad():
            y_pred = model(image_tensor)
            prediction = y_pred.argmax(dim=1).item()

        return {'prediction': classes.get(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
