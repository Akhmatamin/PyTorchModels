import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, File

flower_router = APIRouter(prefix='/flowers', tags=['Flowers'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = {
    0:'daisy',
    1:'dandelion',
    2:'rose',
    3:'sunflower',
    4:'tulip'
}

class FlowerCheck(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 8 * 8, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 5),
        )

  def forward(self, flower):
    flower = self.first(flower)
    return self.second(flower)


transform_data = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


model = FlowerCheck().to(device)
model.load_state_dict(torch.load('model_files/modelFlower.pth', map_location=device))
model.eval()

@flower_router.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        if not image:
            raise HTTPException(status_code=404, detail='No image received')

        img = Image.open(io.BytesIO(image)).convert('RGB')
        image_tensor = transform_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_tensor)
            prediction = y_pred.argmax(dim=1).item()

        return {"prediction": classes.get(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
