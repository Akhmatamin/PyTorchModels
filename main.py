from fastapi import FastAPI
import uvicorn
from models.numbers import num_router
from models.fashion import fashion_router
from models.cifar10 import cifar10_router
from models.cifar100 import cifar100_router
from models.transport import transport_router
from models.flowers import flower_router
from models.audio import check_audio
from models.image_scene import image_scene_router
from models.genres import gtza_router
from models.trash import trash_router
from models.environment import environment_router
from models.urban import urban_router
from models.ravdess import ravdess_router

app = FastAPI()

app.include_router(num_router)
app.include_router(fashion_router)
app.include_router(cifar10_router)
app.include_router(cifar100_router)
app.include_router(transport_router)
app.include_router(flower_router)
app.include_router(check_audio)
app.include_router(image_scene_router)
app.include_router(gtza_router)
app.include_router(trash_router)
app.include_router(environment_router)
app.include_router(urban_router)
app.include_router(ravdess_router)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
