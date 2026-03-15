from fastapi import FastAPI
import uvicorn
from models.numbers import num_router
from models.fashion import fashion_router
from models.cifar10 import cifar10_router
from models.cifar100 import cifar100_router

app = FastAPI()

app.include_router(num_router)
app.include_router(fashion_router)
app.include_router(cifar10_router)
app.include_router(cifar100_router)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
