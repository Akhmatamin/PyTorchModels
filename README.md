# PyTorch ML Models Dashboard

A collection of PyTorch models served through a **FastAPI** backend and explored through a **Streamlit** dashboard. Upload an image or audio clip and get a prediction with a confidence score.

## Models

| Task | Endpoint | Input |
|------|----------|-------|
| Handwritten digits (MNIST) | `/numbers/predict/` | Image |
| Fashion items (Fashion-MNIST) | `/fashion/...` | Image |
| CIFAR-10 objects | `/cifar10/...` | Image |
| CIFAR-100 objects | `/cifar100/...` | Image |
| Transport | `/transport/...` | Image |
| Flowers | `/flowers/...` | Image |
| Image scenes (buildings, forest, glacier, mountain, sea, street) | `/image_scene/...` | Image |
| Trash (cardboard, glass, metal, paper, plastic, trash) | `/trash/...` | Image |
| Audio classification | `/audio/...` | Audio |
| Music genres (GTZAN) | `/gtzan/...` | Audio |
| Environmental sounds | `/environment/...` | Audio |
| Urban sounds | `/urban/...` | Audio |
| Speech emotion (RAVDESS) | `/ravdess/...` | Audio |

Interactive API docs are available at <http://127.0.0.1:8000/docs> once the server is running.

## Project structure

```
.
├── main.py              # FastAPI app, registers all model routers
├── models/              # One module per model (architecture + router)
├── Streamlit/           # Dashboard (main_front.py) and per-model pages
├── photos/              # Screenshots
├── gues_random_num.py   # Standalone number-guessing console game
└── model_files/         # Trained weights (*.pth) – not tracked in git
```

## Setup

```bash
pip install torch torchvision torchaudio fastapi uvicorn python-multipart pillow streamlit requests
```

Depending on the audio models you may also need `librosa` or `soundfile`.

Trained weights are not included in the repository (`*.pth` and `model_files/` are git-ignored). Place them in `model_files/` with the filenames the modules expect, e.g. `model_files/num_model.pth`, `model_files/fashion_model.pth`, `model_files/cifar10_model.pth`. `main.py` loads every model at startup, so all weight files must be present.

## Running

Start the API from the project root:

```bash
python main.py
```

Then, in a second terminal, launch the dashboard:

```bash
streamlit run Streamlit/main_front.py
```

The dashboard talks to the API at `http://127.0.0.1:8000`.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/numbers/predict/ -F "file=@digit.png"
# {"prediction": 7, "confidence": 0.9981}
```

For digits, use a white digit on a black background, centered, one per image (images are resized to 28×28).

## Built with

PyTorch · torchvision · FastAPI · Uvicorn · Streamlit
