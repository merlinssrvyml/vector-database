import torchvision.transforms as T
from app.search import search
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((28, 28)),
    T.ToTensor()
])

@app.post("/search")
async def search_image(file: UploadFile):
    image = Image.open(file.file)
    tensor = transform(image)

    results = search(tensor)

    return [
        {
            "label": hit.entity.get("label"),
            "distance": hit.distance
        }
        for hit in results
    ]
