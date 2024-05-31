from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


import model
import test
import generate
import model_mapping

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed to allow requests from specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test.router)
app.include_router(generate.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

