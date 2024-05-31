import datetime
import os

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import nibabel as nib
import numpy as np
from pathlib import Path

import preprocessing
import model
import model_mapping

UPLOAD_DIR = Path() / "MRIs/uploaded"

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
)

@router.get("/")
async def root():
    return {"message": "Hello World from generate"}

def rename(filename):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y_%H%M%S")

    if filename.endswith('.nii.gz'):
        new_filename = f"{filename[:-7]}_{timestamp}.nii.gz"
    elif filename.endswith('.nii'):
        new_filename = f"{filename[:-4]}_{timestamp}.nii"
    return new_filename

@router.post("/outputShape/{from_seq}/{to_seq}")
async def preprocess_and_predict(from_seq: str, to_seq: str, file: UploadFile = File(...)):
    model.cycle_gan_model.load_weights(model_mapping.get_model(from_seq, to_seq))

    # Load NIfTI file directly from the file contents
    contents = await file.read()

    new_filename = rename(file.filename)

    save_path = UPLOAD_DIR / new_filename

    with open(save_path, "wb") as f:
        f.write(contents)

    nifti_image = nib.load(save_path)
    img_data = nifti_image.get_fdata()

    # Preprocess image
    preprocessed_img = preprocessing.preprocess_image(img_data)
    image_data = preprocessed_img[np.newaxis, :, :, :]


    # Predict output using the model
    output_tensor = model.gen_A2B(image_data)

    output_reshaped = preprocessing.reshape_image(output_tensor)

    file_name = preprocessing.to_nii(output_reshaped)

    # return FileResponse(file_name, filename="generatedMri.nii")
    # return {"filename": file_name}
    return {
        "status": "success",
        "message": "File uploaded and generated successfully",
        "From_Sequence": from_seq,
        "to_sequence": to_seq,
        "input_filename": new_filename,
        "output_filename": file_name,
        "output_path": f"MRIs/generated/{file_name}"
    }