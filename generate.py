import datetime
import os
import shutil

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import nibabel as nib
import numpy as np
from pathlib import Path
import glob

import preprocessing
import model
import model_mapping
import postprocessing

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

def create_temp_dir():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y_%H%M%S")
    # Create a temporary directory to store the generated images
    temp_dir = os.path.join(os.getcwd(), f'temps/temp_{timestamp}')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        Path(os.path.join(temp_dir, f"Converted_png_from_nii")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(temp_dir, f"crop_black_boundary")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(temp_dir, f"Generated_results")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(temp_dir, f"Generated_resized_results")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(temp_dir, f"Generated_results_Final")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(temp_dir, f"Add_black_images")).mkdir(parents=True, exist_ok=True)
    return temp_dir


def delete_temp_dir(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Folder '{temp_dir}' has been deleted.")
    else:
        print(f"Folder '{temp_dir}' does not exist.")

def create_generated_dir():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y_%H%M%S")
    path_filename = f"MRIs/generated/generated_{timestamp}"
    generated_dir = os.path.join(os.getcwd(), path_filename)
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    return generated_dir, path_filename

@router.post("/{from_seq}/{to_seq}")
async def preprocess_and_predict(from_seq: str, to_seq: str, file: UploadFile = File(...)):
    if not model_mapping.get_model(from_seq, to_seq):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Model not found for the given sequence"
            }
        )

    model.load_model(model_mapping.get_model(from_seq, to_seq))

    temp_dir = create_temp_dir()

    # Load NIfTI file directly from the file contents
    contents = await file.read()
    new_filename = rename(file.filename)
    save_path = UPLOAD_DIR / new_filename
    print(f"Saving file to {save_path}")
    with open(save_path, "wb") as f:
        f.write(contents)

    converted_png_from_nii_path = os.path.join(temp_dir, "Converted_png_from_nii")
    preprocessing.nii_to_png(save_path, converted_png_from_nii_path)

    crop_black_boundary_path = os.path.join(temp_dir, "crop_black_boundary")
    preprocessing.process_images(converted_png_from_nii_path, crop_black_boundary_path)

    generated_results_path = os.path.join(temp_dir, "Generated_results")
    model.Generate(crop_black_boundary_path, generated_results_path)

    generated_results_final_path = os.path.join(temp_dir, "Generated_results_Final")
    generated_resized_results_path = os.path.join(temp_dir, "Generated_resized_results")
    postprocessing.resize_images(crop_black_boundary_path, generated_results_path, generated_resized_results_path)
    # postprocessing.postprocess_images(generated_results_path, generated_results_final_path)

    postprocessing.process_images_in_folder(generated_resized_results_path, generated_results_final_path)

    add_black_images_path = os.path.join(temp_dir, "Add_black_images")
    postprocessing.copy_images(generated_results_final_path, add_black_images_path)

    postprocessing.add_black_images(add_black_images_path)

    filename = f"generatedMri_{to_seq}_from_{from_seq}"

    generated_dir, generated_dir_filename = create_generated_dir()
    generated_nii_path = os.path.join(generated_dir, f"{filename}.nii")
    postprocessing.create_rotated_nifti(add_black_images_path, generated_nii_path)

    png_files = glob.glob(f"{add_black_images_path}\*.png")
    generated_gif_path = os.path.join(generated_dir, f"{filename}.gif")
    postprocessing.create_gif_from_pngs(png_files, generated_gif_path)

    generated_png_path = os.path.join(generated_dir, f"{filename}.png")
    postprocessing.create_png(generated_nii_path, generated_png_path)

    delete_temp_dir(temp_dir)
    return {
        "success": True,
        "message": "File uploaded and generated successfully",
        "From_Sequence": from_seq,
        "to_sequence": to_seq,
        "input_filename": new_filename,
        # "output_filename": file_name,
        # "output_path": f"{generated_nii_path}",
        "output_nii_path": f"{generated_dir_filename}/{filename}.nii",
        "output_gif_path": f"{generated_dir_filename}/{filename}.gif",
        "output_png_path": f"{generated_dir_filename}/{filename}.png",
    }