from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import nibabel as nib
import preprocessing

router = APIRouter(
    prefix="/test",
    tags=["test"],
)

@router.get("/")
async def root():
    return {"message": "Hello World from API"}

@router.post("/testNiiIn/{fromSeq}/{toSeq}")
async def generateMri(fromSeq: str, toSeq: str, niftiFile: UploadFile = File(...)  ):
    contents = await niftiFile.read()

    file_path = f"MRIs/{niftiFile.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "from": fromSeq,
        "to": toSeq,
        "file": FileResponse(file_path, filename=niftiFile.filename)
    }

@router.post("/generateTest/{fromSeq}/{toSeq}")
async def generateMri(fromSeq: str, toSeq: str, niftiFile: UploadFile = File(...)  ):
    contents = await niftiFile.read()

    # with open("temp.nii", "wb") as buffer:
    #     buffer.write(contents)
    #
    # nifti_image = nib.load("temp.nii")
    # nifti_data = nifti_image.get_fdata()
    #
    # processed_data = model.predict(np.expand_dims(nifti_data, axis=0))
    #
    # processed_nifti_image = nib.Nifti1Image(processed_data[0], nifti_image.affine)
    #
    # nib.save(processed_nifti_image, "processed.nii")

    return {
        "from": fromSeq,
        "to": toSeq,
        "file": FileResponse("processed.nii", media_type="application/octet-stream")
    }

#view
@router.get("/imgOut")
async def generateMri():
    return FileResponse("epoch_0_visualization.png", media_type="image/png")

#download
@router.get("/niiOut")
async def generateMri():
    return FileResponse("MRIs/BraTS-GLI-00001-001-t1c.nii", filename="mri.nii")

from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@router.get("/download-nii")
async def download_nii():
    # Replace 'path_to_your_file.nii' with the actual path to your .nii file
    file_path = "MRIs/generated/generated_02-06-2024_235045/generatedMri_t2_from_t1.png"
    # Return the file as a response, allowing it to be downloaded
    return FileResponse(file_path, media_type="image/png")

@router.get("/download-png")
async def download_png():
    # Replace 'path_to_your_file.nii' with the actual path to your .nii file
    file_path = "MRIs/BraTS-GLI-00001-001-t1c.nii"
    # Return the file as a response, allowing it to be downloaded
    return FileResponse(file_path)

from pathlib import Path

UPLOAD_DIR = Path() / "MRIs"
@router.post("/upload-nii")
async def upload_nii(niftiFile: UploadFile = File(...) ):
    contents = await niftiFile.read()
    save_path = UPLOAD_DIR / niftiFile.filename
    # file_path = f"MRIs/{niftiFile.filename}"
    with open(save_path, "wb") as f:
        f.write(contents)
    return {"filename": niftiFile.filename}

# upload 1 file
@router.post("/uploadDownload-nii")
async def upload_nii(niftiFile: UploadFile = File(...) ):
    contents = await niftiFile.read()
    save_path = UPLOAD_DIR / niftiFile.filename
    # file_path = f"MRIs/{niftiFile.filename}"
    with open(save_path, "wb") as f:
        f.write(contents)
    return FileResponse(save_path, filename="mri.nii")

# upload multiple files
# @router.post("/upload-nii")
# async def upload_nii(niftiFiles: list[UploadFile] ):
#     for niftiFile in niftiFiles:
#         contents = await niftiFile.read()
#         save_path = UPLOAD_DIR / niftiFile.filename
#         file_path = f"MRIs/{niftiFile.filename}"
#         with open(save_path, "wb") as f:
#             f.write(contents)
#
#     return {"filename": [niftiFile.filename for f in niftiFiles]}

@router.post("/preprocess")
async def preprocess_nifti(file: UploadFile = File(...)):
    # Read NIfTI file
    contents = await file.read()

    # Load NIfTI file directly from the file contents
    nifti_image = nib.Nifti1Image.from_bytes(contents)
    img_data = nifti_image.get_fdata()

    # Preprocess image
    preprocessed_img = preprocessing.preprocess_image(img_data)

    # Get shape of preprocessed image
    preprocessed_shape = preprocessed_img.shape

    return {"preprocessed_shape": preprocessed_shape}