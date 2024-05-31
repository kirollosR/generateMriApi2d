import os
import numpy as np
import glob
import nibabel as nib
import datetime


# Define your data loading and preprocessing functions here
def data_loader(folder_path, mri_type, num_samples=None, random_state=None):
    image_array = []
    pixel_arrays = []
    patients_processed = 0

    for patient_folder in os.listdir(folder_path):
        patient_folder_path = os.path.join(folder_path, patient_folder)

        if os.path.isdir(patient_folder_path):
            images = glob.glob(os.path.join(patient_folder_path, "*" + mri_type + ".nii"))
            for subfolder in os.listdir(patient_folder_path):
                subfolder_path = os.path.join(patient_folder_path, subfolder)

                if os.path.isdir(subfolder_path):
                    images = glob.glob(os.path.join(subfolder_path, "*" + mri_type + ".nii"))
                    image_array.extend(images)
                    patients_processed += 1

                    if num_samples is not None and len(image_array) >= num_samples:
                        break  # Stop loading if desired number of samples is reached

        if num_samples is not None and len(image_array) >= num_samples:
            break  # Stop loading if desired number of samples is reached

    # Optionally shuffle the data
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(image_array)

    return image_array[:num_samples]


def crop_selected_slices_optimized(pixel_arrays, num_of_slices):
    # Assuming pixel_arrays shape is (240, 240, 155)
    depth = pixel_arrays.shape[2]

    # Calculate start and end slices to select
    start_slice = (depth - num_of_slices) // 2
    end_slice = start_slice + num_of_slices

    # Select the middle 'num_of_slices' slices
    selected_slices = pixel_arrays[:, :, start_slice:end_slice]

    # Calculate cropping coordinates
    crop_x_start = (selected_slices.shape[0] - 176) // 2
    crop_x_end = crop_x_start + 176
    crop_y_start = (selected_slices.shape[1] - 176) // 2
    crop_y_end = crop_y_start + 176

    # Crop the selected slices to 200x200
    cropped_images = selected_slices[crop_x_start:crop_x_end, crop_y_start:crop_y_end, :]

    return cropped_images


def Instance_Normalization(input_data):
    mean = np.mean(input_data)
    std = np.std(input_data)
    Normalization_images = (input_data - mean) / std

    return Normalization_images


def preprocess_image(img):
    img = crop_selected_slices_optimized(img, 124)
    img = Instance_Normalization(img)
    # img = img[:, :, :, np.newaxis]
    return img

def reshape_image(img):
    output_array = img.numpy()
    output_array_squeezed = np.squeeze(output_array)

    # Convert data type to float64
    output_array_squeezed = output_array_squeezed.astype(np.float32)
    return output_array_squeezed

def to_nii(output_array):
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(output_array, affine)
    # Get current date and time
    current_time = datetime.datetime.now()
    # Format the timestamp as a string
    # timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = current_time.strftime("%d-%m-%Y_%H%M%S")
    # Create filename
    filename = f'generatedMri_{timestamp}.nii'
    path = f'MRIs/generated/{filename}'
    # Save the file with the new filename
    nib.save(nifti_file, path)
    return filename

