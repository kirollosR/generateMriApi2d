import os
import numpy as np
import glob
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def nii_to_png(input_file, output_dir):
    # Load the NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete existing PNG files in the output directory
        existing_png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        for png_file in existing_png_files:
            os.remove(os.path.join(output_dir, png_file))
        #print(f"Deleted {len(existing_png_files)} existing PNG files in {output_dir}")

    # Iterate over the slices in the 3rd dimension
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]

        # Normalize the slice data to 0-255
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
        slice_data = slice_data.astype(np.uint8)

        output_path = os.path.join(output_dir, f'slice_{i:03d}.png')
        plt.imsave(output_path, slice_data, cmap='gray')

    print("Phase 1 'nii to png' Done")


def crop_black_space(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find all non-black pixels
    coords = cv2.findNonZero(gray)

    # Check if there are any non-black pixels found
    if coords is None:
        # If no non-black pixels are found, the image is considered completely black
        # print("Image is completely black.")
        return None

    # Create a bounding box around the non-black pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Check if the bounding box has a valid size
    if w == 0 or h == 0:
        # print("Invalid bounding box size, returning original image")
        return image

    # Crop the image to the bounding box
    cropped = image[y:y + h, x:x + w]
    return cropped

    # Crop the image to the bounding box
    cropped = image[y:y + h, x:x + w]
    return cropped


def process_images(input_folder, output_folder):
    # # Create the output folder if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            # Check if the image was successfully loaded
            if image is None:
                # print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Crop the black space
            cropped_image = crop_black_space(image)

            # If the image is completely black, delete it
            if cropped_image is None:
                # os.remove(img_path)
                # print(f"Deleted completely black image {img_path}.")
                continue

            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)

    print("Phase 2 'crop images' Done")








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

