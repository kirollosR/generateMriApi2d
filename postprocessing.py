import os
from PIL import Image, ImageOps
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil



def resize_images(source_dir, target_dir, output_dir):
    # Get the list of image files from both directories
    source_images = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    target_images = sorted([f for f in os.listdir(target_dir) if f.endswith('.png')])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for source_img_name, target_img_name in zip(source_images, target_images):
        # Open the source image and get its size
        source_img_path = os.path.join(source_dir, source_img_name)
        source_img = Image.open(source_img_path)
        source_size = source_img.size

        # Open the target image
        target_img_path = os.path.join(target_dir, target_img_name)
        target_img = Image.open(target_img_path)

        # Resize the target image to match the source image's size
        resized_target_img = target_img.resize(source_size, Image.LANCZOS)

        # Save the resized image to the output directory
        output_img_path = os.path.join(output_dir, target_img_name)
        resized_target_img.save(output_img_path)

        # print(f'Resized {target_img_name} to {source_size} and saved to {output_img_path}')
    print("Phase 4 'resize images' Done")


def add_border(input_image_path, output_image_path, desired_size=240):
    image = Image.open(input_image_path)

    # Calculate the border sizes
    width, height = image.size
    left = (desired_size - width) // 2
    top = (desired_size - height) // 2
    right = desired_size - width - left
    bottom = desired_size - height - top

    # Add the border
    image_with_border = ImageOps.expand(image, border=(left, top, right, bottom), fill='black')

    # Save the image
    image_with_border.save(output_image_path)


def process_images_in_folder(input_folder, output_folder, desired_size=240):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            add_border(input_image_path, output_image_path, desired_size)
    print("Phase 5 Done")

def copy_images(source_folder, destination_folder):
    # Check if both folders exist
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return
    if not os.path.exists(destination_folder):
        print(f"Destination folder '{destination_folder}' does not exist.")
        return

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Iterate through each file in the source folder
    for file_name in files:
        # Check if the file is an image (you can modify this condition as needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Build the full path of the source file
            source_file = os.path.join(source_folder, file_name)
            # Build the full path of the destination file
            destination_file = os.path.join(destination_folder, file_name)
            # Copy the image file from the source folder to the destination folder
            shutil.copy2(source_file, destination_file)
            #print(f"Copied '{file_name}' to '{destination_folder}'.")
    print("Phase 6 Done")


def create_black_image(path, size=(240, 240)):
    """Create a black image of the given size and save it to the specified path."""
    img = Image.new('RGB', size, color=(0, 0, 0))
    img.save(path)


def add_black_images(folder_path, total_images=155, image_size=(240, 240)):
    """Add black images to the folder to reach the specified total number of images."""
    # Get the list of PNG images in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    num_images = len(images)

    # Calculate the number of black images needed
    num_black_images = total_images - num_images
    if num_black_images <= 0:
        print("The folder already contains 155 or more images.")
        return

    # Calculate how many black images to add to the beginning and end
    num_black_each_side = num_black_images // 2
    additional_image = num_black_images % 2  # 1 if odd, 0 if even

    # Create black images and add them to the folder
    for i in range(num_black_each_side):
        create_black_image(os.path.join(folder_path, f'black_start_{i + 1}.png'), image_size)
        create_black_image(os.path.join(folder_path, f'black_end_{i + 1}.png'), image_size)

    if additional_image:
        create_black_image(os.path.join(folder_path, 'black_end_extra.png'), image_size)

    # Move existing images to make space for new images
    existing_images = sorted([f for f in os.listdir(folder_path) if f.endswith('.png') and 'black_' not in f])
    for i, image in enumerate(existing_images):
        os.rename(os.path.join(folder_path, image), os.path.join(folder_path, f'existing_{i + 1}.png'))

    # Rename black images to be at the beginning and end of the folder
    black_images_start = sorted([f for f in os.listdir(folder_path) if f.startswith('black_start')])
    black_images_end = sorted([f for f in os.listdir(folder_path) if f.startswith('black_end')])

    for i, black_image in enumerate(black_images_start):
        os.rename(os.path.join(folder_path, black_image), os.path.join(folder_path, f'image_{i + 1:03}.png'))

    for i, black_image in enumerate(black_images_end):
        os.rename(os.path.join(folder_path, black_image),
                  os.path.join(folder_path, f'image_{total_images - num_black_each_side + i:03}.png'))

    # Rename back existing images in order after black images at the beginning
    for i, image in enumerate(existing_images):
        os.rename(os.path.join(folder_path, f'existing_{i + 1}.png'),
                  os.path.join(folder_path, f'image_{num_black_each_side + i + 1:03}.png'))

    print(f"Added {num_black_images} black images to the folder.")

# def postprocess_images(folder_path, output_folder_path, new_size=(240, 240), border_size=150, border_color='black',
#                    num_black_images=10):
#     # Ensure the output folder exists
#     os.makedirs(output_folder_path, exist_ok=True)
#
#     # Delete all existing files in the output folder
#     for file in os.listdir(output_folder_path):
#         file_path = os.path.join(output_folder_path, file)
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#
#     # List all image files in the folder
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#     # Sort the image files to maintain a consistent order
#     image_files.sort()
#
#     # Check if the folder contains fewer than 155 images
#     if len(image_files) < 155:
#         # Generate black images with the same naming format until there are 155 images
#         for i in range(155 - len(image_files)):
#             # Generate a new filename
#             new_filename = f'black_image_{len(image_files) + i + 1:03}.png'
#             image_files.append(new_filename)
#             # Create a black image of size 240x240 (optional, not saving in the folder)
#             # black_image = Image.new('RGB', new_size, (0, 0, 0))
#             # black_image.save(os.path.join(folder_path, new_filename))
#
#     # Refresh the list of image files after adding black images
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     image_files.sort()
#
#     # Identify the first and last num_black_images images
#     first_images = image_files[:num_black_images]
#     last_images = image_files[-num_black_images:]
#
#     # Combine the first and last num_black_images images into a single list
#     replace_images = first_images + last_images
#
#     # Process the images
#     for filename in image_files:
#         original_image_path = os.path.join(folder_path, filename)
#         modified_image_path = os.path.join(output_folder_path, filename)
#
#         if filename in replace_images:
#             # Create a black image of size 240x240
#             black_image = Image.new('RGB', new_size, (0, 0, 0))
#             black_image.save(modified_image_path)
#             # print(f"Replaced with black image: {modified_image_path}")
#         else:
#             # Open the original image
#             original_image = Image.open(original_image_path)
#
#             # Add a black border to maintain aspect ratio after resizing
#             bordered_image = ImageOps.expand(original_image, border=border_size, fill=border_color)
#
#             # Resize the image with the new dimensions
#             resized_image = bordered_image.resize(new_size)
#
#             # Save the modified image
#             resized_image.save(modified_image_path)
#             # print(f"Processed and saved: {modified_image_path}")
#
#     print("Phase 4 'post processing' Done")

def create_rotated_nifti(input_dir, output_path, image_size=(240, 240)):
    # Load the dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                  input_dir,
                                  seed=123,
                                  labels=None,
                                  image_size=image_size,
                                  batch_size=1,
                                  shuffle=False)

    # Initialize an empty list to store images
    image_list = []

    # Iterate through the dataset and append images to the list
    for image_batch in dataset:
        for image in image_batch:
            # Convert image to grayscale if it's not already
            if image.shape[-1] == 3:
                image = tf.image.rgb_to_grayscale(image)
            # Convert image to numpy array
            image_np = image.numpy().squeeze()
            image_list.append(image_np)

    # Stack the list of images into a 3D numpy array
    image_stack = np.stack(image_list, axis=-1)

    # Rotate the 3D numpy array 90 degrees counterclockwise along the first two axes
    image_stack_rotated = np.rot90(image_stack, k=2, axes=(0, 1))

    # Create a NIfTI image
    nii_image = nib.Nifti1Image(image_stack_rotated, affine=np.eye(4))

    # Save the NIfTI image
    nib.save(nii_image, output_path)

    print("Phase 8 'create nii file' Done")


def create_gif_from_pngs(png_files_list, output_gif_path, num_black_images=6, duration=10):
    images = []
    # Determine dimensions for the black images (assuming all images are the same size)
    with Image.open(png_files_list[0]) as img:
        width, height = img.size
        black_image = Image.new('P', (width, height), 0)  # '0' for black in 'P' mode

    # Replace first num_black_images and last num_black_images with black images
    if len(png_files_list) > 2 * num_black_images:
        png_files_list = [black_image]*num_black_images + png_files_list[num_black_images:-num_black_images] + [black_image]*num_black_images
    else:
        # If the list is too short, just fill it with black images
        png_files_list = [black_image] * len(png_files_list)

    # Load images and convert them
    for png_file in png_files_list:
        if isinstance(png_file, Image.Image):
            img = png_file  # This is already a black image
        else:
            with Image.open(png_file) as img:
                img = img.convert('P')
        images.append(img)

    # Save the images as a GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

def create_png(nifti_file, output_path):
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Determine the number of slices
    num_slices = data.shape[2]

    # Determine the middle slices for an 8x4 grid
    rows, cols = 4, 8
    total_plots = rows * cols
    start_slice = (num_slices // 2) - (total_plots // 2)
    end_slice = start_slice + total_plots

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7),
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.patch.set_facecolor('black')  # Set the background color to black

    # Plot the middle slices in the grid
    for i in range(total_plots):
        slice_idx = start_slice + i
        row = i // cols
        col = i % cols
        axes[row, col].imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[row, col].axis('off')

    # Remove spacing and margins around the subplots
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.savefig(output_path)