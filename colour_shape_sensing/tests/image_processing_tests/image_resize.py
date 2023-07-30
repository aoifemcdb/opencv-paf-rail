import os
from PIL import Image


def resize_image(input_directory,output_directory):
    # Define the target size for your images
    target_size = (3200, 1600)

    # Define the resampling filter to use
    resampling_filter = Image.BOX

    # Define the JPEG compression level to use (0-100)
    jpeg_quality = 50

    # Loop through all files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Open the image
            img = Image.open(os.path.join(input_directory, filename))

            # Resize the image
            img = img.resize(target_size, resampling_filter)

            # Define the new filename
            new_filename = os.path.splitext(filename)[0] + '_resized' + os.path.splitext(filename)[1]

            # Save the resized image
            img.save(os.path.join(output_directory, new_filename), quality=jpeg_quality)

            print(f"{filename} has been resized and saved as {new_filename}")

    return

def main():
    input_directory = './test_images/CAD_models/orange'
    output_directory = './test_images/CAD_models/'
    resize_image(input_directory, output_directory)
    return

if __name__ == '__main__':
    main()
