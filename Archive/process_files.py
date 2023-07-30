import os
import cv2
import numpy as np

def process_files_in_directory(directory_path, extension, function, *args):
    """
    Process all files with the specified extension in the specified directory using the specified function.

    Args:
        directory_path (str): Path to the directory containing the files.
        extension (str): File extension to process (e.g. '.jpg').
        function (function): Function to apply to each image.

    Returns:
        numpy.ndarray: Numpy array containing the outputs of the function for each image.
    """
    # Create an empty list to store the outputs
    outputs = []

    # Loop over the files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has the desired extension
        if filename.endswith(extension):
            # Load the image
            filepath = os.path.join(directory_path, filename)
            image = cv2.imread(filepath)

            # Apply the function to the image and append the output to the list
            output = function(image, *args)
            outputs.append(output)

    # Convert the list of outputs to a numpy array
    outputs_array = np.array(outputs, dtype=object)
    print(outputs_array)

    return outputs_array


