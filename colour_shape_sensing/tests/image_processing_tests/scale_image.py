def scale_image(image, object_length_pixels, object_height_pixels, reference_length_units, reference_height_units):
    # Calculate the scaling factors
    scaling_factor_length = reference_length_units / object_length_pixels
    scaling_factor_height = reference_height_units / object_height_pixels

    # Get the original image dimensions
    height, width = image.shape[:2]

    # Scale the image coordinates
    scaled_height = int(height * scaling_factor_height)
    scaled_width = int(width * scaling_factor_length)

    # Create a blank canvas for the scaled image
    scaled_image = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)

    # Scale the image by transferring pixel values
    for y in range(scaled_height):
        for x in range(scaled_width):
            source_x = int(x / scaling_factor_length)
            source_y = int(y / scaling_factor_height)
            scaled_image[y, x] = image[source_y, source_x]

    return scaled_image

