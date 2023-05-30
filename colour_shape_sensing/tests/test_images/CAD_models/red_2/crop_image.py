from PIL import Image

# set the dimensions for cropping
left = 400
top = 200
right = 1700
bottom = 1000

# iterate over all input images
for filename in ['red_calibration.jpg', 'red_50mm.jpg']:
    # open the image and crop it
    with Image.open(filename) as img:
        cropped_img = img.crop((left, top, right, bottom))
        # save the cropped image
        cropped_img.save('cropped_' + filename)
