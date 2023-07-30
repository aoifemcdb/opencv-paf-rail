import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour_shape_sensing.color_boundaries import ColorBoundaries
from perspective_transform import order_points, four_point_transform

# Read the image
image = cv2.imread('./experiment_images_040723/parallel/angle_10/50mm/WIN_20230704_10_28_55_Pro.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color = 'green'
color_boundaries = ColorBoundaries()
lower_color, upper_color = color_boundaries.get_color_boundaries(color)

mask = cv2.inRange(hsv, lower_color, upper_color)
contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Select the largest contour
largest_contour = contours[0]

# Approximate the contour with four points
# epsilon = 0.03 * cv2.arcLength(largest_contour, True)
# approx = cv2.approxPolyDP(largest_contour, epsilon, True)
#
# # Find the convex hull of the contour
hull = cv2.convexHull(largest_contour)

# Find the minimum area bounding rectangle
rect = cv2.minAreaRect(hull)
box = cv2.boxPoints(rect)
box = np.int0(box)

# Draw the contour on the image
contour_image = image.copy()
cv2.drawContours(contour_image, [largest_contour], -1, (0, 0, 255), 2)

# Draw the bounding parallelogram on the image
box_image = image.copy()
cv2.drawContours(box_image, [box], 0, (0, 0, 255), 2)

# Extract the four vertices of the bounding parallelogram
vertices = box.tolist()

# Print the vertices
print("Bounding Parallelogram Vertices:")
for vertex in vertices:
    print(vertex)

vertices = np.array(vertices, dtype=np.float32)

# ordered_points = order_points(vertices)
# print(ordered_points)
warped_image, bl, br, tl, tr = four_point_transform(image, vertices)

print("bl:", bl)
print("br:", br)
print("tl:", tl)
print("tr:", tr)


br = br.astype(int)
bl = bl.astype(int)
tl = tl.astype(int)
tr = tr.astype(int)

cv2.putText(box_image, "br", tuple(br), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
cv2.putText(box_image, "bl", tuple(bl), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
cv2.putText(box_image, "tl", tuple(tl), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
cv2.putText(box_image, "tr", tuple(tr), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)




# Create subplots to show the original image, contour, and bounding parallelogram
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0,0].set_title('Original Image')
axes[0,0].axis('off')
axes[0,1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
axes[0,1].set_title('Mask')
axes[0,1].axis('off')
axes[1,0].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
axes[1,0].set_title('Contour')
axes[1,0].axis('off')
axes[1,1].imshow(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB))
axes[1,1].set_title('Bounding Parallelogram')
# axes[1,1].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()
plt.show()


plt.figure()
plt.imshow(image)
plt.imshow(warped_image)
plt.show()





