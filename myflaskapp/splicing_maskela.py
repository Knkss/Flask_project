# User
#splicing.py

import cv2
import numpy as np

def detect_splicing(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate the edges
    dilated = cv2.dilate(edges, None)

    # Find contours in the edges
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the splicing areas
    mask = np.zeros_like(image)

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # If the area is large enough, draw it on the mask
        if area > 500:
            cv2.drawContours(mask, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Combine the original image with the mask
    result = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    # Save the result
    cv2.imwrite('C:\\Users\\Sanjivkumar Naik\\Desktop\\myflaskapp\\images\\tree-736885_1280.jpg', result)


# Use the function with the correct image path
detect_splicing("C:\\Users\\Sanjivkumar Naik\\Desktop\\myflaskapp\\images\\tree-736885_1280.jpg")