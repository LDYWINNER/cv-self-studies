
"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-1
Quick Win - Step 1: Simple Object Detection by thresholding with mask
File: finding-mask.py
"""


# Convenient way for choosing right Colour Mask to Detect needed Object
#
# Algorithm:
# Reading RGB image --> Converting to HSV --> Getting Mask
#
# Result:
# min_blue, min_green, min_red = 21, 222, 70
# max_blue, max_green, max_red = 176, 255, 255


# Importing needed library - opencv library
import cv2


# Preparing Track Bars (that will help us to choose numbers for color mask)
# Defining empty function
def do_nothing(x):
    pass


# Giving name to the window with Track Bars
# And specifying that window is resizable
# --> Window itself is defined here
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

# Defining Track Bars for convenient process of choosing colours
# first argument: specify name of track bar itself
# second argument: specify name of window in which it is going to be displayed
# Then, we set range of numbers from zero to maximum 255
# Also, we have to pass an argument function that will do something with chosen number
# That's why we define here an empty functon (do_nothing) because we are interested only in numbers
# without any processing

# For minimum range
cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, do_nothing)

# For maximum range
cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, do_nothing)

# Reading images with OpenCV library
# In this way image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
# ~= read input image by opencv function that give us BGR image in form of numpy array
image_BGR = cv2.imread('objects-to-detect.jpg')
# Resizing image in order to use smaller windows
# ~= we scale down it with resize() function in order to use more convenient smaller window
image_BGR = cv2.resize(image_BGR, (600, 426))

# Showing Original Image
# Giving name to the window with Original Image
# And specifying that window is resizable
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# we show original image by function imshow() in which we pass name of the window and image itself as arguments
cv2.imshow('Original Image', image_BGR)
# When we create opencv window we can pass argument WINDOW_NORMAL that will give us an opportunity to change
# size of the window when it's opened

# Converting Original Image to HSV
# ~= Now we convert input image from BGR to HSV color space
# HSV stands for [Hue Saturation Value]
# --> With this color space, it is very easy to detect object based on its color
# By changing value property, we keep needed colors and omit unneeded colors
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

# Showing HSV Image
# Giving name to the window with HSV Image
# And specifying that window is resizable
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.imshow('HSV Image', image_HSV)

# while True:
#     if cv2.waitKey(0):
#         break

# Defining loop for choosing right Colours for the Mask
# ~= Now we start while loop where we define variables to get 6 numbers for color mask according to
# current position of the appropriate track bar
while True:
    # Defining variables for saving values of the Track Bars
    # For minimum range
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    # For maximum range
    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # Implementing Mask with chosen colours from Track Bars to HSV Image
    # Defining lower bounds and upper bounds for thresholding
    # ~= When we have all 6 numbers, we implement thresholding operation by opencv function inRange()
    # where we pass our HSV image and range of pixel values, from minimum to maximum
    mask = cv2.inRange(image_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # Showing Binary Image with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    # ~= Then, we show resulted binary image in window and repeat this operation in loop
    # while we find right 6 numbers for color mask
    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask', mask)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()


# Printing final chosen Mask numbers
print('min_blue, min_green, min_red = {0}, {1}, {2}'.format(min_blue, min_green,
                                                            min_red))
print('max_blue, max_green, max_red = {0}, {1}, {2}'.format(max_blue, max_green,
                                                            max_red))


"""
Some comments

HSV (hue, saturation, value) colour-space is a model
that is very useful in segmenting objects based on the colour.

With OpenCV function cv2.inRange() we perform basic thresholding operation
to detect an object based on the range of pixel values in the HSV colour-space.
"""
