

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-1
Quick Win - Step 2: Simple Object Detection by thresholding with mask
File: detecting-object.py
"""


# Detecting Object with chosen Colour Mask
#
# Algorithm:
# Reading RGB image --> Converting to HSV --> Implementing Mask -->
# --> Finding Contour Points --> Extracting Rectangle Coordinates -->
# --> Drawing Bounding Box --> Putting Label
#
# Result:
# Window with Detected Object, Bounding Box and Label in Real Time

# ~= Detect object by using mask that we have found in previous step
# Then, we will draw bounding box and label
# We will detect blue object with chosen color mask

# Importing needed library
import cv2


# Defining lower bounds and upper bounds of founded Mask
# ~= founded color mask numbers from previous step and by that we define range of minimum and maximum numbers
# to threshold with.
min_blue, min_green, min_red = 21, 222, 70
max_blue, max_green, max_red = 176, 255, 255

# Getting version of OpenCV that is currently used
# Converting string into the list by dot as separator and getting first number
# ~=And we get here version of opencv that is currently used (needed later).
# we get from the string list of elements by using split() and dot as separator.
v = cv2.__version__.split('.')[0]

# Defining object for reading video from camera
# ~= We are going to detect object in real time, so we need to read frames from camera by VideoCapture() function.
# It has argument which is number denoting index of camera.
# By 0, we usually get access to built-in camera.
# If you have additional camera try to use 1 instead.
#
camera = cv2.VideoCapture(0)


# Defining loop for catching frames
# ~= Now we start while loop and catch frame by frame in real time.
while True:
    # Capture frame-by-frame from camera
    # ~= We use function read() that gives us result of catching current frame 'True or False' - '_' and
    # BGR frame itself in form of numpy array.
    _, frame_BGR = camera.read()

    # Converting current frame to HSV
    # ~= As we did in previous step, we convert current frame into HSV color space
    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # Implementing Mask with founded colours from Track Bars to HSV Image
    # ~= implement thresholding with chosen color mask numbers.
    # As a result, we get binary image with black background and white object.
    mask = cv2.inRange(frame_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # Showing current frame with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with Mask', mask)

    # Finding Contours
    # Pay attention!
    # Different versions of OpenCV returns different number of parameters
    # when using function cv2.findContours()

    # In OpenCV version 3 function cv2.findContours() returns three parameters:
    # modified image, found Contours and hierarchy
    # All found Contours from current frame are stored in the list
    # Each individual Contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the Object
    # We are interested only in Contours

    # Checking if OpenCV version 3 is used
    # ~= With binary image, we can now implement function findContours() that gives us
    # list of found contours each in form of numpy array with boundary points of object.
    # We need to be careful here because different versions of OpenCV return different number of parameters
    # when using function findContours().
    # That's why we need to check firstly which version of OpenCV is currently used.
    # If it is third version, then function returns 3 parameters:
    # 1. modified image 2. list of found contours each in form of numpy array with boundary points of Object
    # 3. hierarchy
    # We are only interested in contours.
    if v == '3':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # In OpenCV version 4 function cv2.findContours() returns two parameters:
    # found Contours and hierarchy
    # All found Contours from current frame are stored in the list
    # Each individual Contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the Object
    # We are interested only in Contours
    # ~= If it is fourth version, then function returns only 1. found contours each in form of numpy array with
    # boundary points of object and 2. hierarchy.
    # Again, we are interested only in contours.

    # Checking if OpenCV version 4 is used
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Finding the biggest Contour by sorting from biggest to smallest
    # ~= After we get all found contours, we sort them according to the size of contour area
    # and from biggest to smallest.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extracting Coordinates of the biggest Contour if any was found
    # ~= 'if contours': Now we check if in the current frame caught from camera any contour was found.
    # If yes, we extract coordinates of approximate rectangle around object by function boundingRect().
    if contours:
        # Getting rectangle coordinates and spatial size from the biggest Contour
        # Function cv2.boundingRect() is used to get an approximate rectangle
        # around the region of interest in the binary image after Contour was found
        # ~= We pass to function boundingRect() the first contour from the list which is the biggest one.
        # And we get left upper point: x_min, y_min and box_width & box_height.
        # We get everything we need to draw bounding box around found object.
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing Bounding Box on the current BGR frame
        # ~= We use function rectangle() and pass current BGR frame, because we want to draw rectangle on
        # the initial frame that is without any processing.
        # We pass left upper point x_min, y_min plus some shifting in order to make sure that
        # whole object is inside the box.
        # Then, we pass right bottom point: x_min + box_width, y_min + box_height.
        # Also, we pass color of the box line (0, 255, 0) which is green.
        # And we pass thickness of the line which is 3.
        cv2.rectangle(frame_BGR, (x_min - 15, y_min - 15),
                      (x_min + box_width + 15, y_min + box_height + 15),
                      (0, 255, 0), 3)

        # Preparing text for the Label
        label = 'Detected Object'

        # Putting text with Label on the current BGR frame
        # ~= We now need to put label.
        # We use function putText() where we pass the same initial BGR frame, label's text and coordinates
        # as starting point to type the text.
        # Also, we specify font style, font size, color which is the same with color of box line, and thickness
        # which is 2.
        # When current frame is ready to be shown, we go outside if condition and show current frame.
        cv2.putText(frame_BGR, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Showing current BGR frame with Detected Object
    # Giving name to the window with Detected Object
    # And specifying that window is resizable
    # ~= We do it outside the if condition in order anyway to show current frame
    # with or without bounding box around object.
    # It can be happened that we don't show any blue object, and it can't be found.
    # But, we still can see what is happening now.

    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', frame_BGR)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()


"""
Some comments

With OpenCV function cv2.findContours() we find 
contours of white object from black background.

There are three arguments in cv.findContours() function,
first one is source image, second is contour retrieval mode,
third is contour approximation method.


In OpenCV version 3 three parameters are returned:
modified image, the contours and hierarchy.
Further reading about Contours in OpenCV v3:
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html


In OpenCV version 4 two parameters are returned:
the contours and hierarchy.
Further reading about Contours in OpenCV v4:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html


Contours is a Python list of all the contours in the image.
Each individual contour is a Numpy array of (x,y) coordinates 
of boundary points of the object.

Contours can be explained simply as a curve joining all the 
continuous points (along the boundary), having same colour or intensity.
"""
