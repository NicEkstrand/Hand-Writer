import cv2
import pytesseract
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils


"""
Letter Detection Function written by Geeks for Geeks
Link: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/#
Edited by Nic Ekstrand

"""

"""def letter_detection(image):
    # Change BRG image to Grayscale and RGB
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Establishing Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, kernel, iterations = 1)

    # Find Contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = image.copy()
    
    # A text file is created and flushed
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()
    
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        
        # Open the file in append mode
        file = open("recognized.txt", "a")
        
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        
        # Appending the text into file
        file.write(text)
        file.write("\n")
        
        # Close the file
        file.close"""

def finger_tracker(image, detection_result):
    """
    Draws all the landmarks on the hand
    Args:
        image (Image): Image to draw on
        detection_result (HandLandmarkerResult): HandLandmarker detection results
    """
    # Get a list of the landmarks
    hand_landmarks_list = detection_result.hand_landmarks
    
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Save the landmarks into a NormalizedLandmarkList
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        # Draw the landmarks on the hand
        DrawingUtil.draw_landmarks(image,
                                    hand_landmarks_proto,
                                    solutions.hands.HAND_CONNECTIONS,
                                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                                    solutions.drawing_styles.get_default_hand_connections_style())
        

            


# Open facetime camera for video input
video = cv2.VideoCapture(0)

# Create the hand detector
base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
options = HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
detector = HandLandmarker.create_from_options(options)

# Loop until the end of the video
while(video.isOpened()):

    frame = video.read()[1]

    # Convert it to an RGB image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

    # Convert the image to a readable format and find the hands
    to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(to_detect)

    # Draw the hand landmarks
    #finger_tracker(image, results)
    draw_line(image, results)

    # The image comes in mirrored - flip it
    image = cv2.flip(image, 1)

    # Display the resulting frame
    cv2.imshow("Smiles", image)

    # Define q as the exit button
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break