import cv2
import pytesseract
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import pygame

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

def finger_tracker(image, detection_result):
    """
    Draws all the landmarks on the hand
    Args:
        image (Image): Image to draw on
        detection_result (HandLandmarkerResult): HandLandmarker detection results
    """
    # Get image details
    imageHeight, imageWidth = image.shape[:2]

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
        
        # Get the coordinate of just the index finger
        finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]

        # Map the coordinates back to screen dimensions
        pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)

        if pixelCoord:
            # Draw the circle around the index finger
            cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, (255, 0, 0), 5)
            pygame.draw.circle(screen, (redness, greenness, blueness), (pixelCoord[0] * 1.5, pixelCoord[1] * 1.5), thickness, width=20)

"""
Start of main function
"""
# Open facetime camera for video input
video = cv2.VideoCapture(0)

# Create the hand detector
base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
options = HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
detector = HandLandmarker.create_from_options(options)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
screen.fill((255, 255, 255))
pygame.display.set_caption("Finger Art")

running = True

thickness = int(input("How thick do you want your pen to be?(number)"))
redness = int(input("How much red do you want in your color?"))
greenness = int(input("How much green do you want in your color?"))
blueness = int(input("How much blue do you want in your color?"))

tracking = True

# Loop until the end of the video
while(video.isOpened() and running):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  
           running = False
        # If space key is pressed, the hand will stop being tracked
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                if tracking:
                    tracking = False
                else:
                    tracking = True
        
    frame = video.read()[1]

    # Convert it to an RGB image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # The image comes in mirrored - flip it
    image = cv2.flip(image, 1)



    if tracking:
        # Convert the image to a readable format and find the hands
        to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = detector.detect(to_detect)

        # Draw the hand landmarks
        finger_tracker(image, results)
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow("Finger Reader", image)
    pygame.display.flip()

    # Define q as the exit button
    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        if tracking:
            tracking = False
        else:
            tracking = True