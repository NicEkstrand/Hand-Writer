import cv2
from matplotlib import pyplot as plt

# Open facetime camera for video input
video = cv2.VideoCapture(0)

# Loop until the end of the video
while(video.isOpened()):

    frame = video.read()[1]

    # Display the resulting frame
    cv2.imshow("Smiles", frame)

    # Define q as the exit button
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break