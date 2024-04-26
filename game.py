import cv2
import pytesseract
from matplotlib import pyplot as plt


"""
Letter Detection Function written by Geeks for Geeks
Link: https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/#
Edited by Nic Ekstrand

"""

def letter_detection(image):
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
        file.close
                    


# Open facetime camera for video input
video = cv2.VideoCapture(0)

# Loop until the end of the video
while(video.isOpened()):

    frame = video.read()[1]
    image = letter_detection(frame)

    # Display the resulting frame
    cv2.imshow("Smiles", frame)

    # Define q as the exit button
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break