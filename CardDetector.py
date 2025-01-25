############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras (Modified to remove short-term memory and do basic detection)
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a (now USB) camera feed on Mac. Also labels cards as "My Card"
# vs. "Table Card" based on their y-position.

import cv2
import numpy as np
import time
import os
import math

import Cards
import VideoStream

### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER
## the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

## IF USING USB CAMERA (INSTEAD OF PICAMERA), SET PiOrUSB=2:
videostream = VideoStream.VideoStream(
    (IM_WIDTH, IM_HEIGHT),
    FRAME_RATE,
    2,          # 2 = USB / built-in webcam
    0           # src=0 for default Mac camera
).start()

time.sleep(1)  # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
cam_quit = 0  # Loop control variable

while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)

    # Find and sort the contours of all cards in the image
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # Array of "cards" objects
    cards_in_frame = []
    k = 0

    if len(cnts_sort) != 0:
        # For each contour detected:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:

                # Preprocess card, find corner points, center, etc.
                qCard = Cards.preprocess_card(cnts_sort[i], image)

                # cv2.imshow("Rank Debug", qCard.rank_img)
                # cv2.imshow("Suit Debug", qCard.suit_img)
                # cv2.waitKey(0)

                # Identify rank/suit using the training images
                qCard.best_rank_match, qCard.best_suit_match, \
                    qCard.rank_diff, qCard.suit_diff = Cards.match_card(
                        qCard, train_ranks, train_suits
                    )

                # Decide if this is "My Card" vs. "Table Card" based on y center
                if qCard.center[1] > (IM_HEIGHT * 0.6):
                    qCard.owner = "My Card"
                else:
                    qCard.owner = "Table Card"

                # Draw results on the image
                image = Cards.draw_results(image, qCard)

                cards_in_frame.append(qCard)
                k += 1

        # Draw card contours on the image (must do all at once)
        if len(cards_in_frame) != 0:
            temp_cnts = [c.contour for c in cards_in_frame]
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    # Draw frame rate in corner of image
    cv2.putText(
        image,
        "FPS: " + str(int(frame_rate_calc)),
        (10, 26),
        font,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )

    # Display the image
    cv2.imshow("Card Detector", image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Poll keyboard. If 'q' is pressed, exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

# Clean up
cv2.destroyAllWindows()
videostream.stop()
