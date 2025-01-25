############## Playing Card Detector Functions ###############
#
# Author: Evan Juras (Modified to help detect cards on a wooden table, no memory)
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py

import numpy as np
import cv2
import time

### Constants ###

# Adaptive threshold offset. 60–100 is a good typical range for a well-lit environment.
BKG_THRESH = 80

# How much to reduce the corner threshold by when isolating suit/rank.
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# Thresholds for deciding unknown
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

# Card contour area thresholds
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 15000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""
    def __init__(self):
        self.contour = []           # Contour of card
        self.width, self.height = 0, 0
        self.corner_pts = []        # Corner points of card
        self.center = []            # Center point of card
        self.warp = []              # 200x300, flattened, grayed, blurred image
        self.rank_img = []          # Thresholded rank image
        self.suit_img = []          # Thresholded suit image
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0
        self.owner = "Unknown"      # Field to label the card's owner

class Train_ranks:
    """Structure to store information about train rank images."""
    def __init__(self):
        self.img = []
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""
    def __init__(self):
        self.img = []
        self.name = "Placeholder"


### Functions ###

def load_ranks(filepath):
    """Loads rank images from the specified directory."""
    train_ranks = []
    i = 0
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i += 1
    return train_ranks

def load_suits(filepath):
    """Loads suit images from the specified directory."""
    train_suits = []
    i = 0
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i += 1
    return train_suits

def preprocess_image(image):
    """
    Returns a grayed, blurred, and thresholded camera image.
    We do:
      - Convert to grayscale
      - Gaussian blur
      - Use an adaptive threshold approach via bkg_level + BKG_THRESH
      - Use THRESH_BINARY (white card on darker background)
      - Morphological 'close' to fill small holes
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    # Close to fill small holes
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return thresh

def find_cards(thresh_image):
    """
    Finds all card-sized contours in a thresholded camera image.
    Returns sorted contours and array of which are likely cards.
    """
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return [], []

    # Sort by contour area
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which contours are card-sized
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1)
            and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """
    Uses contour to find a flattened 200x300 image of the card,
    plus rank and suit images.
    Then we threshold using THRESH_BINARY, then invert with bitwise_not()
    so the rank/suit is white on black or black on white, as needed.
    """
    qCard = Query_card()
    qCard.contour = contour

    # Approximate corner points, bounding rect, center
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    x, y, w, h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Flatten the card into a top-down grayscale 200x300
    qCard.warp = flattener(image, pts, w, h)

    # Isolate the corner area (top-left corner)
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # Threshold for rank/suit
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH
    if thresh_level < 1:
        thresh_level = 1

    # We use THRESH_BINARY, then invert manually
    _, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY)

    # Manually invert the image instead of using THRESH_BINARY_INV
    query_thresh = cv2.bitwise_not(query_thresh)

    # Morphological close to fill in small holes or noise in corner image
    kernel = np.ones((3,3), np.uint8)
    query_thresh = cv2.morphologyEx(query_thresh, cv2.MORPH_CLOSE, kernel)

    # Split into top (rank) vs bottom (suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find largest contour in rank sub-image
    Qrank_cnts, _ = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find largest contour in suit sub-image
    Qsuit_cnts, _ = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)
    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

def match_card(qCard, train_ranks, train_suits):
    """
    Finds the best rank and suit match for the query card.
    Returns (best_rank, best_suit, rank_diff, suit_diff).
    """
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    if len(qCard.rank_img) != 0 and len(qCard.suit_img) != 0:
        # Compare query rank to each trained rank
        for Trank in train_ranks:
            diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img) / 255)
            if rank_diff < best_rank_match_diff:
                best_rank_match_diff = rank_diff
                best_rank_match_name = Trank.name

        # Compare query suit to each trained suit
        for Tsuit in train_suits:
            diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img) / 255)
            if suit_diff < best_suit_match_diff:
                best_suit_match_diff = suit_diff
                best_suit_match_name = Tsuit.name

    # If difference is too high, revert to "Unknown"
    if best_rank_match_diff > RANK_DIFF_MAX:
        best_rank_match_name = "Unknown"
    if best_suit_match_diff > SUIT_DIFF_MAX:
        best_suit_match_name = "Unknown"

    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff

def draw_results(image, qCard):
    """Draws the card's name, owner label, and center point on the image."""
    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match
    owner_str = qCard.owner

    line1 = f"{rank_name} of"
    line2 = f"{suit_name} ({owner_str})"

    cv2.putText(image, line1, (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, line1, (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(image, line2, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, line2, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """
    Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, resized, grayed image.
    """
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h:  # card is vertical
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl
    elif w >= 1.2*h:  # card is horizontal
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br
    else:
        # card is "diamond" oriented
        if pts[1][0][1] <= pts[3][0][1]:
            # tilt left
            temp_rect[0] = pts[1][0]
            temp_rect[1] = pts[0][0]
            temp_rect[2] = pts[3][0]
            temp_rect[3] = pts[2][0]
        else:
            # tilt right
            temp_rect[0] = pts[0][0]
            temp_rect[1] = pts[3][0]
            temp_rect[2] = pts[2][0]
            temp_rect[3] = pts[1][0]

    maxWidth = 200
    maxHeight = 300

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return warp
