import cv2
import os
import time
import shutil
import pyautogui
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from mss import mss

def categorical_distance(col):
    def dist(v1, v2): return np.linalg.norm(v1-v2, ord=2)
    
    BLACK = dist(col, np.array([0,0,0]))
    GOLD = dist(col, np.array([10, 155, 195]))
    BROWN = dist(col, np.array([48, 115, 153]))
    GREEN = dist(col, np.array([24, 188, 78]))

    M = np.min([BLACK, GOLD, BROWN, GREEN])

    if M==BLACK: return 0
    if M in [GOLD, BROWN]: return 1
    return -1

def evaluate_target(image, background):
    x,y = image.shape[:2]
    x,y = x//2, y//2

    imCenter = image[x-15:x+15, y-15:y+15]
    backgroundCenter = background[x-15:x+15, y-15:y+15]

    diff = cv2.absdiff(imCenter, backgroundCenter).mean(axis=(0, 1))

    return categorical_distance(diff)

def split_frame(im, background):
    SX = 32
    SY = 13
    H = 292-SY
    W = 206-SX

    Xtau = 4
    YTau = 14
    deltaX = W+Xtau
    deltaY = H+YTau

    arr = np.zeros((15))
    for i in range(3):
        for j in range(5):
            v = im[SY+i*deltaY:SY+i*deltaY+H, SX+j*deltaX:SX+j*deltaX+W]
            b = background[SY+i*deltaY:SY+i*deltaY+H, SX+j*deltaX:SX+j*deltaX+W]
            click = evaluate_target(v, b)
            arr[i*5+j]=click
    return arr
    
def frame_decision(arr):
    zeros = np.sum(arr == 0)
    ones = np.sum(arr == 1)
    if zeros == 14 and ones == 1: return np.argmax(arr)
    return -1

def decide(frame, background):
    arr = split_frame(frame, background)
    return frame_decision(arr)

def movementFromDecision(inx):
    if inx < 0: return False

    SX = 32
    SY = 13
    H = 292-SY
    W = 206-SX
    Xtau = 4
    YTau = 14
    deltaX = W+Xtau
    deltaY = H+YTau

    SHIFTX = 450 + (inx%5)*deltaX + W//2
    SHIFTY = 300 + (inx//5)*deltaY + H//2

    pyautogui.moveTo(SHIFTX, SHIFTY)
    pyautogui.click()
    return True

def run(verbose=False):
    global CLICKTIME
    cnt = 0
    time.sleep(4)
    start = time.time()

    while time.time() - start < DURATION:
        while time.time() - CLICKTIME < CLICKWAIT: continue
        img = np.array(sct.grab({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }))

        img = img[300:1200, 450:1400]
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[-1] == 4 else img
        decision = decide(frame, background)
        move = movementFromDecision(decision)
        if move: 
            CLICKTIME = time.time()
            cnt += 1
        if verbose:
            print(cnt / (time.time() - start))
    
    return cnt


CLICKTIME = time.time()
CLICKWAIT = 0.052
pyautogui.PAUSE = 0.052
pyautogui.MINIMUM_DURATION = 0
DURATION = 61

background = cv2.imread('./imgs/grab_0025.png', cv2.IMREAD_COLOR)
sct = mss()
monitor = next((m for m in sct.monitors[1:] if m["width"] > 0 and m["height"] > 0), sct.monitors[0])
left = monitor["left"]
top = monitor["top"]
width = monitor["width"]
height = monitor["height"]
start = time.time()