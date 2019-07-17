import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

# definitions
# get the screen resolution
app = wx.App(False)
screen_x, screen_y = wx.GetDisplaySize()
cam_w, cam_h = 340, 220

mouse = Controller()



# detect green colored obj
def object_detect(img):
    # HSV => Hue, Saturation, Value
    lower_bound = np.array([33, 80, 40])
    upper_bound = np.array([102, 255, 255])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_noised = cv2.inRange(img_hsv, lower_bound, upper_bound)
    # mask = open_close_operations(mask_noised)
    return open_close_operations(mask_noised)


# some optimization and noise removed
def open_close_operations(mask):
    kernel_open = np.ones((5,5))
    kernel_close = np.ones((20,20))

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    return mask_close


# initial definitions
cam = cv2.VideoCapture(0)  # camera init
font = cv2.FONT_HERSHEY_SIMPLEX
click_flag = 0

while cam.isOpened():
    isRead, frame = cam.read()
    frame = cv2.resize(frame,(340, 220))
    mask_obj_detected = object_detect(frame)
    # img_conts = draw_contours(mask_obj_detected, frame)

    conts, h = cv2.findContours(mask_obj_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = frame
    if (len(conts) == 2):  # mouse move only
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])
        # print('hello')
        if click_flag == 1:
            print(2)
            click_flag = 0
            mouse.release(Button.left)

        # draw rectangle over the objects
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
        # center coordinate of first obj
        cx1 = x1 + w1 / 2
        cy1 = y1 + h1 / 2
        # center coordinate of second obj
        cx2 = x2 + w2 / 2
        cy2 = y2 + h2 / 2
        # center coordinate between two objects
        cx = (cx1 + cx2) / 2
        cy = (cy1 + cy2) / 2
        # drawing the line connecting the centers
        cv2.line(img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)
        # drawing the point
        cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), 2)

        mouseLoc = (screen_x - (cx * screen_x / cam_w), cy * screen_y / cam_h)
        mouse.position = mouseLoc
        # while mouse.position != mouseLoc:
        #     pass
        # place mouse courser correspond to camera objects
        # ensure left btn is not pressed


    elif (len(conts) == 1):  # mouse click and move
        x, y, w, h = cv2.boundingRect(conts[0])
        if click_flag == 0:  # set on
            print(1)
            click_flag = 1  # set off
            mouse.press(Button.left)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w / 2
        cy = y + h / 2
        cv2.circle(img, (int(cx), int(cy)), int((w + h) / 4), (0, 0, 255), 2)
        mouseLoc = (screen_x - (cx * screen_x / cam_w), cy * screen_y / cam_h)
        mouse.position = mouseLoc
        # while mouse.position != mouseLoc:
        #     pass
    # return img
    cv2.imshow('cam', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

