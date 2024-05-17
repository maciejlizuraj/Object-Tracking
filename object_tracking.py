import cv2 as cv
import numpy as np

MIN_MATCH_COUNT = 10
MIN_CONTOUR_AREA = 100
X_SIZE = 60
Y_SIZE = 60
VIDEO_FILE = "resources/video.mp4"
IMAGE_FILE = "resources/image.jpg"

rect_coord = []
mean_coord = []


def find_frame(frame, sift, img_find, kp2):
    global rect_coord
    global mean_coord
    img_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img_frame, None)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des2, des1, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if True:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        lowest_x = 0
        lowest_y = 0
        highest_x = lowest_x + X_SIZE
        highest_y = lowest_y + Y_SIZE
        if M is not None:
            matchesMask = mask.ravel().tolist()
            h, w = img_frame.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            reshaped = dst_pts.reshape(-1, 2)

            all_x = reshaped[:, 0].tolist()
            all_y = reshaped[:, 1].tolist()

            max_points = 0

            for x in all_x:
                for y in all_y:

                    lowest_x_tmp = int(x)
                    lowest_y_tmp = int(y)
                    highest_x_tmp = lowest_x_tmp + X_SIZE
                    highest_y_tmp = lowest_y_tmp + Y_SIZE

                    condition1 = (reshaped[:, 0] < lowest_x_tmp) | (reshaped[:, 0] > highest_x_tmp)
                    condition2 = (reshaped[:, 1] < lowest_y_tmp) | (reshaped[:, 1] > highest_y_tmp)
                    mask = ~condition1 & ~condition2
                    filtered_tmp = reshaped[mask]

                    if filtered_tmp.size > max_points:
                        max_points = filtered_tmp.size
                        lowest_x = lowest_x_tmp
                        lowest_y = lowest_y_tmp
                        highest_x = highest_x_tmp
                        highest_y = highest_y_tmp

            rect_coord = [lowest_x, lowest_y, highest_x, highest_y]

            if len(reshaped) > 0:
                mean_x = int(np.mean(reshaped[:, 0]))
                mean_y = int(np.mean(reshaped[:, 1]))

                mean_coord = [mean_x, mean_y]




        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        if len(rect_coord) > 0:
            cv.rectangle(frame, (rect_coord[0], rect_coord[1]), (rect_coord[2], rect_coord[3]),
                         (255, 0, 0),
                         3)

        if len(mean_coord) > 0:
            cv.circle(frame, (mean_coord[0], mean_coord[1]), 10, 255, -1)

        img3 = cv.drawMatches(img_find, kp2, frame, kp1, good, frame, **draw_params)

    return img3


if __name__ == '__main__':
    cap = cv.VideoCapture(VIDEO_FILE)
    img_to_find = cv.imread(IMAGE_FILE, cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img_to_find, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    if (cap.isOpened() == False):
        print("Error opening video file")

    while cap.isOpened():
        cv.startWindowThread()
        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        det_frame = find_frame(frame, sift, img_to_find, kp2)
        cv.imshow('frame, click q to quit', det_frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
