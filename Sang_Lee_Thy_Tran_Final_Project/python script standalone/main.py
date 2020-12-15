import cv2
import numpy as np
import dlib
import sys


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


vid1 = (str(sys.argv[1]))
vid2 = (str(sys.argv[2]))
cap2 = cv2.VideoCapture(vid1)
cap1 = cv2.VideoCapture(vid2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap1.read()
    if frame is None:
        cap1 = cv2.VideoCapture(vid2)
        _, frame = cap1.read()

    _, frame2 = cap2.read()
    if frame2 is None:
        cap2 = cv2.VideoCapture(vid1)
        _, frame2 = cap2.read()

    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(f_gray)

    f_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    height2, width2, channels2 = frame2.shape
    frame2_new_face = np.zeros((height2, width2, channels2), np.uint8)

    # Feature detection and face extraction for face 1 --------
    dets = detector(f_gray,1)
    for k, d in enumerate(dets):
        landmarks = predictor(f_gray, d)
        landmarks_pts = []
        for n in range(0, 68):
            landmarks_pts.append((landmarks.part(n).x, landmarks.part(n).y))

    # Create convex hull from landmark points---------------------
    pts = np.array(landmarks_pts, np.int32)
    convexhull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, convexhull, 255)

    # Mask face and everything else is black
    face_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # Delaunay triangulation for face 1--------------------------------
    subdiv = cv2.Subdiv2D(cv2.boundingRect(convexhull))
    subdiv.insert(landmarks_pts)
    triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)

    indices_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = extract_index_nparray(np.where((pts == pt1).all(axis=1)))
        index_pt2 = extract_index_nparray(np.where((pts == pt2).all(axis=1)))
        index_pt3 = extract_index_nparray(np.where((pts == pt3).all(axis=1)))

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indices_triangles.append(triangle)

    # Feature detection and face extraction for face 2
    dets2 = detector(f_gray2,1)
    for k, d in enumerate(dets2):
        landmarks = predictor(f_gray2, d)
        landmarks_pts2 = []
        for n in range(0, 68):
            landmarks_pts2.append((landmarks.part(n).x, landmarks.part(n).y))

    # Create convex hull from landmark points for face 2
    pts2 = np.array(landmarks_pts2, np.int32)
    convexhull2 = cv2.convexHull(pts2)

    # Mask for lines artifacts
    lines_mask = np.zeros_like(f_gray)
    lines_new_face = np.zeros_like(frame2)

    # Delaunay triangulation for both faces ----------------------
    for tri_dex in indices_triangles:
        # Face 1 triangulation
        tr1_1 = landmarks_pts[tri_dex[0]]
        tr1_2 = landmarks_pts[tri_dex[1]]
        tr1_3 = landmarks_pts[tri_dex[2]]
        tri1 = np.array([tr1_1, tr1_2, tr1_3], np.int32)
        (x, y, w, h) = cv2.boundingRect(tri1)

        cropped_tri = frame[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        pts = np.array([[tr1_1[0] - x, tr1_1[1] - y],
                          [tr1_2[0] - x, tr1_2[1] - y],
                          [tr1_3[0] - x, tr1_3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, pts, 255)
        lines_space = cv2.bitwise_and(frame, frame, mask=lines_mask)

        # Face 2 Delaunay triangulation-------------------------------------------
        tr2_1 = landmarks_pts2[tri_dex[0]]
        tr2_2 = landmarks_pts2[tri_dex[1]]
        tr2_3 = landmarks_pts2[tri_dex[2]]
        tri2 = np.array([tr2_1, tr2_2, tr2_3], np.int32)
        (x, y, w, h) = cv2.boundingRect(tri2)

        pts2 = np.array([[tr2_1[0] - x, tr2_1[1] - y],
                           [tr2_2[0] - x, tr2_2[1] - y],
                           [tr2_3[0] - x, tr2_3[1] - y]], np.int32)

        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(cropped_tr2_mask, pts2, 255)

        # Blend triangles-------------------------------------
        pts = np.float32(pts)
        pts2 = np.float32(pts2)
        mat = cv2.getAffineTransform(pts, pts2)
        warped_tri= cv2.warpAffine(cropped_tri, mat, (w, h))
        warped_tri = cv2.bitwise_and(warped_tri, warped_tri, mask=cropped_tr2_mask)

        # Reconstruct the destination face
        frame2_new_area = frame2_new_face[y: y + h, x: x + w]
        frame2_new_gray = cv2.cvtColor(frame2_new_area, cv2.COLOR_BGR2GRAY)
        _, mask_tri = cv2.threshold(frame2_new_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_tri = cv2.bitwise_and(warped_tri, warped_tri, mask=mask_tri)

        frame2_new_area = cv2.add(frame2_new_area, warped_tri)
        frame2_new_face[y: y + h, x: x + w] = frame2_new_area

    # Face swapped (place 1st face onto the 2nd face)
    frame2_head_mask = cv2.fillConvexPoly(np.zeros_like(f_gray2), convexhull2, 255)
    frame2_face_mask = cv2.bitwise_not(frame2_head_mask)
    frame2_no_face = cv2.bitwise_and(frame2, frame2, mask=frame2_face_mask)
    result = cv2.add(frame2_no_face, frame2_new_face)

    # seamless cloning with OpenCV2
    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamless_output = cv2.seamlessClone(result, frame2, frame2_head_mask, center_face2, cv2.NORMAL_CLONE)
    cv2.imshow("Result", seamless_output)

    key = cv2.waitKey(1)
    if key == 0:
        break
