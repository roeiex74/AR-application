# ======= imports
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm


def ratio_test(matches, ratio_test=0.6) -> list:
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < ratio_test:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]
    # Return best match and best and second best match
    return good_match_arr, good_and_second_good_match_list


# ======= constants
# SET DEBUG FOR TESTING PURPOSES
ROOT_DIR = os.getcwd()
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv()

#################################
# Convert DEBUG to boolean
DEBUG = os.getenv("DEBUG", "FALSE").upper() in ("TRUE", "1")

# Get numeric values
KNN_K = int(os.getenv("KNN_K", 2))  # Default to 2 if not defined
RATIO_TEST = float(os.getenv("RATIO_TEST", 0.6))  # Convert to float
MIN_MATCHES = int(os.getenv("MIN_MATCHES", 4))  # Convert to int
FIGSIZE_WIDTH = int(os.getenv("FIGSIZE_WIDTH", 5))
FIGSIZE_HEIGHT = int(os.getenv("FIGSIZE_HEIGHT", 10))
figsize = (FIGSIZE_WIDTH, FIGSIZE_HEIGHT)

MEDIA_DIR = os.path.join(ROOT_DIR, "media")
TEMPLATE_IMAGE_PATH = os.path.join(MEDIA_DIR, "image_template.jpg")
# TEMPLATE_IMAGE_PATH = os.path.join(MEDIA_DIR, "test_template.DNG")
INPUT_VIDEO_PATH = os.path.join(MEDIA_DIR, "picture_input.mp4")

rgb_template = cv2.imread(TEMPLATE_IMAGE_PATH, cv2.COLOR_BGR2RGB)
gray_template = cv2.imread(TEMPLATE_IMAGE_PATH, cv2.COLOR_RGB2GRAY)
# === template image keypoint and descriptors
feature_extractor = cv2.SIFT_create()
# Find template keypoins and descriptors
kp_template, desc_template = feature_extractor.detectAndCompute(
    gray_template, None
)

if DEBUG:
    test = cv2.drawKeypoints(
        rgb_template,
        kp_template,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    plt.figure(figsize=figsize)
    plt.imshow(test)
    plt.title("keypoints")
    plt.show()

# Statless Brute force matcher init
bf = cv2.BFMatcher()
# ===== video input, output and metadata
input_cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = int(input_cap.get(cv2.CAP_PROP_FPS))
frame_count = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
# Used for creating a bounding box of the homography key features matching
frame_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
template_height, template_width = rgb_template.shape[:2]
template_corners = np.float32(
    [
        [template_width * 0.08, template_height * 0.04],
        [template_width * 0.95, template_height * 0.04],
        [template_width * 0.96, template_height * 0.93],
        [template_width * 0.12, template_height * 0.955],
    ]
).reshape(-1, 1, 2)
if DEBUG:
    draw_image = rgb_template.copy()
    corner_points_int = np.int32(template_corners)
    cv2.polylines(
        draw_image,
        [corner_points_int],
        color=(255, 0, 0),
        thickness=5,
        isClosed=True,
    )
    plt.figure(figsize=(20, 20))
    plt.imshow(draw_image)
    plt.title("Box")
    plt.show()

# Check if the file opened properly
if not input_cap.isOpened():
    print(f"Error: Could not open {INPUT_VIDEO_PATH}")
    exit()

# ========== run on all frames
# For debugging purposes
OUTPUT_VIDEO_PATH = os.path.join(MEDIA_DIR, "output_overlay.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Or "XVID", etc.
out_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height)
)
frame_index = 0


def pre_process_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = (5, 5)
    gaussian_blurred = cv2.GaussianBlur(gray_frame, kernel_size, 0)
    # Step 3: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_frame = clahe.apply(gaussian_blurred)
    return clahe_frame


while True:
    try:
        ret, frame = input_cap.read()
        if not ret:
            break  # no more frames or can't read

        # Start frame processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = pre_process_frame(frame)

        # ====== find keypoints matches of frame and template
        # we saw this in the SIFT notebook
        kp_frame, desc_frame = feature_extractor.detectAndCompute(
            gray_frame, None
        )

        if desc_frame is None:
            # If no keypoints found, just write the original frame out
            out_writer.write(frame)
            frame_index += 1
            continue

        # Find current matches - using KNN
        current_matches = bf.knnMatch(desc_template, desc_frame, k=KNN_K)

        # Apply ratio test
        good_match_arr, good_and_second_good_match_list = ratio_test(
            current_matches, RATIO_TEST
        )
        # Show matches every 500 frames (debug)
        if DEBUG and frame_index % 500 == 0:
            im_matches = cv2.drawMatchesKnn(
                rgb_template,
                kp_template,
                rgb_frame,
                kp_frame,
                good_and_second_good_match_list[0:30],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            plt.figure(figsize=figsize)
            plt.imshow(im_matches)
            plt.title("Keypoints Matches (Debug)")
            plt.show()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # Minimum 4 matches needed to compute a homography
        if len(good_match_arr) >= MIN_MATCHES:
            # Template (queryIdx) <-> Frame (trainIdx)
            good_kp_template = np.float32(
                [kp_template[m.queryIdx].pt for m in good_match_arr]
            )
            good_kp_frame = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_match_arr]
            )

            # Find homography that maps FRAME -> TEMPLATE
            H, mask = cv2.findHomography(
                good_kp_frame, good_kp_template, cv2.RANSAC, 3.0
            )

            if H is not None:
                # Valid homography found

                # Invert H to warp the template onto the frame
                H_inv = np.linalg.inv(H)

                # Project the corners on the computed inverse
                projected_corners = cv2.perspectiveTransform(
                    template_corners, H_inv
                )
                x_min, y_min = np.min(projected_corners, axis=0).ravel()
                x_max, y_max = np.max(projected_corners, axis=0).ravel()
                filtered_matches = []
                for i, match in enumerate(good_match_arr):
                    x, y = good_kp_frame[i]  # Keypoint location in the frame
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        filtered_matches.append(match)
                if len(filtered_matches) >= MIN_MATCHES:
                    good_kp_template_filtered = np.float32(
                        [kp_template[m.queryIdx].pt for m in filtered_matches]
                    )
                    good_kp_frame_filtered = np.float32(
                        [kp_frame[m.trainIdx].pt for m in filtered_matches]
                    )

                    # Recompute homography with filtered matches
                    H_filtered, mask_filtered = cv2.findHomography(
                        good_kp_frame_filtered,
                        good_kp_template_filtered,
                        cv2.RANSAC,
                        5.0,
                    )

                    if H_filtered is not None:
                        H_inv_filtered = np.linalg.inv(H_filtered)
                        warped_template = cv2.warpPerspective(
                            rgb_template,
                            H_inv_filtered,
                            (frame_width, frame_height),
                        )
                        # overlay (alpha blend)
                        alpha = 0.7
                        overlay_bgr = cv2.addWeighted(
                            frame, 1 - alpha, warped_template, alpha, 0
                        )
                        out_writer.write(overlay_bgr)
                        # If debug, show current overlay
                        if DEBUG and frame_index % 100 == 0:
                            inliers = [
                                m
                                for m, inlier in zip(good_match_arr, mask)
                                if inlier
                            ]
                            im_inliers = cv2.drawMatches(
                                rgb_template,
                                kp_template,
                                frame,
                                kp_frame,
                                inliers,
                                None,
                            )
                            plt.imshow(im_inliers)
                            plt.title("Found inliers using ransac")
                            plt.show()

                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                            cv2.imshow("Overlay Debug", overlay_bgr)

                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        continue  # Go to next frame
            # Not enough matches, write original frame
        out_writer.write(frame)
        frame_index += 1
    except Exception as e:
        out_writer.write(frame)

# ======== end all
# Release and close
input_cap.release()
cv2.destroyAllWindows()
