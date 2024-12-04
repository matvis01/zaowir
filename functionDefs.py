import cv2 as cv
import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import glob

def calibrate_camera_stereo(folder_path_left="./Mono 1/cam2", folder_path_right="./Mono 1/cam3",ext = "png", size_x = 10, size_y = 7, SINGLE_SQUARE = 28.67,show = False):
    images_left = glob.glob(folder_path_left + "/*." + ext)
    images_right = glob.glob(folder_path_right + "/*." + ext)
    objp = np.zeros((size_x * size_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:size_x, 0:size_y].T.reshape(-1, 2) * SINGLE_SQUARE

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    correct_images  = []

    if images_left and images_right:
        for i, img_path_left in enumerate(images_left):
            img_left = cv.imread(img_path_left)
            if img_left is None:
                print(f"Error: Couldn't load image {img_path_left}")
                continue

            gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            ret_left, corners_left = cv.findChessboardCorners(gray_left, (size_x, size_y), None)

            if ret_left:
                img_right = cv.imread(images_right[i])
                if img_right is None:
                    print(f"Error: Couldn't load image {images_right[i]}")
                    continue

                gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
                ret_right, corners_right = cv.findChessboardCorners(gray_right, (size_x, size_y), None)

                if ret_right:
                    print(f"found corners: {img_path_left}")
                    correct_images.append(img_path_left)
                    objpoints.append(objp)

                    # Refine corner points
                    corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                    corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)

                    # Optional: Visualize corners
                    if show:
                        cv.drawChessboardCorners(img_left, (size_x, size_y), corners_left, ret_left)
                        cv.imshow('Left Image', img_left)
                        cv.waitKey(1)

                        cv.drawChessboardCorners(img_right, (size_x, size_y), corners_right, ret_right)
                        cv.imshow('Right Image', img_right)
                        cv.waitKey(1)

        cv.destroyAllWindows()

        # Perform calibration
        print("Start Calibration")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
        print("Calibration Left Done")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
        print("Calibration Right Done")
        #Calculate error
        error_left = 0
        error_right = 0
        for i in range(len(objpoints)):
            imgpoints_left2, _ = cv.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left)
            error_left += cv.norm(imgpoints_left[i], imgpoints_left2, cv.NORM_L2) / len(imgpoints_left2)
            imgpoints_right2, _ = cv.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right)
            error_right += cv.norm(imgpoints_right[i], imgpoints_right2, cv.NORM_L2) / len(imgpoints_right2)
        print("Error Calculation Done")
        # Stereo calibration
        image_size = gray_left.shape[::-1]
        stereocalibration_flags = cv.CALIB_FIX_INTRINSIC + cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                                                     mtx_left, dist_left, mtx_right, dist_right, image_size,
                                                                     criteria=criteria, flags=stereocalibration_flags)
        print("Stereo Calibration Done")
        # Compute baseline and FOV
        baseline = baseline_calculate(T) # Convert baseline to cm
        fov_left = calculate_fov(CM1, image_size)
        fov_right = calculate_fov(CM2, image_size)
        print(f"Baseline {baseline},Fov_Left: {fov_left},Fov_Right: {fov_right}")
        tmp_left = folder_path_left.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".","_")
        tmp_right = folder_path_right.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".","_")
        distortion_params = distorsion_coefficients(dist_left)
        save_to_json(distortion_params,f"distorsion_left_{tmp_left}.json")
        dictionary_left = {
            "Return": ret_left,
            "CameraMatrix": mtx_left.tolist(),  # Convert matrix to list
            "DistCoeffs": dist_left.tolist(),  # Convert distance coefficients to list
            "rvecs": [rvec.tolist() for rvec in rvecs_left],  # Convert each rvec to a list
            "tvecs": [tvec.tolist() for tvec in tvecs_left],  # Convert each tvec to a list
            "mean_error": error_left,
            "CorrectImages": correct_images
        }

        distortion_params = distorsion_coefficients(dist_right)
        save_to_json(distortion_params,f"distorsion_right_{tmp_right}.json")
        dictionary_right = {
            "Return": ret_right,
            "CameraMatrix": mtx_right.tolist(),  # Convert matrix to list
            "DistCoeffs": dist_right.tolist(),  # Convert distance coefficients to list
            "rvecs": [rvec.tolist() for rvec in rvecs_right],  # Convert each rvec to a list
            "tvecs": [tvec.tolist() for tvec in tvecs_right],  # Convert each tvec to a list
            "mean_error": error_right,
            "CorrectImages": correct_images
        }

        stereo_calibration_data = {
            "Return": ret,
            "baseline": baseline,
            "fov_left Horizontal": fov_left[0],
            "fov_left Vertical": fov_left[1],
            "fov_right Horizontal": fov_right[0],
            "fov_right Vertical": fov_right[1],
            "CameraMatrix1": CM1.tolist(),
            "DistCoeffs1": dist1.tolist(),
            "CameraMatrix2": CM2.tolist(),
            "DistCoeffs2": dist2.tolist(),
            "RotationMatrix": R.tolist(),
            "TranslationVector": T.tolist(),
            "EssentialMatrix": E.tolist(),
            "FundamentalMatrix": F.tolist(),
        }
        # Writing to sample.json
        save_to_json(dictionary_left,f"camera_stereo_calibration_left_{tmp_left}.json")
        save_to_json(dictionary_right,f"camera_stereo_calibration_right_{tmp_right}.json")
        save_to_json(stereo_calibration_data,f"camera_stereo_calibration_stereo_{tmp_left}_{tmp_right}.json")
        save_to_json({"CorrectImages": correct_images}, f"correctImagesStereo_{tmp_left}_{tmp_right}.json")

    else:
        print("No images found in the specified folder.")

def calibrate_camera_mono(folder_path="./Mono 1/cam2", size_x=10, size_y=7, SINGLE_SQUARE=28.67,ext = "png",show = False):
    # Load images from the folder
    images = glob.glob(folder_path + "/*." + ext)
    objp = np.zeros((size_x * size_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:size_x, 0:size_y].T.reshape(-1, 2) * SINGLE_SQUARE

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    correct_images = []

    if images:
        for i, img_path in enumerate(images):
            img = cv.imread(img_path)
            if img is None:
                print(f"Error: Couldn't load image {img_path}")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (size_x, size_y), None)

            if ret:
                print(f"found corners: {img_path}")
                correct_images.append(img_path)
                objpoints.append(objp)
                # Refine corner points
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Optional: Visualize corners
                if (show):
                    cv.drawChessboardCorners(img, (size_x, size_y), corners, ret)
                    cv.imshow('Image', img)
                    cv.waitKey(1)

        cv.destroyAllWindows()

        print("Start Calibration")
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Calibration Done")
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            mean_error += cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error /= len(objpoints)
        print("Error Calculation done")
        tmp = folder_path.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".","_")
        distortion_params = distorsion_coefficients(dist)
        save_to_json(distortion_params,f"distorsion_mono_{tmp}.json")
        # Compute field of view (FOV)
        fov = calculate_fov(mtx, gray.shape[::-1])
        print(f"Fov Calculation Done {fov}")

        # Save calibration data
        calibration_data = {
            "Return": ret,
            "CameraMatrix": mtx.tolist(),
            "DistCoeffs": dist.tolist(),
            "RotationVectors": [rvec.tolist() for rvec in rvecs],
            "TranslationVectors": [tvec.tolist() for tvec in tvecs],
            "MeanError": mean_error,
            "FieldOfView": list(fov),
        }
        save_to_json(calibration_data,f"camera_mono_calibration_{tmp}.json") #tu można zmienić nazwę pliku 
        save_to_json({"CorrectImages": correct_images},f"correctImagesMono_{tmp}.json")

        print(f"Calibration successful. Results saved to camera_mono_calibration_{tmp}.json")
    else:
        print("No images found in the specified folder.")

def calculate_fov(cameraMatrix: np.ndarray, imageSize: tuple[float, float]):
    fx = cameraMatrix[0, 0]  # Focal length in x-axis
    fy = cameraMatrix[1, 1]  # Focal length in y-axis
    width, height = imageSize

    fov_horizontal = 2 * np.arctan2(width,(2 *fx)) * (180 / np.pi)  # Convert radians to degrees
    fov_vertical = 2 * np.arctan2(height , (2 * fy)) * (180 / np.pi)  # Convert radians to degrees

    return fov_horizontal, fov_vertical

def baseline_calculate(T):
    baseline = np.round(np.linalg.norm(T) * 0.1, 2)  # Convert baseline to cm important on the test !!!!!!!
    return baseline

def distorsion_coefficients(dist_left):
    k1, k2, p1, p2, k3 = dist_left[0][0], dist_left[0][1], dist_left[0][2], dist_left[0][3], dist_left[0][4]
    distortion_params = {
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3
    }
    return distortion_params

def rectification_camera_stereo(json_file, image_file_left, image_file_right,alpha = 0.1,show = False):
    img_left = cv.imread(image_file_left)
    img_right = cv.imread(image_file_right)
    grayImg_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    with open(json_file, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    # Extracting camera parameters and stereo calibration data
    mtx1 = np.array(json_object["CameraMatrix1"])
    mtx2 = np.array(json_object["CameraMatrix2"])
    dist1 = np.array(json_object["DistCoeffs1"])
    dist2 = np.array(json_object["DistCoeffs2"])
    R = np.array(json_object["RotationMatrix"])
    T = np.array(json_object["TranslationVector"])
    pairname = f"{os.path.splitext(os.path.basename(image_file_left))[0]}"
    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        mtx1, dist1, mtx2, dist2, grayImg_left.shape[::-1], R, T, alpha=alpha
    )

    # Computing rectification maps
    map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, grayImg_left.shape[::-1], cv.CV_16SC2)
    map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, grayImg_left.shape[::-1], cv.CV_16SC2)

    # Remapping images to rectify
    rectified_img_left = cv.remap(img_left, map1x, map1y, cv.INTER_LINEAR)
    rectified_img_right = cv.remap(img_right, map2x, map2y, cv.INTER_LINEAR)
    if show:
        # Task 2.4: Compare interpolation methods
        compare_interpolation_times(img_left, map1x, map1y)

    # Task 2.5: Draw epipolar lines
    rectified_left_with_lines, rectified_right_with_lines = draw_epilines_aligned(rectified_img_left, rectified_img_right,20,roi1,roi2,10,5)
    rectified_pair = np.hstack((rectified_left_with_lines, rectified_right_with_lines))

    # Task 2.6: Export rectified images
    export_rectified_images(rectified_img_left, rectified_img_right,rectified_pair,f"rectified_{pairname}")

    print(f"Stereo Rectified images saved: rectified_{pairname}")

def rectification_camera_mono(json_file, image_file, alpha):
    img = cv.imread(image_file)
    grayImg_left = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    with open(json_file, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    pairname = f"{os.path.splitext(os.path.basename(image_file))[0]}"

    mtx = np.array(json_object["CameraMatrix"])
    dist = np.array(json_object["DistCoeffs"])
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, grayImg_left.shape[::-1], alpha, grayImg_left.shape[::-1])

    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, grayImg_left.shape[::-1], 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    export_rectified_image(dst,f"rectified_{pairname}")

    print(f"Mono Rectified images saved: rectified_{pairname}")

# Task 2.4: Evaluate interpolation methods
def compare_interpolation_times(img, mapx, mapy):
    """
    Compare the computation times and subjective results of different interpolation methods.
    """
    interpolation_methods = {
        "INTER_NEAREST": cv.INTER_NEAREST,
        "INTER_LINEAR": cv.INTER_LINEAR,
        "INTER_CUBIC": cv.INTER_CUBIC,
        "INTER_AREA": cv.INTER_AREA,
        "INTER_LANCZOS4": cv.INTER_LANCZOS4,
    }

    results = {}
    for name, method in interpolation_methods.items():
        start_time = time.time()
        remapped_img = cv.remap(img, mapx, mapy, method)
        duration = time.time() - start_time
        results[name] = (remapped_img, duration)
        print(f"Interpolation: {name}, Time: {duration:.4f} seconds")

    # Display results for visual comparison
    fig, axes = plt.subplots(1, len(interpolation_methods), figsize=(15, 5))
    for i, (name, (image, _)) in enumerate(results.items()):
        axes[i].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axes[i].set_title(name)
        axes[i].axis("off")
    plt.show()

    return results

# Task 2.5: Visualize epipolar lines
def draw_epilines_aligned(
        img_left: np.ndarray,
        img_right: np.ndarray,
        num_lines: int = 15,
        roi_left: tuple = None,
        roi_right: tuple = None,
        line_thickness: int = 2,
        roi_thickness: int = 2
) -> tuple[np.ndarray, np.ndarray]:

    img_left_with_lines = img_left.copy()
    img_right_with_lines = img_right.copy()

    # Use image dimensions or ROI if available
    height, width = img_left.shape[:2]
    roi_left = roi_left if roi_left else (0, 0, width, height)
    roi_right = roi_right if roi_right else (0, 0, width, height)

    # Generate y-coordinates for lines within the ROI
    y_start, y_end = roi_left[1], roi_left[1] + roi_left[3]
    y_coords = np.linspace(y_start, y_end - 1, num_lines).astype(int)

    # Draw horizontal epipolar lines
    for y in y_coords:
        color = (0, 0, 255)  # Red for lines
        cv.line(img_left_with_lines, (roi_left[0], y), (roi_left[0] + roi_left[2], y), color, line_thickness)
        cv.line(img_right_with_lines, (roi_right[0], y), (roi_right[0] + roi_right[2], y), color, line_thickness)

    # Draw ROI rectangles
    if roi_left:
        cv.rectangle(img_left_with_lines, (roi_left[0], roi_left[1]),
                     (roi_left[0] + roi_left[2], roi_left[1] + roi_left[3]), (0, 255, 0), roi_thickness)
    if roi_right:
        cv.rectangle(img_right_with_lines, (roi_right[0], roi_right[1]),
                     (roi_right[0] + roi_right[2], roi_right[1] + roi_right[3]), (0, 255, 0), roi_thickness)

    return img_left_with_lines, img_right_with_lines


# Task 2.6: Export rectified images
def export_rectified_images(image_left, image_right, image_pair, filename_prefix="rectified"):
    cv.imwrite(f"{filename_prefix}_image_left.png", image_left)
    cv.imwrite(f"{filename_prefix}_image_right.png", image_right)
    cv.imwrite(f"{filename_prefix}_image_pair.png", image_pair)

def export_rectified_image(image, filename_prefix="rectified"):
    cv.imwrite(f"{filename_prefix}_image.png", image)

def save_to_json(dictionary,file_name):
    with open(file_name, "w") as outfile:
        json.dump(dictionary, outfile,indent=3)
    print(f"Saved data to '{file_name}'")

if __name__ == "__main__":
    #Chessboard
    #calibrate_camera_stereo(folder_path_left=".\Chessboard\Chessboard\Stereo 2\cam1", folder_path_right=".\Chessboard\Chessboard\Stereo 2\cam4", size_x=10, size_y=7,ext="png")

    #calibrate_camera_mono(folder_path=".\Chessboard\Chessboard\Mono 1\cam1",size_x=10, size_y=7,ext="png") # prawdopodnie 2 razy puścić
    #calibrate_camera_mono(folder_path=".\Chessboard\Chessboard\Mono 1\cam4",size_x=10, size_y=7,ext="png") # prawdopodnie 2 razy puścić

    #Rectification
    #rectification_camera_mono("camera_mono_calibration___Chessboard_Chessboard_Mono_1_cam1.json",".\Chessboard\Chessboard\Mono 1\cam1\\60.png", 1)
    rectification_camera_mono("camera_mono_calibration___Chessboard_Chessboard_Mono_1_cam4.json",".\Chessboard\Chessboard\Mono 1\cam4\\55.png", 0.1)

    rectification_camera_stereo('camera_stereo_calibration_stereo___Chessboard_Chessboard_Stereo_2_cam1___Chessboard_Chessboard_Stereo_2_cam4.json', ".\Chessboard\Chessboard\Stereo 2\cam1\\55.png", ".\Chessboard\Chessboard\Stereo 2\cam4\\55.png",alpha=0)