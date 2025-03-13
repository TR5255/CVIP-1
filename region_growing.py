import cv2
import numpy as np
from segmentation import Region_Growing
import pydicom
import os, sys, getopt
from sklearn.metrics import precision_score, recall_score, f1_score

DICOM_IMAGE_EXT = '.dcm'
OTHER_IMAGE_EXT = ['.jpg', '.png', '.jpeg']
IMAGE_PATH = "C:/Users/madha/OneDrive/Desktop/Interactive-Region-Growing-Segmentation/CHNCXR_0092_0.png"  # Default image path
CONN = 4
GROUND_TRUTH_PATH = "C:/Users/madha/OneDrive/Desktop/Interactive-Region-Growing-Segmentation/CHNCXR_0092_0_mask.png"  # Path to the ground truth mask

def run_region_growing_on_image(image_path, ground_truth_path):
    """
    1. Load Image in grayscale
    2. Segment the image using region growing and user seeds
    3. Display the result and ask for additional seeds
    4. Repeat Steps 2-3 until user presses Esc
    """
    image_data, image_name = get_image_data(image_path)
    ground_truth_data = get_image_data(ground_truth_path)[0]

    image_data = resize_image(image_data)
    image_data_post_smoothing = apply_gaussian_smoothing(image_data)

    segmentation = region_growing(image_data_post_smoothing, segmentation_name=image_name + " segmentation", neighbours=CONN)

    print(f"Segmentation shape: {segmentation.shape}")  # Debug check

    # Calculate and display evaluation metrics
    evaluate_metrics(segmentation, ground_truth_data)


def region_growing(image_data, neighbours, threshold=10, segmentation_name="Region Growing"):
    region_growing = Region_Growing(image_data, threshold=threshold, conn=neighbours)
    # Set Seeds
    region_growing.set_seeds()
    # Segmentation
    segmentation = region_growing.segment()
    # Display Segmentation
    region_growing.display_and_resegment(name=segmentation_name)
    return segmentation


def get_image_data(image_path):
    name, ext = os.path.splitext(image_path)
    if ext == DICOM_IMAGE_EXT:
        return (pydicom.read_file(image_path).pixel_array, name)
    elif ext in OTHER_IMAGE_EXT:
        return (cv2.imread(image_path, 0), name)
    else:
        print("Invalid Image Format. Supported Image Formats are: {}, {}".format(DICOM_IMAGE_EXT, OTHER_IMAGE_EXT))
        sys.exit()


def resize_image(image_data):
    if image_data.shape[0] > 1000:
        image_data = cv2.resize(image_data, (0, 0), fx=0.25, fy=0.25)
    if image_data.shape[0] > 500:
        image_data = cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5)
    return image_data


def apply_gaussian_smoothing(image_data, filter_size=3):
    return cv2.GaussianBlur(image_data, (filter_size, filter_size), 0)


import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score

def evaluate_metrics(segmentation, ground_truth):
    """
    Evaluate IoU, Dice score, Precision, Recall, and Pixel Accuracy.
    Assumes both segmentation and ground_truth are binary masks (0 and 255)
    where 255 represents the positive class (lungs) and 0 represents the background.
    """
    # Resize ground_truth to match segmentation shape
    ground_truth_resized = cv2.resize(ground_truth, (segmentation.shape[1], segmentation.shape[0]))

    # Convert both segmentation and ground truth to binary (0 and 1)
    ground_truth_binary = (ground_truth_resized == 255).astype(np.uint8).flatten()  # Lungs as 1, background as 0
    segmentation_binary = (segmentation == 255).astype(np.uint8).flatten()  # Lungs as 1, background as 0

    # Intersection and Union
    intersection = np.sum(segmentation_binary * ground_truth_binary)
    union = np.sum((segmentation_binary + ground_truth_binary) > 0)
    iou = intersection / union if union != 0 else 0
    print(f"IoU: {iou:.4f}")

    # Dice score
    dice_score = (2 * intersection) / (np.sum(segmentation_binary) + np.sum(ground_truth_binary)) if (np.sum(segmentation_binary) + np.sum(ground_truth_binary)) != 0 else 0
    print(f"Dice Score: {dice_score:.4f}")

    # Precision
    precision = precision_score(ground_truth_binary, segmentation_binary)
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(ground_truth_binary, segmentation_binary)
    print(f"Recall: {recall:.4f}")

    # Pixel Accuracy
    correct_pixels = np.sum(segmentation_binary == ground_truth_binary)
    total_pixels = len(segmentation_binary)
    pixel_accuracy = correct_pixels / total_pixels
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

    return {"IoU": iou, "Dice Score": dice_score, "Precision": precision, "Recall": recall, "Pixel Accuracy": pixel_accuracy}






def set_cmd_line_arguments():
    global IMAGE_PATH
    global CONN
    n_args = len(sys.argv)
    if n_args == 1:
        print("No image path specified. TERMINATING!!")
        sys.exit()
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "", ["image_path=", "conn="])
    for opt, arg in opts:
        if opt == "--image_path":
            IMAGE_PATH = arg
        elif opt == "--conn":
            print(arg)
            CONN = int(arg)
        else:
            print("Make sure to spell 'image_path' correctly")
            sys.exit()
    print("Image Path: {}".format(IMAGE_PATH))


if __name__ == "__main__":
    set_cmd_line_arguments()
    run_region_growing_on_image(IMAGE_PATH, GROUND_TRUTH_PATH)
