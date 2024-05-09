from glob import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

# get a list of ball images and ground truth images
ball_frame_gt_files = glob("ball_frames/*GT.png")
ball_frame_files = [
    f for f in glob("ball_frames/*.png") if f not in ball_frame_gt_files
]

# sort the lists by frame number
ball_frame_gt_files.sort(
    key=lambda x: int(x.split("/")[-1].split("_")[0].split(".")[0].split("-")[1])
)
ball_frame_files.sort(
    key=lambda x: int(x.split("/")[-1].split("_")[0].split(".")[0].split("-")[1])
)


def estimate_motion_area(img1, img2):
    """
    Create a mask of the moving area between two images.
    based on the difference between the pixel values in the 2 images.
    """
    diff_img = cv2.absdiff(img1, img2)
    diff_gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(diff_gray_img, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=25)
    mask = (1 * (mask > 30)).astype(np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=15)

    return mask


def yellow_color_thresholding(img):
    """
    Create a mask of the yellow ball in the image by thresholding the color.
    Then filter by picking the largest connected component.
    """
    # color threshold
    yellow_ball_color = [233, 178, 32]
    yellow_ball_hsv_low = np.array([15, 100, 100])
    yellow_ball_hsv_high = np.array([30, 256, 256])

    yellow_img = np.zeros([1, 1, 3], dtype=np.uint8)
    yellow_img[0, 0] = yellow_ball_color
    # print(cv2.cvtColor(yellow_img, cv2.COLOR_RGB2HSV))

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    frame_threshold = cv2.inRange(hsv_img, yellow_ball_hsv_low, yellow_ball_hsv_high)

    # de-noise
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opened = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)

    # Close the wholes
    closed = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)

    # Find the biggest connected component
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity, cv2.CV_32S
    )

    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255

    return img2.astype(np.uint8)


def ball_detection_morph(img, edge_only=False):
    """
    Use edge detection and morphological operations to segment the balls.
    """
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((3, 3), np.uint8)

    if edge_only:
        return edges

    img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=20)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)

    return img


def get_mask(idx):
    """
    Get the mask of the ball in the image at index idx.
    Combines the mask from color segmentation and morphological operations.
    Filtered by the motion mask.
    """
    img = cv2.imread(ball_frame_files[idx])

    mask_1 = ball_detection_morph(img)
    mask_2 = yellow_color_thresholding(img)

    nxt_idx = idx + 1
    if nxt_idx == len(ball_frame_files):
        nxt_idx = idx - 1
    next_img = cv2.imread(ball_frame_files[nxt_idx])
    mask = cv2.bitwise_or(mask_1, mask_2)
    motion_mask = estimate_motion_area(img, next_img)
    mask = cv2.bitwise_and(mask, motion_mask)

    return mask


def get_gt_mask(idx):
    """
    Get the ground truth mask of the ball in the image at index idx.
    """
    gt_mask = cv2.imread(ball_frame_gt_files[idx])
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    gt_mask = gt_mask.astype(np.uint8)
    gt_mask = (1 * (gt_mask > 50)).astype(np.uint8)
    return gt_mask


if __name__ == "__main__":
    _, axs = plt.subplots(16, 4, figsize=(12, 36))
    axs = axs.flatten()

    DSs = []
    # loop over the images and display the masked images
    for idx, fn, fn_gt, ax in list(
        zip(range(len(ball_frame_files)), ball_frame_files, ball_frame_gt_files, axs)
    ):
        # read the image
        img = cv2.imread(fn)

        # get the mask of the ball
        mask = get_mask(idx)

        # get the ground truth mask of the ball
        gt_mask = get_gt_mask(idx)

        # calculate the Dice Similarity score
        DS = (
            2
            * np.sum(cv2.bitwise_and(gt_mask, mask))
            / (np.sum(mask) + np.sum(gt_mask))
        )
        DSs.append(DS)

        # apply the mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # display the masked image
        ax.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        ax.title.set_text(f"i: {idx}, DS: {DS:.2f}")

    # print the mean and std of the Dice Similarity scores
    print(f"DS mean: {np.mean(DSs):.03f}, {np.std(DSs):.03f}")

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
