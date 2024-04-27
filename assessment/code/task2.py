import scipy
import skimage
from glob import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

ball_frame_gt_files = glob('ball_frames/*GT.png')
ball_frame_files = [f for f in glob('ball_frames/*.png') if f not in ball_frame_gt_files]

ball_frame_gt_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0].split(".")[0].split("-")[1]))
ball_frame_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0].split(".")[0].split("-")[1]))

def get_gt_mask(idx):
    """
    return the ground truth mask
    """
    gt_mask = cv2.imread(ball_frame_gt_files[idx])
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    gt_mask = gt_mask.astype(np.uint8)
    gt_mask = (1 * (gt_mask > 50)).astype(np.uint8)
    return gt_mask

ball_features = {}
ball_names = ["tennis", "football", "rugby"]

# defines the angles at which to calculate the co-occurrence matrix
angles = np.array([0, 45, 90, 135])

# initialise the data structure
for ball_name in ball_names:
    ball_features[ball_name] = {
        "solidity": [],
        "circularity": [],
        "non_compactness": [],
        "eccentricity": [],
        "ASM": {c: {a: [] for a in angles} for c in ["r", "g", "b"]},
        "contrast": {c: {a: [] for a in angles} for c in ["r", "g", "b"]},
        "correlation": {c: {a: [] for a in angles} for c in ["r", "g", "b"]},
    }

# loop over all frames
for i in range(len(ball_frame_files)):
    img = cv2.imread(ball_frame_files[i])
    mask = get_gt_mask(i)

    # separate masks for the 3 ball types
    
    # connected component analysis
    label, num_features = scipy.ndimage.label(mask)

    # the smallest connected component is the tennis
    label_sizes = [np.sum(label == j + 1) for j in range(num_features)]
    tennis_label = np.argmin(label_sizes) + 1

    # The bottom most ball is rugby
    label_center_y = [
        scipy.ndimage.center_of_mass(mask, label, j + 1)[0] for j in range(num_features)
    ]
    rugby_label = np.argmax(label_center_y) + 1

    football_label = [j + 1 for j in range(num_features)]
    football_label.remove(tennis_label)
    football_label.remove(rugby_label)
    football_label = football_label[0]

    props = skimage.measure.regionprops(label, img, cache=True)
    solidities = [prop.solidity for prop in props]
    circularity = [(4 * np.pi * prop.area) / prop.perimeter**2 for prop in props]

    non_compactness = []
    for j in range(num_features):
        ball_mask = (label == j + 1).astype(np.uint8)
        mu = skimage.measure.moments_central(ball_mask)
        nc = 2 * np.pi * (mu[0, 2] + mu[2, 0]) / (mu[0, 0] ** 2)
        non_compactness.append(nc)

    eccentricity = [prop.eccentricity for prop in props]

    for ball_name, ball_label in zip(
        ball_names, [tennis_label, football_label, rugby_label]
    ):
        ball_features[ball_name]["solidity"].append(solidities[ball_label - 1])
        ball_features[ball_name]["circularity"].append(circularity[ball_label - 1])
        ball_features[ball_name]["non_compactness"].append(
            non_compactness[ball_label - 1]
        )
        ball_features[ball_name]["eccentricity"].append(eccentricity[ball_label - 1])

        ball_mask = (label == ball_label).astype(np.uint8)
        patch = cv2.bitwise_and(img, img, mask=ball_mask)

        for channel, channel_name in zip(range(3), ["b", "g", "r"]):
            g = skimage.feature.graycomatrix(
                patch[:, :, channel],
                distances=[1],
                angles=angles / 180 * np.pi,
                levels=256,
                normed=True,
                symmetric=True,
            )
            contrast = skimage.feature.graycoprops(g, "contrast")
            correlation = skimage.feature.graycoprops(g, "correlation")
            ASM = skimage.feature.graycoprops(g, "ASM")

            for i, a in enumerate(angles):
                ball_features[ball_name]["contrast"][channel_name][a].append(
                    contrast[0, i]
                )
                ball_features[ball_name]["correlation"][channel_name][a].append(
                    correlation[0, i]
                )
                ball_features[ball_name]["ASM"][channel_name][a].append(ASM[0, i])

for j, feature in enumerate(
    ["solidity", "circularity", "non_compactness", "eccentricity"]
):
    axes[j].hist(
        [ball_features[ball][feature] for ball in ball_names], label=ball_names
    )
    axes[j].set_title(f"{feature}")
    # axes[j].legend()

handles, labels = plt.gca().get_legend_handles_labels()
plt.figlegend(
    labels, loc="lower center", labelspacing=0.0, ncol=3, bbox_to_anchor=(0.5, -0.05)
)
plt.tight_layout()
plt.savefig("../report/figures/shape_features_hist.png")

import seaborn as sns

colors = sns.color_palette()

texture_features = {
    f: {c: [] for c in ["r", "g", "b"]} for f in ["ASM", "contrast", "correlation"]
}

for c in ["r", "g", "b"]:
    for feature in ["ASM", "contrast", "correlation"]:
        texture_features_data = []
        for ball_name in ball_names:
            features_means = []
            features_ranges = []
            for i in range(len(ball_frame_files)):
                feature_values = [
                    ball_features[ball_name][feature][c][a][i] for a in angles
                ]
                feature_mean = np.mean(feature_values)
                feature_range = np.max(feature_values) - np.min(feature_values)
                features_means.append(feature_mean)
                features_ranges.append(feature_range)
            texture_features_data.append((features_means, features_ranges))

        fig, axes = plt.subplots(1, 2)
        data = [x[0] for x in texture_features_data]
        axes[0].hist(
            data,
            label=ball_names,
            # color=[colors[3], colors[2], colors[0]],
            alpha=0.8
        )
        axes[0].set_xlabel("Mean")


        axes[1].hist(
            [x[1] for x in texture_features_data],
            label=ball_names,
            # color=[colors[3], colors[2], colors[0]],
            alpha=0.8
        )
        axes[1].set_xlabel("Range")

        fig.suptitle(f"Channel: {c} Feature: {feature}")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(
            labels, loc="lower center", labelspacing=0.0, ncol=3, bbox_to_anchor=(0.5, -0.05)
        )
        plt.tight_layout()
        plt.savefig(f"../report/figures/{c}_{feature}.png", bbox_inches='tight')