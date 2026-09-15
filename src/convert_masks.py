from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

IMAGE_DIR = Path("dataset/raw/images/images/images")
MASK_DIR = Path("dataset/raw_images/masks/masks/masks")

OUTPUT_IMAGE_DIR = Path("dataset/images/all")
OUTPUT_LABEL_DIR = Path("dataset/labels/all")


# Create output folders if they don't exist
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# PROCESS ALL IMAGES
# ---------------------------------------------------------

image_files = sorted(IMAGE_DIR.glob("*.tif"))

print(f"Found {len(image_files)} images.")
print("Starting conversion...\n")

total_nuclei = 0
processed_images = 0


for image_path in image_files:

    # Find matching mask
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        print(f"WARNING: Mask not found for {image_path.name}")
        continue

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image = Image.open(image_path)

    # Convert 16-bit grayscale image to 8-bit
    image_array = np.array(image)

    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(
            image_array,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    # Save image as PNG
    output_image_path = OUTPUT_IMAGE_DIR / f"{image_path.stem}.png"

    Image.fromarray(image_array).save(output_image_path)


    # -----------------------------------------------------
    # READ MASK
    # -----------------------------------------------------

    mask = np.array(Image.open(mask_path))

    # The mask is RGBA.
    # The red channel contains the label values.
    mask = mask[:, :, 0]


    # -----------------------------------------------------
    # CREATE YOLO LABELS
    # -----------------------------------------------------

    height, width = mask.shape

    yolo_lines = []

    # BBBC039 uses non-zero mask values for nucleus regions.
    # We process each encoded value separately.
    for value in [1, 2, 3]:

        binary_mask = (mask == value).astype(np.uint8)

        # Find connected nucleus regions
        number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )

        # Ignore label 0 because it is the background
        for component_id in range(1, number_of_labels):

            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]
            w = stats[component_id, cv2.CC_STAT_WIDTH]
            h = stats[component_id, cv2.CC_STAT_HEIGHT]

            # Ignore extremely tiny regions
            if w < 3 or h < 3:
                continue

            # Convert bounding box to YOLO format
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            box_width = w / width
            box_height = h / height

            # Class 0 = nucleus
            yolo_lines.append(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

    # -----------------------------------------------------
    # SAVE LABEL FILE
    # -----------------------------------------------------

    output_label_path = OUTPUT_LABEL_DIR / f"{image_path.stem}.txt"

    with open(output_label_path, "w") as file:
        file.write("\n".join(yolo_lines))

    total_nuclei += len(yolo_lines)
    processed_images += 1

    print(
        f"{processed_images:3d}/{len(image_files)}  "
        f"{image_path.name}  ->  {len(yolo_lines)} nuclei"
    )


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

print("\n----------------------------------------")
print("CONVERSION COMPLETE")
print("----------------------------------------")
print(f"Images processed : {processed_images}")
print(f"Total nuclei     : {total_nuclei}")
print(f"Images saved     : {OUTPUT_IMAGE_DIR}")
print(f"Labels saved     : {OUTPUT_LABEL_DIR}")
print("----------------------------------------")