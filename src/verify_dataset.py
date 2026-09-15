from pathlib import Path
import cv2


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

IMAGE_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")
OUTPUT_DIR = Path("results/dataset_check")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CHECK FIRST 5 TRAINING IMAGES
# ---------------------------------------------------------

image_files = sorted(IMAGE_DIR.glob("*.png"))[:5]

print(f"Checking {len(image_files)} images...\n")


for image_path in image_files:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"Label missing: {image_path.name}")
        continue

    with open(label_path, "r") as file:
        lines = file.readlines()

    box_count = 0

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, center_x, center_y, box_width, box_height = map(
            float, parts
        )

        # Convert normalized YOLO coordinates to pixels
        center_x *= width
        center_y *= height
        box_width *= width
        box_height *= height

        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

        box_count += 1

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(str(output_path), image)

    print(
        f"{image_path.name} -> "
        f"{box_count} bounding boxes"
    )


print("\n----------------------------------------")
print("DATASET VERIFICATION COMPLETE")
print("----------------------------------------")
print(f"Checked images saved to: {OUTPUT_DIR}")