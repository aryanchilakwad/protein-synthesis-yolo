from pathlib import Path
import random
import shutil


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

IMAGE_SOURCE = Path("dataset/images/all")
LABEL_SOURCE = Path("dataset/labels/all")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42


# ---------------------------------------------------------
# OUTPUT FOLDERS
# ---------------------------------------------------------

splits = ["train", "val", "test"]

for split in splits:
    (Path("dataset/images") / split).mkdir(parents=True, exist_ok=True)
    (Path("dataset/labels") / split).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# GET IMAGES
# ---------------------------------------------------------

images = sorted(IMAGE_SOURCE.glob("*.png"))

print(f"Found {len(images)} images.")


# ---------------------------------------------------------
# SHUFFLE
# ---------------------------------------------------------

random.seed(SEED)
random.shuffle(images)


# ---------------------------------------------------------
# CALCULATE SPLIT SIZES
# ---------------------------------------------------------

total = len(images)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]


# ---------------------------------------------------------
# COPY FILES
# ---------------------------------------------------------

def copy_split(image_list, split_name):

    copied = 0

    for image_path in image_list:

        label_path = LABEL_SOURCE / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        destination_image = Path("dataset/images") / split_name / image_path.name
        destination_label = Path("dataset/labels") / split_name / label_path.name

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

        copied += 1

    print(f"{split_name}: {copied} images")


# ---------------------------------------------------------
# RUN SPLIT
# ---------------------------------------------------------

print("\nCreating dataset split...\n")

copy_split(train_images, "train")
copy_split(val_images, "val")
copy_split(test_images, "test")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n----------------------------------------")
print("DATASET SPLIT COMPLETE")
print("----------------------------------------")
print(f"Training   : {len(train_images)} images")
print(f"Validation : {len(val_images)} images")
print(f"Testing    : {len(test_images)} images")
print("----------------------------------------")