import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def convert_bbox(size, box):
    """
    Convert VOC bounding box format to YOLO format.

    VOC format:
        xmin, ymin, xmax, ymax

    YOLO format:
        x_center, y_center, width, height
    normalized to [0, 1].
    """

    img_width, img_height = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return x_center, y_center, width, height


def convert_annotation(xml_path, label_path):
    """
    Convert one Pascal VOC XML annotation file to YOLO txt format.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        if class_name not in VOC_CLASSES:
            continue

        class_id = VOC_CLASSES.index(class_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center, y_center, width, height = convert_bbox(
            (img_width, img_height),
            (xmin, ymin, xmax, ymax)
        )

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def collect_voc_items(vocdevkit_dir):
    """
    Collect image and annotation paths from VOC2007 and VOC2012.
    """

    vocdevkit_dir = Path(vocdevkit_dir)
    items = []

    for year in ["VOC2007", "VOC2012"]:
        voc_dir = vocdevkit_dir / year

        if not voc_dir.exists():
            continue

        images_dir = voc_dir / "JPEGImages"
        annotations_dir = voc_dir / "Annotations"

        for xml_path in annotations_dir.glob("*.xml"):
            image_name = xml_path.stem + ".jpg"
            image_path = images_dir / image_name

            if image_path.exists():
                items.append((image_path, xml_path))

    return items


def prepare_dataset(
    vocdevkit_dir="datasets/VOCdevkit",
    output_dir="datasets/VOC",
    val_ratio=0.1,
    random_seed=42
):
    """
    Prepare PASCAL VOC dataset in YOLO format.

    Output structure:
        datasets/VOC/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    """

    random.seed(random_seed)

    vocdevkit_dir = Path(vocdevkit_dir)
    output_dir = Path(output_dir)

    all_items = collect_voc_items(vocdevkit_dir)

    if len(all_items) == 0:
        raise RuntimeError(
            "No images were found. Check the path to VOCdevkit directory."
        )

    random.shuffle(all_items)

    val_count = int(len(all_items) * val_ratio)

    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    # In this simple version, test split is not separated automatically.
    # It can be created manually or by changing the split logic.
    splits = {
        "train": train_items,
        "val": val_items,
    }

    for split_name, split_items in splits.items():
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_path, xml_path in split_items:
            dst_image = images_out / image_path.name
            dst_label = labels_out / f"{xml_path.stem}.txt"

            shutil.copy(image_path, dst_image)
            convert_annotation(xml_path, dst_label)

    print("Dataset preparation completed.")
    print(f"Total images: {len(all_items)}")
    print(f"Train images: {len(train_items)}")
    print(f"Validation images: {len(val_items)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    prepare_dataset()
