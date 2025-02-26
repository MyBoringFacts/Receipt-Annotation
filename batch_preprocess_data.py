import os
import json
import shutil
import base64
import cv2
import glob
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from sklearn.model_selection import train_test_split

def rename_images(folder="images", ext=".jpg"):
    files = sorted(os.listdir(folder))
    for i, filename in enumerate(files):
        src = os.path.join(folder, filename)
        new_name = os.path.join(folder, f"{i}{ext}")
        os.rename(src, new_name)
        print(f"Renamed '{filename}' to '{i}{ext}'")

def process_images_with_azure(input_folder="images", output_folder="images_json", endpoint="DUMMY_ENDPOINT", key="DUMMY_KEY"):
    os.makedirs(output_folder, exist_ok=True)
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path} ...")
            with open(file_path, "rb") as file_stream:
                poller = client.begin_analyze_document("prebuilt-receipt", body=file_stream, content_type="application/octet-stream")
                result = poller.result()
            json_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".json")
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(result.as_dict(), f, indent=4)
            print(f"Saved result to {json_file_path}")

def polygon_to_yolo(polygon, image_width, image_height):
    xs, ys = polygon[0::2], polygon[1::2]
    x_center, y_center = (min(xs) + max(xs)) / 2.0 / image_width, (min(ys) + max(ys)) / 2.0 / image_height
    box_width, box_height = (max(xs) - min(xs)) / image_width, (max(ys) - min(ys)) / image_height
    return x_center, y_center, box_width, box_height

def process_json_folder(json_folder="images_json", output_folder="yolo_annotations"):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.lower().endswith(".json")]
    for json_file in json_files:
        process_json_file(json_file, output_folder)

def split_dataset(original_images_folder="images", yolo_annotations_folder="yolo_annotations", dataset_dir="yolo_dataset"):
    os.makedirs(dataset_dir, exist_ok=True)
    images_train_dir, images_val_dir = os.path.join(dataset_dir, "images", "train"), os.path.join(dataset_dir, "images", "val")
    labels_train_dir, labels_val_dir = os.path.join(dataset_dir, "labels", "train"), os.path.join(dataset_dir, "labels", "val")
    
    for folder in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(folder, exist_ok=True)
    
    all_images = sorted([f for f in os.listdir(original_images_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    for img in train_imgs:
        shutil.copy(os.path.join(original_images_folder, img), os.path.join(images_train_dir, img))
        txt_name = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(yolo_annotations_folder, txt_name)):
            shutil.copy(os.path.join(yolo_annotations_folder, txt_name), os.path.join(labels_train_dir, txt_name))
    
    for img in val_imgs:
        shutil.copy(os.path.join(original_images_folder, img), os.path.join(images_val_dir, img))
        txt_name = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(yolo_annotations_folder, txt_name)):
            shutil.copy(os.path.join(yolo_annotations_folder, txt_name), os.path.join(labels_val_dir, txt_name))
    
    print("Dataset folder structure created!")

def convert_yolo_to_labelme(yolo_images_dir, yolo_labels_dir, labelme_output_dir, classes):
    os.makedirs(labelme_output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(yolo_images_dir, "*.jpg"))
    
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        yolo_label_path = os.path.join(yolo_labels_dir, base_name + ".txt")
        
        if not os.path.exists(yolo_label_path):
            print(f"Annotation not found for {image_path}")
            continue
        
        labelme_data = yolo_to_labelme(image_path, yolo_label_path, classes)
        if labelme_data is None:
            continue
        
        json_path = os.path.join(labelme_output_dir, base_name + ".json")
        with open(json_path, "w") as f:
            json.dump(labelme_data, f, indent=4)
        print(f"Saved {json_path}")

def yolo_to_labelme(image_path, yolo_label_path, classes):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    height, width, _ = img.shape
    with open(image_path, "rb") as image_file:
        img_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    shapes = []
    with open(yolo_label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, w, h = map(float, parts)
            x_min, y_min = (x_center - w / 2) * width, (y_center - h / 2) * height
            x_max, y_max = (x_center + w / 2) * width, (y_center + h / 2) * height
            
            shape = {
                "label": classes[int(class_id)] if int(class_id) < len(classes) else str(class_id),
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)
    
    return {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": img_data,
        "imageHeight": height,
        "imageWidth": width
    }

if __name__ == "__main__":
    rename_images()
    split_dataset()
    convert_yolo_to_labelme("yolo_dataset/images/train", "yolo_dataset/labels/train", "labelme_data", ["SellerName", "SellerVAT", "DocumentDate", "ProductDescription", "Quantity", "Price", "TotalDue"])
    print("Processing complete!")
