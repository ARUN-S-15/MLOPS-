import os
import json
import shutil
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def load_class_mapping(yaml_path=None):
    # Try to read DATASET/data.yaml automatically if available
    mapping = {}
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            names_block = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('names:'):
                    names_block = True
                    continue
                if names_block:
                    if not stripped:
                        break
                    # Expect lines like '0: Label'
                    if ':' in stripped:
                        key, val = stripped.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        if key.isdigit():
                            mapping[val.lower()] = int(key)
        except Exception:
            mapping = {}

    # Fallback mapping (user-provided example)
    if not mapping:
        mapping = {
            'person': 0,
            'ladle': 1,
            'forklift': 2,
        }

    return mapping


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax


def convert_labelme_to_yolo(json_file, output_dir, image_dir, class_map):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_height = data.get('imageHeight')
        image_width = data.get('imageWidth')
        if not image_height or not image_width:
            print(f"Missing image size in {json_file}")
            return False

        txt_file = os.path.splitext(os.path.basename(json_file))[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)

        lines_out = []
        for shape in data.get('shapes', []):
            st = shape.get('shape_type')
            if st not in ('polygon', 'rectangle', 'box', 'rect'):
                # skip unsupported shapes
                continue

            category = shape.get('label', '').lower()
            if category not in class_map:
                print(f"Unknown class '{category}' in {json_file}; skipping shape")
                continue

            class_id = class_map[category]

            # handle rectangle (two points) and polygon
            if st == 'rectangle' or st in ('box', 'rect'):
                pts = shape.get('points', [])
                if len(pts) < 2:
                    continue
                xmin, ymin = pts[0]
                xmax, ymax = pts[1]
            else:  # polygon
                polygon = shape.get('points', [])
                if not polygon:
                    continue
                xmin, ymin, xmax, ymax = polygon_to_bbox(polygon)

            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            lines_out.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Write label file only if we have shapes
        if lines_out:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(lines_out)

        # Copy image: prefer absolute path in JSON, otherwise look in provided image_dir
        image_file = data.get('imagePath', '')
        if os.path.isabs(image_file) and os.path.exists(image_file):
            src_image = image_file
        else:
            src_image = os.path.join(image_dir, image_file)

        dst_image = os.path.join(output_dir, os.path.basename(image_file))
        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)
        else:
            print(f"Image not found: {src_image}")

        return True
    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")
        return False


def create_dataset_split(input_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    random.shuffle(files)

    train_split = int(len(files) * split_ratio[0])
    val_split = int(len(files) * (split_ratio[0] + split_ratio[1]))

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for file in file_list:
            txt_src = os.path.join(input_dir, file)
            base = os.path.splitext(file)[0]
            img_src = None
            for e in exts:
                candidate = os.path.join(input_dir, base + e)
                if os.path.exists(candidate):
                    img_src = candidate
                    break

            txt_dst = os.path.join(split_dir, file)
            if img_src:
                img_dst = os.path.join(split_dir, os.path.basename(img_src))
            else:
                img_dst = None

            if os.path.exists(txt_src) and (img_src is None or os.path.exists(img_src)):
                shutil.copy(txt_src, txt_dst)
                if img_src:
                    shutil.copy(img_src, img_dst)
            else:
                print(f"Missing file: {txt_src} or image for {base}")


def process_files(json_files, output_dir, image_dir, class_map):
    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as executor:
        futures = [executor.submit(convert_labelme_to_yolo, str(json_file), output_dir, image_dir, class_map)
                   for json_file in json_files]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting files"):
            pass


def main():
    json_dir = r'D:\\Meltshop\\json_dir'
    image_dir = r'D:\\Meltshop\\image_dir'
    output_dir = r'D:\\Meltshop\\output_dir'
    final_dataset_dir = r'D:\\Meltshop\\final_dataset_dir'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_dataset_dir, exist_ok=True)

    # Try to load class mapping from the workspace DATASET/data.yaml
    yaml_path = r'E:\\AI Interview & Communication Analyzer\\DATASET\\data.yaml'
    class_map = load_class_mapping(yaml_path)

    print(f"Using class mapping: {class_map}")

    json_files = list(Path(json_dir).glob('*.json'))
    total_files = len(json_files)

    print(f"Total JSON files found: {total_files}")

    # Process files in batches of 15000
    batch_size = 15000
    for i in range(0, total_files, batch_size):
        batch = json_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(total_files-1)//batch_size + 1}")
        process_files(batch, output_dir, image_dir, class_map)

    print("Conversion completed. Starting dataset splitting...")
    create_dataset_split(output_dir, final_dataset_dir)
    print("Dataset splitting completed.")


if __name__ == "__main__":
    main()
