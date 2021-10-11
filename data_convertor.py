'''
Run the script as follows
    > python data_convertor.py --source 'DataOriginal' --target 'DataYolo'
'''
import argparse
import os
import random
import shutil
from tqdm import tqdm
import cv2
from PIL import Image


def train_conversion(source_root: str, target_root: str) -> None:
    '''
    Convert the Original Train Data to YOLO format
    Args:
        source_root (str): Directory containing original train dataset
        target_root (str): Directory to save the converted train dataset
    '''
    # Define source paths
    src_images_dir = os.path.join(source_root, 'Image')
    src_labels_dir = os.path.join(source_root, 'Annotation')
    
    # Define target paths
    tar_images_dir = os.path.join(target_root, 'images')
    tar_labels_dir = os.path.join(target_root, 'labels')

    # Create target directory structure
    os.makedirs(tar_images_dir, exist_ok=True)
    os.makedirs(tar_labels_dir, exist_ok=True)

    file_ctr = 0
    # Load Annotations and Images
    for img_subfolder, lbl_subfolder in tqdm(zip(os.listdir(src_images_dir), 
                                                 os.listdir(src_labels_dir)), 
                                             total=len(os.listdir(src_images_dir))):
        
        img_subfolder_path = os.path.join(src_images_dir, img_subfolder)
        lbl_subfolder_path = os.path.join(src_labels_dir, lbl_subfolder)
        # print(f'Currently in SubFolder {img_subfolder}')

        for img_file, lbl_file in tqdm(zip(os.listdir(img_subfolder_path), 
                                           os.listdir(lbl_subfolder_path)), 
                                       total=len(os.listdir(img_subfolder_path))):
            
            img_file_path = os.path.join(img_subfolder_path, img_file)
            lbl_file_path = os.path.join(lbl_subfolder_path, lbl_file)

            # print(img_file_path, lbl_file_path)
            
            # load image
            image = Image.open(img_file_path)
            
            # load original annotations
            with open(lbl_file_path, 'r') as f:
                annotations = f.read().strip().split('\n')

            # print(annotations)
            
            # create an ordered list of coordinate tuples [(x1, y1), ...]
            coordinates = []
            for a in annotations:
                vals = a.split()[:-1]
                coord = []
                for i in range(4):
                    x = int(float(vals[i]))
                    y = int(float(vals[i+4]))
                    coord.append((x if x >= 0 else 0, y if y >= 0 else 0))

                coordinates.append(coord)
            

            # convert to yolo annotations
            # cls_id, centerX, centerY, width, height
            # (values are normalized)
            boxes = []
            yolo_annots = []

            for coord in coordinates:
                top, _, _, bottom = sorted(coord, key=lambda c: c[1])
                left, _, _, right = sorted(coord, key=lambda c: c[0])

                top_left = (left[0], top[1])
                bottom_right = (right[0], bottom[1])

                center_x, center_y = (top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2
                width, height = abs(top_left[0] - bottom_right[0]), abs(top_left[1] - bottom_right[1])

                boxes.append((top_left, bottom_right))

                yolo_annots.append((0, center_x / image.width, center_y / image.height, width / image.width, height / image.height))

            # converting the annotations into string
            yolo_annotation_string = []
            
            for y in yolo_annots:
                yolo_annotation_string.append(" ".join(list(map(lambda x: str(round(x, 3)), y))))
            
            yolo_annotation_string = "\n".join(yolo_annotation_string)
            
            # write the annotations and image to the target dir
            with open(os.path.join(tar_labels_dir, f"{file_ctr}.txt"), 'w') as f:
                f.write(yolo_annotation_string)
            
            image.save(os.path.join(tar_images_dir, f"{file_ctr}.jpg"))
            
            file_ctr += 1


def test_conversion(source_root: str, target_root: str) -> None:
    '''
    Convert the Test Data to YOLO format
    Args:
        source_root (str): Directory containing original test dataset
        target_root (str): Directory to save the converted test dataset
    '''
    # Define source paths
    src_images_dir = os.path.join(source_root, 'Image')
    src_labels_dir = os.path.join(source_root, 'Annotation')
    
    # Define target paths
    tar_images_dir = os.path.join(target_root, 'images')
    tar_labels_dir = os.path.join(target_root, 'labels')

    # Create target directory structure
    os.makedirs(tar_images_dir, exist_ok=True)
    os.makedirs(tar_labels_dir, exist_ok=True)

    file_ctr = 0
    # Load Annotations and Images
    for img_file, lbl_file in tqdm(zip(os.listdir(src_images_dir), 
                                       os.listdir(src_labels_dir)), 
                                   total=len(os.listdir(src_images_dir))):

        img_file_path = os.path.join(src_images_dir, img_file)
        lbl_file_path = os.path.join(src_labels_dir, lbl_file)

        # print(img_file_path, lbl_file_path)

        # load image
        image = Image.open(img_file_path)

        # load original annotations
        with open(lbl_file_path, 'r') as f:
            annotations = f.read().strip().split('\n')

        # print(annotations)

        # create an ordered list of coordinate tuples [(x1, y1), ...]
        coordinates = []
        for a in annotations:
            vals = a.split(',')[:-1]
            coord = []
            for i in range(4):
                x = int(float(vals[i]))
                y = int(float(vals[i+4]))
                coord.append((x if x >= 0 else 0, y if y >= 0 else 0))

            coordinates.append(coord)


        # convert to yolo annotations
        # cls_id, centerX, centerY, width, height
        # (values are normalized)
        boxes = []
        yolo_annots = []

        for coord in coordinates:
            top, _, _, bottom = sorted(coord, key=lambda c: c[1])
            left, _, _, right = sorted(coord, key=lambda c: c[0])

            top_left = (left[0], top[1])
            bottom_right = (right[0], bottom[1])

            center_x, center_y = (top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2
            width, height = abs(top_left[0] - bottom_right[0]), abs(top_left[1] - bottom_right[1])

            boxes.append((top_left, bottom_right))

            yolo_annots.append((0, center_x / image.width, center_y / image.height, width / image.width, height / image.height))

        # converting the annotations into string
        yolo_annotation_string = []

        for y in yolo_annots:
            yolo_annotation_string.append(" ".join(list(map(lambda x: str(round(x, 3)), y))))

        yolo_annotation_string = "\n".join(yolo_annotation_string)

        # write the annotations and image to the target dir
        with open(os.path.join(tar_labels_dir, f"{file_ctr}.txt"), 'w') as f:
            f.write(yolo_annotation_string)

        image.save(os.path.join(tar_images_dir, f"{file_ctr}.jpg"))

        file_ctr += 1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default='DataOriginal', help='Source directory of original dataset.')
    parser.add_argument('-t', '--target', type=str, default='DataOriginal', help='Target directory to save converted yolo dataset.')
    parser.add_argument('-v', '--n_val', type=int, default=10000, help="Number of validation samples to fetch from train.")
    args = parser.parse_args()
    
    # Convert train data
    train_conversion(args.source, args.target)
    
    # Sample validation data from train data
    train_len = len(os.listdir(f'{args.target}/Train/labels/'))
    val_selected = random.choices(list(range(train_len)), k=args.n_val)
    
    # Move the selected val images and labels from train
    for n in tqdm(val_selected):
        src_image = f"{args.target}/Train/images/{n}.jpg"
        src_label = f"{args.target}/Train/labels/{n}.txt"
        if os.path.isfile(src_image) and os.path.isfile(src_label):
            shutil.move(src_image, f"{args.target}/Val/images/{n}.jpg")
            shutil.move(src_label, f"{args.target}/Val/labels/{n}.txt")
    
    # Convert test data
    test_conversion(args.source, args.target)
    
    print('Data Converted to YOLO format Successfully!!')
    
    
