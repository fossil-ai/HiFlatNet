import os
import cv2
import albumentations as A
from albumentations.augmentations.transforms import RGBShift, HueSaturationValue, RandomBrightnessContrast
from albumentations.augmentations.blur import GaussianBlur
from albumentations.augmentations.geometric import HorizontalFlip, VerticalFlip
import random
from tqdm import tqdm

input_dir = '/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/clear'
output_dir = '/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/clear_augmented_w_clear'
output_train_text_file = '/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/clear_aug_labels_train_w_clear.txt'
output_val_text_file = '/home/mohamf1-ll1/Desktop/School/robustml/HiFlatClassifier/data/clear_aug_labels_val_w_clear.txt'


def augment_and_save_images(input_dir, output_dir, num_augmentations=100):
    with open(output_train_text_file, 'w') as train_file, open(output_val_text_file, 'w') as val_file:
        if os.path.isdir(input_dir):
            os.makedirs(output_dir, exist_ok=True)

            for img_name in tqdm(os.listdir(input_dir), desc=f'Processing Clear Dataset'):
                img_path = os.path.join(input_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not read image {img_path}. Skipping.")
                    continue

                output_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nonaug.jpg")
                cv2.imwrite(output_img_path, image)
                if random.random() < 0.80:
                    train_file.write(f"{output_img_path} {1} {0}\n")
                else:
                    val_file.write(f"{output_img_path} {1} {0}\n")

                # Add clean image w/horizontal flip
                output_img_path_h = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nonaug_horizontal.jpg")
                augmentation_pipeline_h = A.Compose([
                    HorizontalFlip(p=1.0)
                ])
                augmented_h = augmentation_pipeline_h(image=image)
                aug_image_h = augmented_h['image']
                cv2.imwrite(output_img_path_h, aug_image_h)
                if random.random() < 0.80:
                    train_file.write(f"{output_img_path_h} {1} {0}\n")
                else:
                    val_file.write(f"{output_img_path_h} {1} {0}\n")


                # Add clean image w/ vertical flip
                output_img_path_v = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nonaug_vertical.jpg")
                augmentation_pipeline_v = A.Compose([
                    VerticalFlip(p=1.0)
                ])
                augmented_v = augmentation_pipeline_v(image=image)
                aug_image_v = augmented_v['image']
                cv2.imwrite(output_img_path_v, aug_image_v)
                if random.random() < 0.80:
                    train_file.write(f"{output_img_path_v} {1} {0}\n")
                else:
                    val_file.write(f"{output_img_path_v} {1} {0}\n")

                # Add clean image w/ vertical and horizontal flip
                output_img_path_vh = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_nonaug_vertical_horizontal.jpg")
                augmentation_pipeline_vh = A.Compose([
                    VerticalFlip(p=1.0),
                    HorizontalFlip(p=1.0)
                ])
                augmented_vh = augmentation_pipeline_vh(image=image)
                aug_image_vh = augmented_vh['image']
                cv2.imwrite(output_img_path_vh, aug_image_vh)
                if random.random() < 0.80:
                    train_file.write(f"{output_img_path_vh} {1} {0}\n")
                else:
                    val_file.write(f"{output_img_path_vh} {1} {0}\n")

                for i in range(num_augmentations):

                    blur = random.choice([i for i in range(3, 63) if i % 2 != 0])
                    hue = random.randint(-30, 30)

                    augmentation_pipeline = A.Compose([
                        GaussianBlur(blur_limit=(blur, blur + 2), p=1.0),
                        # RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=30, p=1.0),  # Apply a blue color shift
                        HueSaturationValue(hue_shift_limit=(hue, hue + 1), sat_shift_limit=0, val_shift_limit=0,
                                           p=1.0),  # Distort colors
                        # RandomBrightnessContrast(brightness_limit=(0.1, 0.01), contrast_limit=(0.2,0.21), p=1.0)  # Adjust brightness/contrast
                    ])

                    augmented = augmentation_pipeline(image=image)
                    aug_image = augmented['image']

                    output_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg")
                    cv2.imwrite(output_img_path, aug_image)

                    if random.random() < 0.80:
                        train_file.write(f"{output_img_path} {blur} {hue}\n")
                    else:
                        val_file.write(f"{output_img_path} {blur} {hue}\n")



# Run the augmentation and save process
augment_and_save_images(input_dir, output_dir)
