import os
import hashlib
import random
import shutil
from PIL import Image

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def prepare_test_dataset():
    # Paths
    base_dir = r"c:\Users\anush\OneDrive\Desktop\practicle DevOps\dogvison-classifier\dogvision-classifier\dog-detector"
    train_cat_dir = os.path.join(base_dir, "dataset", "cat")
    train_dog_dir = os.path.join(base_dir, "dataset", "dog")
    
    test_cat_dir = os.path.join(base_dir, "test_images", "cat")
    test_dog_dir = os.path.join(base_dir, "test_images", "dog")
    
    orig_cat_dir = r"C:\Users\anush\Downloads\cat-dog\PetImages\Cat"
    orig_dog_dir = r"C:\Users\anush\Downloads\cat-dog\PetImages\Dog"

    os.makedirs(test_cat_dir, exist_ok=True)
    os.makedirs(test_dog_dir, exist_ok=True)

    # 1. Count existing training images and get hashes
    print("Hashing existing training dataset...")
    existing_hashes = set()
    train_count = 0
    for d in [train_cat_dir, train_dog_dir]:
        if os.path.exists(d):
            for filename in os.listdir(d):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(d, filename)
                    existing_hashes.add(get_file_hash(path))
                    train_count += 1
    
    print(f"Total training images found: {train_count}")

    def process_class(orig_dir, target_dir, class_name, count_needed=100):
        print(f"Processing {class_name}...")
        if not os.path.exists(orig_dir):
            print(f"Original directory not found: {orig_dir}")
            return 0, 0
            
        all_orig_files = [f for f in os.listdir(orig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(all_orig_files)
        
        copied_count = 0
        duplicates_found = 0
        corrupted_found = 0
        
        for filename in all_orig_files:
            if copied_count >= count_needed:
                break
                
            orig_path = os.path.join(orig_dir, filename)
            
            # Check validity first to save hash time if corrupted
            if not is_valid_image(orig_path):
                corrupted_found += 1
                continue
                
            file_hash = get_file_hash(orig_path)
            
            if file_hash in existing_hashes:
                duplicates_found += 1
                continue
                
            # If we reach here, it's valid and not a duplicate
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(orig_path, target_path)
            existing_hashes.add(file_hash)  # Add to prevent duplicates within test set
            copied_count += 1
            
        return copied_count, duplicates_found, corrupted_found

    # Process cats
    cats_copied, cat_dups, cat_corrupt = process_class(orig_cat_dir, test_cat_dir, "cats", 100)
    
    # Process dogs
    dogs_copied, dog_dups, dog_corrupt = process_class(orig_dog_dir, test_dog_dir, "dogs", 100)
    
    total_dups = cat_dups + dog_dups
    total_corrupt = cat_corrupt + dog_corrupt
    
    print("\n--- Summary ---")
    print(f"Unseen test cats: {cats_copied}")
    print(f"Unseen test dogs: {dogs_copied}")
    print(f"Total test images: {cats_copied + dogs_copied}")
    print(f"Duplicate images found: {total_dups}")
    print(f"Corrupted images found: {total_corrupt}")

if __name__ == "__main__":
    prepare_test_dataset()
