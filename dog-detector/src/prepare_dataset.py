import os
import shutil
from pathlib import Path
from PIL import Image

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def copy_valid_images(src_dir, dest_dir, limit=1000):
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return count
        
    print(f"Reading from {src_dir} ...")
    for file in os.listdir(src_dir):
        if count >= limit:
            break
            
        file_path = src_dir / file
        if file_path.is_file():
            # Check if it is a valid image to avoid copying corrupted ones
            if is_valid_image(file_path):
                dest_path = dest_dir / file
                # Only copy if it doesn't already exist to avoid duplicates if run multiple times
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                count += 1
            else:
                print(f"Skipping corrupt or invalid file: {file_path}")
                
    return count

if __name__ == "__main__":
    # Source dataset found in the user's Downloads folder
    src_base = Path(r"C:\Users\anush\Downloads\cat-dog\PetImages")
    dest_base = Path("dataset")
    
    print("Preparing Cat dataset...")
    cat_count = copy_valid_images(src_base / "Cat", dest_base / "cat", limit=1000)
    print(f"Copied {cat_count} valid cat images.")
    
    print("\nPreparing Dog dataset...")
    dog_count = copy_valid_images(src_base / "Dog", dest_base / "dog", limit=1000)
    print(f"Copied {dog_count} valid dog images.")
    
    print("\nDataset preparation complete!")
