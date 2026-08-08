import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def validate_dataset(dataset_dir="dataset"):
    cat_dir = Path(dataset_dir) / "cat"
    dog_dir = Path(dataset_dir) / "dog"

    print(f"Checking dataset structure in '{dataset_dir}'...")

    # 1. & 2. Check directories
    if not cat_dir.exists():
        print(f"Error: '{cat_dir}' does not exist.")
        return
    else:
        print(f"Found: '{cat_dir}'")
        
    if not dog_dir.exists():
        print(f"Error: '{dog_dir}' does not exist.")
        return
    else:
        print(f"Found: '{dog_dir}'")

    # 4. Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

    def process_directory(directory):
        valid_images = 0
        corrupt_images = 0
        unsupported = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                
                # Check extension
                if file_path.suffix.lower() not in supported_formats:
                    unsupported += 1
                    continue
                    
                # 5. Detect unreadable/corrupted images
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    valid_images += 1
                except (IOError, SyntaxError, UnidentifiedImageError):
                    print(f"Warning: Corrupt or unreadable image -> {file_path}")
                    corrupt_images += 1
                    
        return valid_images, corrupt_images, unsupported

    print("\nValidating images...")
    
    # 3. Count images in each class
    cat_count, cat_corrupt, cat_unsupp = process_directory(cat_dir)
    dog_count, dog_corrupt, dog_unsupp = process_directory(dog_dir)
    
    # 6. Total number of images
    total_count = cat_count + dog_count

    # 7. Check whether classes are reasonably balanced
    balance_status = "Unknown"
    if total_count > 0:
        cat_pct = cat_count / total_count
        dog_pct = dog_count / total_count
        if abs(cat_pct - dog_pct) <= 0.20:
            balance_status = f"Reasonably balanced (Cats: {cat_pct:.1%}, Dogs: {dog_pct:.1%})"
        else:
            balance_status = f"Imbalanced (Cats: {cat_pct:.1%}, Dogs: {dog_pct:.1%})"
    elif total_count == 0:
        balance_status = "No valid images found. Cannot determine balance."

    print(f"\nBalance Check: {balance_status}")
    if cat_corrupt > 0 or dog_corrupt > 0:
        print(f"Corrupted Images found - Cats: {cat_corrupt}, Dogs: {dog_corrupt}")
    if cat_unsupp > 0 or dog_unsupp > 0:
        print(f"Unsupported Files found - Cats: {cat_unsupp}, Dogs: {dog_unsupp}")

    print("\n--- Final Count ---")
    print(f"Cats: {cat_count}")
    print(f"Dogs: {dog_count}")
    print(f"Total: {total_count}")

if __name__ == "__main__":
    validate_dataset()
