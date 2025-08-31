#!/usr/bin/env python3
"""
Dataset splitting script for airplane SNN tracker dataset.
Splits the dataset into train/val/test with 80/10/10 ratio for each plane type.
"""

import os
import shutil
import glob
from collections import defaultdict
import random

def analyze_dataset(dataset_dir):
    """Analyze the dataset to understand the plane types and file distribution"""
    
    # Get all .mat files
    mat_files = glob.glob(os.path.join(dataset_dir, "*.mat"))
    
    # Group files by plane type
    plane_types = defaultdict(list)
    
    for mat_file in mat_files:
        filename = os.path.basename(mat_file)
        
        # Extract plane type from filename
        if filename.startswith("F117_"):
            plane_types["F117"].append(mat_file)
        elif filename.startswith("Mig"):
            plane_types["Mig31"].append(mat_file)
        elif filename.startswith("Su"):
            plane_types["Su35"].append(mat_file)
        else:
            print(f"Warning: Unknown plane type for file {filename}")
    
    # Print analysis
    print("Dataset Analysis:")
    print(f"Total files: {len(mat_files)}")
    for plane_type, files in plane_types.items():
        print(f"  {plane_type}: {len(files)} files")
    
    return plane_types


def split_files(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split a list of files into train/val/test sets"""
    
    # Shuffle files for random split
    files_copy = files.copy()
    random.shuffle(files_copy)
    
    total_files = len(files_copy)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # Remaining files go to test
    
    train_files = files_copy[:train_count]
    val_files = files_copy[train_count:train_count + val_count]
    test_files = files_copy[train_count + val_count:]
    
    return train_files, val_files, test_files


def create_split_directories(output_dir):
    """Create the directory structure for train/val/test splits"""
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created directory: {split_dir}")


def copy_files_to_split(files, destination_dir, split_name):
    """Copy files to the appropriate split directory"""
    
    dest_path = os.path.join(destination_dir, split_name)
    
    print(f"Copying {len(files)} files to {split_name}...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        dest_file = os.path.join(dest_path, filename)
        shutil.copy2(file_path, dest_file)
    
    print(f"  Copied {len(files)} files to {dest_path}")


def main():
    """Main function to split the dataset"""
    
    # Configuration
    dataset_dir = "/Users/banika/Desktop/airplane_SNN_tracker/ATIS_PlaneDroppingDataSet"
    output_dir = "/Users/banika/Desktop/airplane_SNN_tracker/split_dataset"
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    print("=" * 60)
    print("AIRPLANE SNN TRACKER DATASET SPLITTER")
    print("=" * 60)
    
    # Check if source directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Source directory {dataset_dir} not found!")
        return
    
    # Analyze the dataset
    plane_types = analyze_dataset(dataset_dir)
    
    if not plane_types:
        print("No .mat files found in the dataset!")
        return
    
    # Create output directory structure
    print(f"\nCreating split directories in: {output_dir}")
    create_split_directories(output_dir)
    
    # Combine all files from all plane types
    all_files = []
    for plane_type, files in plane_types.items():
        all_files.extend(files)
    
    print(f"\nTotal files to split: {len(all_files)}")
    
    # Split all files together (mixed plane types)
    print(f"Splitting dataset with 80/10/10 ratio...")
    train_files, val_files, test_files = split_files(all_files)
    
    # Copy files to appropriate directories
    copy_files_to_split(train_files, output_dir, 'train')
    copy_files_to_split(val_files, output_dir, 'val')
    copy_files_to_split(test_files, output_dir, 'test')
    
    # Count plane types in each split for statistics
    def count_plane_types(files):
        counts = {'F117': 0, 'Mig31': 0, 'Su35': 0}
        for file_path in files:
            filename = os.path.basename(file_path)
            if filename.startswith("F117_"):
                counts['F117'] += 1
            elif filename.startswith("Mig"):
                counts['Mig31'] += 1
            elif filename.startswith("Su"):
                counts['Su35'] += 1
        return counts
    
    train_counts = count_plane_types(train_files)
    val_counts = count_plane_types(val_files)
    test_counts = count_plane_types(test_files)
    
    # Print overall statistics
    total_files = len(all_files)
    print(f"\n" + "=" * 60)
    print("OVERALL SPLIT STATISTICS")
    print("=" * 60)
    print(f"Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)")
    print(f"Val:   {len(val_files)} files ({len(val_files)/total_files*100:.1f}%)")
    print(f"Test:  {len(test_files)} files ({len(test_files)/total_files*100:.1f}%)")
    print(f"Total: {total_files} files")
    
    # Print plane type distribution in each split
    print(f"\nPlane type distribution:")
    print(f"Train set: F117={train_counts['F117']}, Mig31={train_counts['Mig31']}, Su35={train_counts['Su35']}")
    print(f"Val set:   F117={val_counts['F117']}, Mig31={val_counts['Mig31']}, Su35={val_counts['Su35']}")
    print(f"Test set:  F117={test_counts['F117']}, Mig31={test_counts['Mig31']}, Su35={test_counts['Su35']}")
    
    # Print directory structure
    print(f"\nDataset split completed! Files organized in:")
    print(f"{output_dir}/")
    print("├── train/")
    print("├── val/")
    print("└── test/")

if __name__ == "__main__":
    main()
