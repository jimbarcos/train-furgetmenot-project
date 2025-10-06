"""
Fur-get Me Not: Data Exploration and Preprocessing
==================================================

This module handles data loading, exploration, preprocessing, and preparation
for the Siamese Capsule Network with MobileNetV2 for pet recognition.

Authors: Based on the thesis research
Dataset: Oxford-IIIT Pet Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as _yolo_err:
    YOLO = None

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class PetDataProcessor:
    """
    A comprehensive data processor for the Oxford-IIIT Pet Dataset
    tailored for Siamese Capsule Network training and evaluation.
    """
    
    def __init__(self, data_root='breed_organized_with_images', image_size=(224, 224), enable_detection=True):
        """
        Initialize the data processor.
        
        Args:
            data_root (str): Root directory containing the dataset
            image_size (tuple): Target image size for preprocessing
        """
        self.data_root = data_root
        self.image_size = image_size
        self.breed_to_species = {}  # Map breeds to dog/cat
        self.breed_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.preprocessed_root = f"(Preprocessed) {self.data_root}"
        self.enable_detection = enable_detection
        self.yolo_model = None
        self.yolo_target_cls = {15, 16}  # COCO: 15=cat, 16=dog
        
        # Create directories for processed data and models
        self.create_directories()
        if self.enable_detection:
            self._load_yolo_model()
        
    def create_directories(self):
        """Create necessary directories for saving processed data and models."""
        directories = [
            'processed_data',
            'models',
            'results/plots'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        # Preprocessed dataset root (created lazily per-breed/split later too)
        os.makedirs(self.preprocessed_root, exist_ok=True)

    def _load_yolo_model(self):
        """Load YOLOv8 model for cat/dog body detection."""
        if YOLO is None:
            print("Warning: ultralytics not available. Skipping detection and using simple resize.")
            self.enable_detection = False
            return
        weights_path = 'yolov8n.pt'
        if not os.path.exists(weights_path):
            print(f"Warning: '{weights_path}' not found. Skipping detection and using simple resize.")
            self.enable_detection = False
            return
        try:
            self.yolo_model = YOLO(weights_path)
            # Warm-up with a tiny blank image to initialize model
            _ = self.yolo_model.predict(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}. Detection disabled.")
            self.enable_detection = False
            
    def load_dataset_info(self):
        """Load and combine all dataset information from CSV files."""
        print("Loading dataset information...")
        
        # Load train, validation, and test info
        train_info = pd.read_csv(os.path.join(self.data_root, 'all_train_info.csv'))
        val_info = pd.read_csv(os.path.join(self.data_root, 'all_val_info.csv'))
        test_info = pd.read_csv(os.path.join(self.data_root, 'all_test_info.csv'))
        
        # Add split information
        train_info['split'] = 'train'
        val_info['split'] = 'val'
        test_info['split'] = 'test'
        
        # Combine all data
        all_data = pd.concat([train_info, val_info, test_info], ignore_index=True)
        
        # Normalize species to 0=dog, 1=cat regardless of source encoding
        all_data = self._normalize_species(all_data)
        
        # Create breed to species mapping
        for _, row in all_data.iterrows():
            self.breed_to_species[row['breed_name']] = 'cat' if row['species'] == 1 else 'dog'
        
        print(f"Total samples: {len(all_data)}")
        print(f"Breeds: {all_data['breed_name'].nunique()}")
        print(f"Species distribution:")
        species_counts = all_data['species'].value_counts()
        print(f"  Cats (1): {species_counts.get(1, 0)}")
        print(f"  Dogs (0): {species_counts.get(0, 0)}")
        
        return all_data

    def _normalize_species(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure species column is numeric with 0=dog, 1=cat.
        Supports original encodings {0,1}, {1,2}, or string labels.
        """
        if 'species' not in df.columns:
            return df
        col = df['species']
        # Handle strings
        if col.dtype == object:
            lower = col.astype(str).str.lower()
            mapping = {'dog': 0, 'dogs': 0, 'cat': 1, 'cats': 1}
            df['species'] = lower.map(mapping).fillna(lower).astype(str)
        # Try numeric conversion
        try:
            df['species'] = pd.to_numeric(df['species'])
        except Exception:
            # Fallback: leave as-is
            pass
        vals = set(pd.unique(df['species']))
        # If already binary 0/1 do nothing
        if vals.issubset({0, 1}):
            pass
        # If encoded as {1,2} assume 1=cat, 2=dog (common in Oxford-IIIT annotations)
        elif vals.issubset({1, 2}):
            df['species'] = df['species'].map({1: 1, 2: 0}).astype(int)
        else:
            # Best-effort: any value >1 becomes dog(0), 1 stays cat(1), 0 stays dog(0)
            df['species'] = df['species'].apply(lambda v: 1 if v == 1 else 0).astype(int)
        return df

    def preprocess_dataset(self, data):
        """Run detection->crop->resize for all images and save to preprocessed folder.
        Skips files that are missing or fail strict image validation.
        Returns a filtered DataFrame only for successfully processed images.
        """
        print("\nStarting preprocessing: detection -> crop -> resize -> save")
        total = len(data)
        missing = 0
        corrupted = 0
        processed = 0
        kept_indices = []
        skipped_records = []
        for idx, row in tqdm(data.iterrows(), total=total, desc="Preprocessing images", unit="img"):
            src_path = self._get_image_path(row, use_preprocessed=False)
            dst_path = self._get_image_path(row, use_preprocessed=True)

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Skip if already processed and valid
            if os.path.exists(dst_path):
                try:
                    if self._is_image_valid(dst_path):
                        processed += 1
                        kept_indices.append(idx)
                        continue
                except Exception:
                    pass  # If destination has issues, reprocess from source

            if not os.path.exists(src_path):
                missing += 1
                skipped_records.append({
                    'filename': row.get('filename', ''),
                    'breed_name': row.get('breed_name', ''),
                    'split': row.get('split', ''),
                    'reason': 'missing'
                })
                continue

            # Validate source image strictly before processing
            if not self._is_image_valid(src_path):
                corrupted += 1
                skipped_records.append({
                    'filename': row.get('filename', ''),
                    'breed_name': row.get('breed_name', ''),
                    'split': row.get('split', ''),
                    'reason': 'corrupt-invalid'
                })
                continue

            try:
                with Image.open(src_path) as pil_im:
                    pil_im.load()
                    img_rgb = np.array(pil_im.convert('RGB'))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                if img_bgr is None or img_bgr.size == 0:
                    corrupted += 1
                    skipped_records.append({
                        'filename': row.get('filename', ''),
                        'breed_name': row.get('breed_name', ''),
                        'split': row.get('split', ''),
                        'reason': 'corrupt-empty'
                    })
                    continue
                crop_bgr = self._detect_and_crop_bgr(img_bgr)
                ok = cv2.imwrite(dst_path, crop_bgr)
                if not ok:
                    corrupted += 1
                    skipped_records.append({
                        'filename': row.get('filename', ''),
                        'breed_name': row.get('breed_name', ''),
                        'split': row.get('split', ''),
                        'reason': 'write-failed'
                    })
                    continue
                processed += 1
                kept_indices.append(idx)
            except Exception:
                # Skip problematic files silently to reduce noise
                corrupted += 1
                skipped_records.append({
                    'filename': row.get('filename', ''),
                    'breed_name': row.get('breed_name', ''),
                    'split': row.get('split', ''),
                    'reason': 'exception-while-processing'
                })
                continue

        print(f"Preprocessing complete. Processed: {processed}, Missing: {missing}, Corrupted skipped: {corrupted}")
        # Save skipped records for review
        if skipped_records:
            try:
                pd.DataFrame(skipped_records).to_csv('processed_data/skipped_images.csv', index=False)
            except Exception:
                pass
        # Switch data root to preprocessed for subsequent steps
        self.data_root = self.preprocessed_root
        # Return filtered dataframe containing only the kept indices
        return data.loc[kept_indices].reset_index(drop=True)

    def _is_image_valid(self, path: str) -> bool:
        """Strictly validate an image file.
        - Must exist and be > 0 bytes
        - For JPEG, must start with SOI (FFD8) and end with EOI (FFD9)
        - PIL Image.verify and full decode must pass
        """
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            # Quick JPEG SOI/EOI check
            ext = os.path.splitext(path)[1].lower()
            if ext in {'.jpg', '.jpeg'}:
                with open(path, 'rb') as f:
                    head = f.read(2)
                    if head != b'\xff\xd8':
                        return False
                    f.seek(-2, os.SEEK_END)
                    tail = f.read(2)
                    if tail != b'\xff\xd9':
                        return False
            # PIL verify
            with Image.open(path) as im:
                im.verify()
            # Reopen and force full decode to catch truncated data
            with Image.open(path) as im2:
                im2.load()
            return True
        except Exception:
            return False
    
    def explore_dataset(self, data):
        """Perform comprehensive dataset exploration and visualization."""
        print("\n" + "="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Basic statistics
        print("\n1. Dataset Overview:")
        print(f"   Total images: {len(data)}")
        print(f"   Number of breeds: {data['breed_name'].nunique()}")
        print(f"   Number of species: {data['species'].nunique()}")
        
        # Split distribution
        print("\n2. Data Split Distribution:")
        split_counts = data['split'].value_counts()
        for split, count in split_counts.items():
            percentage = (count / len(data)) * 100
            print(f"   {split.capitalize()}: {count} images ({percentage:.1f}%)")
        
        # Species distribution
        print("\n3. Species Distribution:")
        species_map = {0: 'Dog', 1: 'Cat'}
        species_counts = data['species'].value_counts()
        for species_id, count in species_counts.items():
            species_name = species_map.get(species_id, 'Unknown')
            percentage = (count / len(data)) * 100
            print(f"   {species_name}: {count} images ({percentage:.1f}%)")
        
        # Breed distribution
        print("\n4. Top 10 Most Common Breeds:")
        breed_counts = data['breed_name'].value_counts().head(10)
        for breed, count in breed_counts.items():
            species = self.breed_to_species.get(breed, 'Unknown')
            print(f"   {breed} ({species}): {count} images")
        
        # Create visualizations
        self.create_exploration_plots(data)
        
        return data
    
    def create_exploration_plots(self, data):
        """Create comprehensive visualization plots for dataset exploration."""
        print("\nGenerating exploration plots...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Species distribution pie chart
        ax1 = plt.subplot(2, 3, 1)
        species_counts = data['species'].value_counts()
        values = [species_counts.get(0, 0), species_counts.get(1, 0)]
        species_labels = ['Dogs', 'Cats']
        colors = ['#FF9999', '#66B2FF']
        plt.pie(values, labels=species_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Species Distribution', fontsize=14, fontweight='bold')
        
        # 2. Data split distribution
        ax2 = plt.subplot(2, 3, 2)
        split_counts = data['split'].value_counts()
        bars = plt.bar(split_counts.index, split_counts.values, color=['#87CEEB', '#98FB98', '#F0E68C'])
        plt.title('Data Split Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset Split')
        plt.ylabel('Number of Images')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
        # 3. Top 15 breeds distribution
        ax3 = plt.subplot(2, 3, 3)
        top_breeds = data['breed_name'].value_counts().head(15)
        bars = plt.barh(range(len(top_breeds)), top_breeds.values)
        plt.yticks(range(len(top_breeds)), top_breeds.index)
        plt.title('Top 15 Breeds by Image Count', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Images')
        for i, v in enumerate(top_breeds.values):
            plt.text(v, i, f' {v}', va='center')
        plt.gca().invert_yaxis()
        
        # 4. Breed distribution by species
        ax4 = plt.subplot(2, 3, 4)
        breed_species_data = []
        for breed in data['breed_name'].unique():
            breed_data = data[data['breed_name'] == breed]
            species = breed_data['species'].iloc[0]
            species_name = 'Cat' if species == 1 else 'Dog'
            breed_species_data.append({'breed': breed, 'species': species_name, 'count': len(breed_data)})
        breed_species_df = pd.DataFrame(breed_species_data)
        species_breed_counts = breed_species_df.groupby('species')['breed'].count()
        bars = plt.bar(species_breed_counts.index, species_breed_counts.values, color=['#FF9999', '#66B2FF'])
        plt.title('Number of Breeds per Species', fontsize=14, fontweight='bold')
        plt.xlabel('Species')
        plt.ylabel('Number of Breeds')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
        # 5. Images per breed distribution (histogram)
        ax5 = plt.subplot(2, 3, 5)
        images_per_breed = data['breed_name'].value_counts().values
        plt.hist(images_per_breed, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Images per Breed', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Images per Breed')
        plt.ylabel('Number of Breeds')
        plt.axvline(np.mean(images_per_breed), color='red', linestyle='--', label=f'Mean: {np.mean(images_per_breed):.1f}')
        plt.legend()
        
        # 6. Breed distribution heatmap by split
        ax6 = plt.subplot(2, 3, 6)
        breed_split_crosstab = pd.crosstab(data['breed_name'], data['split'])
        top_20_breeds = data['breed_name'].value_counts().head(20).index
        breed_split_subset = breed_split_crosstab.loc[top_20_breeds]
        sns.heatmap(breed_split_subset, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Images'})
        plt.title('Images per Breed by Split (Top 20 Breeds)', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset Split')
        plt.ylabel('Breed')
        
        plt.tight_layout()
        plt.savefig('results/plots/dataset_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.create_detailed_statistics_plot(data)

    def _detect_and_crop_bgr(self, image_bgr):
        """Detect cat/dog, crop with padding, resize to target, return BGR image."""
        h, w = image_bgr.shape[:2]
        # Default: center square crop as fallback
        def center_square_crop(img):
            hh, ww = img.shape[:2]
            side = min(hh, ww)
            y0 = (hh - side) // 2
            x0 = (ww - side) // 2
            return img[y0:y0+side, x0:x0+side]

        if not self.enable_detection or self.yolo_model is None:
            crop = center_square_crop(image_bgr)
            return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)

        try:
            results = self.yolo_model.predict(image_bgr, verbose=False)
            if not results:
                crop = center_square_crop(image_bgr)
                return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)
            r0 = results[0]
            if r0.boxes is None or r0.boxes.shape[0] == 0:
                crop = center_square_crop(image_bgr)
                return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)

            # Filter for cat/dog classes
            boxes = r0.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            candidates = [(xyxy[i], conf[i]) for i in range(len(cls)) if cls[i] in self.yolo_target_cls]
            if not candidates:
                crop = center_square_crop(image_bgr)
                return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)

            # Choose highest-confidence bbox
            best_bbox, _ = max(candidates, key=lambda x: x[1])
            x1, y1, x2, y2 = best_bbox
            # Add padding
            pad_ratio = 0.15
            bw = x2 - x1
            bh = y2 - y1
            x1p = max(0, int(x1 - pad_ratio * bw))
            y1p = max(0, int(y1 - pad_ratio * bh))
            x2p = min(w, int(x2 + pad_ratio * bw))
            y2p = min(h, int(y2 + pad_ratio * bh))

            crop = image_bgr[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                crop = center_square_crop(image_bgr)
            return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)
        except Exception:
            crop = center_square_crop(image_bgr)
            return cv2.resize(crop, self.image_size, interpolation=cv2.INTER_AREA)
    
    def create_detailed_statistics_plot(self, data):
        """Create detailed statistics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot of images per breed by species
        breed_species_data = []
        for breed in data['breed_name'].unique():
            breed_data = data[data['breed_name'] == breed]
            species = breed_data['species'].iloc[0]
            species_name = 'Cat' if species == 1 else 'Dog'
            breed_species_data.append({
                'species': species_name,
                'count': len(breed_data)
            })
        
        breed_species_df = pd.DataFrame(breed_species_data)
        sns.boxplot(data=breed_species_df, x='species', y='count', ax=axes[0,0])
        axes[0,0].set_title('Distribution of Images per Breed by Species')
        axes[0,0].set_ylabel('Images per Breed')
        
        # 2. Cumulative distribution
        breed_counts = data['breed_name'].value_counts().sort_values(ascending=False)
        cumulative_counts = np.cumsum(breed_counts.values)
        cumulative_percentage = (cumulative_counts / len(data)) * 100
        
        axes[0,1].plot(range(1, len(breed_counts)+1), cumulative_percentage, marker='o')
        axes[0,1].set_title('Cumulative Distribution of Images by Breed Rank')
        axes[0,1].set_xlabel('Breed Rank')
        axes[0,1].set_ylabel('Cumulative Percentage of Images')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Split distribution by species
        split_species = pd.crosstab(data['split'], data['species'], normalize='columns') * 100
        cols = [c for c in [0, 1] if c in split_species.columns]
        split_species = split_species.reindex(columns=cols)
        split_species = split_species.rename(columns={0: 'Dogs', 1: 'Cats'})
        split_species.plot(kind='bar', ax=axes[1,0], color=['#FF9999', '#66B2FF'])
        axes[1,0].set_title('Split Distribution by Species (%)')
        axes[1,0].set_ylabel('Percentage')
        axes[1,0].set_xlabel('Dataset Split')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 4. Breed diversity by split
        diversity_by_split = data.groupby('split')['breed_name'].nunique()
        bars = axes[1,1].bar(diversity_by_split.index, diversity_by_split.values,
                           color=['#87CEEB', '#98FB98', '#F0E68C'])
        axes[1,1].set_title('Number of Breeds per Split')
        axes[1,1].set_ylabel('Number of Breeds')
        axes[1,1].set_xlabel('Dataset Split')
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/plots/detailed_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to the image file
            augment (bool): Whether to apply augmentation
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load image (BGR)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # If already preprocessed (224x224), skip detection/cropping
            if image_bgr.shape[:2] != (self.image_size[1], self.image_size[0]):
                # Detect & crop then resize (BGR)
                image_bgr = self._detect_and_crop_bgr(image_bgr)
            
            # Convert BGR to RGB for model consumption
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation if requested
            if augment:
                image = self.apply_augmentation(image)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image):
        """Apply random augmentation to an image."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast control
            beta = random.uniform(-10, 10)    # Brightness control
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image
    
    def create_siamese_pairs(self, data, num_pairs=20000):
        """
        Create positive and negative pairs for Siamese network training.
        
        Args:
            data (pd.DataFrame): Dataset information
            num_pairs (int): Number of pairs to generate (half positive, half negative)
            
        Returns:
            tuple: (pairs, labels) where pairs is array of image pairs and labels is binary array
        """
        print(f"\nCreating {num_pairs} Siamese pairs...")
        
        # Separate by split
        train_data = data[data['split'] == 'train'].reset_index(drop=True)
        val_data = data[data['split'] == 'val'].reset_index(drop=True)
        test_data = data[data['split'] == 'test'].reset_index(drop=True)
        
        train_pairs = self._generate_pairs_for_split(train_data, num_pairs)
        val_pairs = self._generate_pairs_for_split(val_data, num_pairs//2)
        test_pairs = self._generate_pairs_for_split(test_data, num_pairs//2)
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def _generate_pairs_for_split(self, data, num_pairs):
        """Generate pairs for a specific data split."""
        pairs_data = []
        labels = []
        
        # Group by breed for positive pairs
        breed_groups = data.groupby('breed_name')
        breed_names = list(breed_groups.groups.keys())
        
        # Generate positive pairs (same breed)
        positive_pairs = 0
        max_positive = num_pairs // 2
        
        while positive_pairs < max_positive and len(breed_names) > 0:
            breed = random.choice(breed_names)
            breed_images = breed_groups.get_group(breed)
            
            if len(breed_images) >= 2:
                # Select two different images from the same breed
                idx1, idx2 = random.sample(range(len(breed_images)), 2)
                img1_info = breed_images.iloc[idx1]
                img2_info = breed_images.iloc[idx2]
                
                pairs_data.append({
                    'img1_path': self._get_image_path(img1_info),
                    'img2_path': self._get_image_path(img2_info),
                    'img1_breed': img1_info['breed_name'],
                    'img2_breed': img2_info['breed_name'],
                    'same_breed': True
                })
                labels.append(1)  # Positive pair
                positive_pairs += 1
        
        # Generate negative pairs (different breeds)
        negative_pairs = 0
        max_negative = num_pairs - positive_pairs
        
        while negative_pairs < max_negative:
            # Select two different breeds
            if len(breed_names) >= 2:
                breed1, breed2 = random.sample(breed_names, 2)
                
                breed1_images = breed_groups.get_group(breed1)
                breed2_images = breed_groups.get_group(breed2)
                
                # Select random image from each breed
                img1_info = breed1_images.iloc[random.randint(0, len(breed1_images)-1)]
                img2_info = breed2_images.iloc[random.randint(0, len(breed2_images)-1)]
                
                pairs_data.append({
                    'img1_path': self._get_image_path(img1_info),
                    'img2_path': self._get_image_path(img2_info),
                    'img1_breed': img1_info['breed_name'],
                    'img2_breed': img2_info['breed_name'],
                    'same_breed': False
                })
                labels.append(0)  # Negative pair
                negative_pairs += 1
        
        print(f"Generated {positive_pairs} positive pairs and {negative_pairs} negative pairs")
        
        return {
            'pairs_info': pairs_data,
            'labels': np.array(labels),
            'num_positive': positive_pairs,
            'num_negative': negative_pairs
        }
    
    def _get_image_path(self, img_info, use_preprocessed=False):
        """Get full path to an image file. If use_preprocessed=True, point to preprocessed root."""
        split_dir = img_info['split']
        breed_name = img_info['breed_name']
        filename = img_info['filename']
        # The filename might not have extension, add .jpg if needed
        if not str(filename).lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = f"{filename}.jpg"
        root = self.preprocessed_root if use_preprocessed else self.data_root
        return os.path.join(root, breed_name, split_dir, filename)
    
    def create_data_generators(self, pairs_data):
        """
        Create data generators for Siamese network training.
        
        Args:
            pairs_data (dict): Dictionary containing pairs for train/val/test splits
            
        Returns:
            dict: Dictionary containing data generators
        """
        print("\nCreating data generators...")
        
        generators = {}
        
        for split, pairs in pairs_data.items():
            print(f"Creating {split} generator with {len(pairs['labels'])} pairs...")
            
            generator = SiameseDataGenerator(
                pairs_info=pairs['pairs_info'],
                labels=pairs['labels'],
                batch_size=32 if split == 'train' else 16,
                image_size=self.image_size,
                augment=(split == 'train'),
                shuffle=(split == 'train'),
                processor=self
            )
            
            generators[split] = generator
        
        return generators
    
    def save_processed_data(self, data, pairs_data):
        """Save processed data for future use."""
        print("\nSaving processed data...")
        
        # Save dataset information
        data.to_csv('processed_data/dataset_info.csv', index=False)
        
        # Save pairs information
        for split, pairs in pairs_data.items():
            pairs_df = pd.DataFrame(pairs['pairs_info'])
            pairs_df['label'] = pairs['labels']
            pairs_df.to_csv(f'processed_data/{split}_pairs.csv', index=False)
        
        # Save breed mappings
        breed_mapping = pd.DataFrame([
            {'breed': breed, 'species': species}
            for breed, species in self.breed_to_species.items()
        ])
        breed_mapping.to_csv('processed_data/breed_species_mapping.csv', index=False)
        
        print("Processed data saved successfully!")
    
    def get_sample_images(self, data, num_samples=16):
        """Get sample images for visualization."""
        print(f"\nLoading {num_samples} sample images for visualization...")
        
        # Select random samples from different breeds
        sample_data = data.groupby('breed_name').apply(
            lambda x: x.sample(min(1, len(x)))
        ).reset_index(drop=True)
        
        sample_data = sample_data.sample(min(num_samples, len(sample_data)))
        
        sample_images = []
        sample_info = []
        
        for _, row in sample_data.iterrows():
            img_path = self._get_image_path(row)
            image = self.load_and_preprocess_image(img_path, augment=False)
            
            if image is not None:
                sample_images.append(image)
                sample_info.append({
                    'breed': row['breed_name'],
                    'species': 'Cat' if row['species'] == 1 else 'Dog',
                    'split': row['split']
                })
        
        # Create visualization
        self.visualize_sample_images(sample_images, sample_info)
        
        return sample_images, sample_info
    
    def visualize_sample_images(self, images, info):
        """Visualize sample images from the dataset."""
        num_images = len(images)
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].set_title(f"{info[i]['breed']}\n({info[i]['species']}, {info[i]['split']})")
            axes[i].axis('off')
        
        # Hide remaining subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/plots/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()


class SiameseDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for Siamese network training.
    Loads image pairs on-the-fly to manage memory efficiently.
    """
    
    def __init__(self, pairs_info, labels, batch_size, image_size, augment=False, shuffle=True, processor=None):
        self.pairs_info = pairs_info
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.processor = processor
        self.indices = np.arange(len(pairs_info))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.pairs_info) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_img1 = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)
        batch_img2 = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)
        batch_labels = np.zeros(self.batch_size, dtype=np.float32)
        
        # Load and process images
        for i, idx in enumerate(batch_indices):
            pair_info = self.pairs_info[idx]
            
            # Load images
            img1 = self.processor.load_and_preprocess_image(
                pair_info['img1_path'], augment=self.augment
            )
            img2 = self.processor.load_and_preprocess_image(
                pair_info['img2_path'], augment=self.augment
            )
            
            # Handle failed image loading
            if img1 is None or img2 is None:
                # Use random images as fallback
                img1 = np.random.random((*self.image_size, 3)).astype(np.float32)
                img2 = np.random.random((*self.image_size, 3)).astype(np.float32)
            
            batch_img1[i] = img1
            batch_img2[i] = img2
            batch_labels[i] = self.labels[idx]
        
        return [batch_img1, batch_img2], batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    """Main function to run the data exploration and preprocessing pipeline."""
    print("Fur-get Me Not: Data Exploration and Preprocessing")
    print("=" * 60)
    
    # Initialize processor
    processor = PetDataProcessor()
    
    # Load dataset information
    data = processor.load_dataset_info()
    
    # Preprocess all images: detection -> crop -> resize -> save to preprocessed folder
    data = processor.preprocess_dataset(data)
    
    # Explore dataset
    data = processor.explore_dataset(data)
    
    # Get sample images for visualization
    sample_images, sample_info = processor.get_sample_images(data, num_samples=16)
    
    # Create Siamese pairs
    pairs_data = processor.create_siamese_pairs(data, num_pairs=20000)
    
    # Create data generators
    generators = processor.create_data_generators(pairs_data)
    
    # Save processed data
    processor.save_processed_data(data, pairs_data)
    
    print(f"\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Dataset summary:")
    print(f"  Total images: {len(data)}")
    print(f"  Breeds: {data['breed_name'].nunique()}")
    print(f"  Training pairs: {len(pairs_data['train']['labels'])}")
    print(f"  Validation pairs: {len(pairs_data['val']['labels'])}")
    print(f"  Test pairs: {len(pairs_data['test']['labels'])}")
    print(f"\nProcessed data saved in 'processed_data/' directory")
    print(f"Visualizations saved in 'results/plots/' directory")
    print("="*60)
    
    return processor, data, pairs_data, generators


if __name__ == "__main__":
    processor, data, pairs_data, generators = main()
