from typing import Any
import cog
import torch
import open_clip
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import zipfile
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
from cog import BasePredictor, Input, Path
class Predictor(BasePredictor):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model.eval()
    def predict(self, zip_file: Path = Input(description="Image to classify")) -> Any: 
        # Extract ZIP file
        extract_path = "/tmp/dataset"
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Load and process images
        images, labels, image_files = self.load_and_process_images(extract_path)

        # Encode images
        embeddings = []
        for img in tqdm(images):
            embedding = self.encode_image(img)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # Find mislabeled images
        mislabeled_images = self.find_mislabeled_images(embeddings, labels, image_files)

        # Save results to file
        output_file = "/tmp/mislabeled_images.txt"
        with open(output_file, 'w') as f:
            for image, probability in mislabeled_images:
                f.write(f"{image}: {probability:.4f}\n")

        return output_file

    def load_and_process_images(self, dataset_path: str) -> Tuple[List[Image.Image], List[int], List[str]]:
        images = []
        image_files = []
        labels = []

        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')

        for img_file in os.listdir(images_path):
            img_path = os.path.join(images_path, img_file)
            label_path = os.path.join(labels_path, img_file.replace('.jpg', '.txt'))

            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size

                with open(label_path, 'r') as f:
                    label_data = f.read().strip().split('\n')

                for obj in label_data:
                    obj_data = obj.split()
                    if len(obj_data) < 5:
                        continue
                    class_id = int(obj_data[0])
                    bbox = [float(x) for x in obj_data[1:5]]

                    x, y, width, height = bbox
                    left = max(0, int((x - width/2) * w))
                    top = max(0, int((y - height/2) * h))
                    right = min(w, int((x + width/2) * w))
                    bottom = min(h, int((y + height/2) * h))

                    if right > left and bottom > top:
                        img_cropped = img.crop((left, top, right, bottom))
                        images.append(img_cropped)
                        image_files.append(img_file)
                        labels.append(class_id)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        return images, labels, image_files

    def encode_image(self, img: Image.Image) -> np.ndarray:
        img_tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
        return embedding.numpy().flatten()

    def find_mislabeled_images(self, embeddings: np.ndarray, labels: List[int], image_files: List[str], k: 20) -> List[Tuple[str, float]]:
        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        knn.fit(embeddings)

        distances, indices = knn.kneighbors(embeddings)

        mislabeled_probabilities = []
        for i, (distances_i, indices_i) in enumerate(zip(distances, indices)):
            neighbor_labels = [labels[j] for j in indices_i[1:]]
            mislabeled_count = sum(1 for label in neighbor_labels if label != labels[i])
            mislabel_probability = mislabeled_count / k

            if mislabel_probability > 0.5:
                mislabeled_probabilities.append((image_files[i], mislabel_probability))

        mislabeled_probabilities.sort(key=lambda x: x[1], reverse=True)

        return mislabeled_probabilities
