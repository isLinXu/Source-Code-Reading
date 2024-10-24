import torch
from models.yolo import Model

import cv2

def load_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img
# def extract_features(model, img):
#     # Remove the last layer (Detect) from the model
#     feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#     features = feature_extractor(img)
#     return features
def extract_features(model, img):
    # Remove the last layer (Detect) from the model
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    features = feature_extractor(img)
    return features.view(features.size(0), -1)  # Flatten the features

import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose

# Load the YoloV5 model
model = Model("/Users/gatilin/PycharmProjects/yolo-lab/yolov5/models/yolov5s.yaml", ch=3, nc=80)
model.eval()

# Prepare the data
image_dir = "/Users/gatilin/PycharmProjects/yolo-lab/yolov5/data/images"
image_filenames = sorted(os.listdir(image_dir))
num_images = len(image_filenames)

# Transform the images
transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Extract features
features = []
for filename in image_filenames:
    img_path = os.path.join(image_dir, filename)
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = load_image(img_path)
    img = transform(img).unsqueeze(0)
    feature = extract_features(model, img)
    features.append(feature.detach().numpy())

# Stack the features into a matrix
feature_matrix = np.vstack(features)

import faiss

# Normalize the features
faiss.normalize_L2(feature_matrix)

# Build the index
index = faiss.IndexFlatL2(feature_matrix.shape[1])
# index.add(feature_matrix)

def search_similar_images(query_img, index, k=5):
    query_img = cv2.resize(query_img, (224, 224))  # Resize the query image
    query_img = transform(query_img).unsqueeze(0)
    query_feature = extract_features(model, query_img).detach().numpy()
    faiss.normalize_L2(query_feature)
    _, indices = index.search(query_feature, k)
    return indices
# def search_similar_images(query_img, index, k=5):
#     query_img = transform(query_img).unsqueeze(0)
#     query_feature = extract_features(model, query_img).detach().numpy()
#     faiss.normalize_L2(query_feature)
#     _, indices = index.search(query_feature, k)
#     return indices

# Load a query image
query_img_path = "/Users/gatilin/PycharmProjects/yolo-lab/yolov5/data/images/bus.jpg"
query_img = cv2.imread(query_img_path)
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

# Search similar images
similar_image_indices = search_similar_images(query_img, index)
print("Similar image indices:", similar_image_indices)