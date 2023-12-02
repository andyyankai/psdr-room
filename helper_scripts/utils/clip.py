import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from PIL import Image
import clip




class ClipDescriptor(torch.nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        model, preprocess = clip.load("ViT-L/14",device=device,jit=False)#"ViT-B/32"
        if device == "cpu":
            model.float()
        else :
            clip.model.convert_weights(model)
        self.device = device
        self.preprocess = preprocess
        self.model = model

    def forward(self, image, layer_inds=None, cat_layers=None, mask=None):                
        image = torch.unsqueeze(self.preprocess(image), 0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features

def build_knn(data_features):
    """Builds a nearest neighbor object to be queried"""
    if not data_features is None:
        NN = NearestNeighbors(n_neighbors=40, metric="cosine")
        data_features = np.squeeze(data_features)
        NN.fit(data_features)
        return NN

def get_image_features(im, NN, descriptor):
    height, width = im.size
    heightCenter = height // 2
    widthCenter = width // 2

    #crop the image
    finalSizeOver2 = np.minimum(height, width) //2
    im = im.crop((widthCenter - finalSizeOver2, heightCenter - finalSizeOver2,widthCenter + finalSizeOver2,heightCenter + finalSizeOver2))

    #extract its features
    features = descriptor(im, layer_inds = [0,1,2,3])
    features = features.cpu().detach().numpy()
    features = np.squeeze(features)
    features = features
    return features

def get_neighbor_names(allNeighbors_ids, names):
    # print(allNeighbors_ids)
    return [names[neighbor] for neighbor in allNeighbors_ids]
    

def runImage(im, KNN, descriptor, names, nb_neighbors = 5):
    features = get_image_features(im, KNN, descriptor)
    distances, allNeighbors_ids = KNN.kneighbors([features], nb_neighbors, return_distance=True)
    return get_neighbor_names(allNeighbors_ids, names)


def resize_image(img, size=(128,128)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return img

    dif = h if h > w else w

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype) + 254
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype) + 254
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return mask

def search_item(image, cat, nb_neighbors=1):
    dataset = np.load(f"./model_data/FUTURE_{cat}_features.npy")#[0:4000]
    names = np.load(f"./model_data/FUTURE_{cat}_name.npy")#[0:4000]

    # print(dataset.shape)
    # exit()

    NN = build_knn(dataset)
    descriptor = ClipDescriptor(device ="cuda")

    image_to_search = Image.fromarray(image)
    if cat == "gray":
        image_to_search = image_to_search.convert("L").convert("RGB")
    rs = runImage(image_to_search, NN, descriptor, names, nb_neighbors=nb_neighbors)

    return rs

def search_text(image, nb_neighbors=2):
    dataset = np.load(f"./model_data/text_features.npy")
    names = np.load(f"./model_data/text_name.npy")

    NN = build_knn(dataset)
    descriptor = ClipDescriptor(device ="cuda")

    image_to_search = Image.fromarray(image)
    rs = runImage(image_to_search, NN, descriptor, names, nb_neighbors=nb_neighbors)
    return rs
    # print(rs)
