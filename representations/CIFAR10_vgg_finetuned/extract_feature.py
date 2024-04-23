import numpy as np
import os
import torch
from pathlib import Path
import pickle
from tqdm import tqdm

from dataset1 import get_cifar10_mu_std_img, normalize

from skimage.feature import hog

from vgg_network import vgg11_bn
# from resnet_network import resnet50
from path import *

def compute_raw_pixel_features(x_train, x_test):
    raw_pixel_train_features = np.reshape(x_train, (x_train.shape[0], -1))
    raw_pixel_test_features = np.reshape(x_test, (x_test.shape[0], -1))
    print("======> Done with computation of raw pixel features")

    return raw_pixel_train_features, raw_pixel_test_features

def compute_hog_features(x_train, x_test):
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]

    # compute hog features for training images
    hog_train_features = list()
    for i in tqdm(range(num_train_samples)):
        # x_train[i]: [c, w, h] 
        fd = hog(x_train[i], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), feature_vector=True, channel_axis=0)
        
        hog_train_features.append(fd)
        
    # compute hog features for test images
    hog_test_features = list()
    for i in tqdm(range(num_test_samples)):
        # x_test[i]: [c, w, h] 
        fd = hog(x_test[i], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), feature_vector=True, channel_axis=0)
        
        hog_test_features.append(fd)
        
    hog_train_features = np.array(hog_train_features)
    hog_test_features = np.array(hog_test_features)

    print("======> Done with computation of HoG features")

    return hog_train_features, hog_test_features\


def compute_pretrained_vgg_features(x_train, x_test, mu_img, std_img, layer):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the VGG11 model with weights pre-trained on CIFAR-10
    deep_model = vgg11_bn(pretrained=True, device=device)
    deep_model.eval() 

    return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

def compute_random_vgg_features(x_train, x_test, mu_img, std_img, layer):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the VGG11 model with random weights
    deep_model = vgg11_bn(pretrained=False, device=device)
    deep_model.eval() 

    return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

# def compute_pretrained_resnet_features(x_train, x_test, mu_img, std_img, layer):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load the ResNet model with weights pre-trained on CIFAR-10
#     deep_model = resnet50(pretrained=True, device=device)
#     deep_model.eval() 

#     return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

# def compute_random_resnet_features(x_train, x_test, mu_img, std_img, layer):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Create the ResNet model with random weights
#     deep_model = resnet50(pretrained=False, device=device)
#     deep_model.eval() 

#     return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

def compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move deep_model to the GPU
    deep_model.to(device)

    # Normalize image dataset
    x_train_ = normalize(np.copy(x_train).astype(np.float32), mu_img, std_img)
    x_test_ = normalize(np.copy(x_test).astype(np.float32), mu_img, std_img)

    # Move data to the GPU
    x_train_tensor = torch.tensor(x_train_, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test_, dtype=torch.float32).to(device)

    # Compute features in batches
    batch_size = 100 
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    assert num_train_samples % batch_size == 0, "Error: The number of training samples should be divisible by the batch size"
    assert num_test_samples % batch_size == 0, "Error: The number of test samples should be divisible by the batch size"

    # Compute train features
    x_deep_features_train = []
    num_train_batches = num_train_samples // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_train_batches)):
            # Forward NN
            cur_feature_batch = deep_model.extract_features(
                x=x_train_tensor[i*batch_size : (i+1)*batch_size],
                layer=layer).cpu().detach().numpy()
            x_deep_features_train.append(cur_feature_batch)

    x_deep_features_train = np.array(x_deep_features_train).reshape(num_train_samples, -1)

    # Compute test features
    x_deep_features_test = []
    num_test_batches = num_test_samples // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_test_batches)):
            # Forward NN
            cur_feature_batch = deep_model.extract_features(
                x=x_test_tensor[i*batch_size :(i+1)*batch_size],
                layer=layer).cpu().detach().numpy()
            x_deep_features_test.append(cur_feature_batch)

    x_deep_features_test = np.array(x_deep_features_test).reshape(num_test_samples, -1)

    print("======> Done with computation of CNN features")

    return x_deep_features_train, x_deep_features_test

def load_features(sp_feature_path):
    with open(sp_feature_path, 'rb') as f:
        features = pickle.load(f)
        
    print('======> Loaded train and test features from ', sp_feature_path)

    return features["train"], features["test"]
    

def save_features(train_features, test_features, sp_feature_path):
    features = {"train": train_features, "test": test_features}
    
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    
    # save
    with open(sp_feature_path, 'wb') as f:
        pickle.dump(features, f)
    
    print('======> Saved train and test features to ', sp_feature_path)

# compute or load features
feature_types = ['raw_pixel', 'hog', 'pretrained_vgg', 'random_vgg','pretrained_resnet', 'random_resnet']
non_layer_types = ['raw_pixel', 'hog']
vgg_types = ['pretrained_vgg', 'random_vgg']
vgg_layers = ['last_conv', 'last_fc']
# resnet_types = ['pretrained_resnet', 'random_resnet']
# resnet_layers = []

def compute_or_load_features(x_train, x_test, feature_type, layer=None):
    assert feature_type in feature_types, "Error: Invalid d feature type"
    if feature_type in non_layer_types:
        assert layer is None, "Error: Layer can only be set when feature type is pretrained_cnn or random_cnn"
    elif feature_type in vgg_types:
        assert layer in vgg_layers, "Error: Invalid VGG layer type"
    # elif feature_type in resnet_types:
    #     assert layer in resnet_layers, "Error: Invalid layer type"
    else:
        assert False, "Error: Invalid layer type"
    
    if layer is None:
        sp_feature_path = os.path.join(feature_path, feature_type+".pkl")
    else:
        sp_feature_path = os.path.join(feature_path, feature_type+"_"+layer+".pkl")
    
    feature_file = Path(sp_feature_path)
    # load features from existing file
    if feature_file.is_file():
        train_features, test_features = load_features(sp_feature_path)
    # compute features
    else:
        if feature_type == "raw_pixel":
            train_features, test_features = compute_raw_pixel_features(x_train, x_test)
        elif feature_type == "hog":
            train_features, test_features = compute_hog_features(x_train, x_test)
        elif feature_type == "pretrained_vgg":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_pretrained_vgg_features(x_train, x_test, mu_img, std_img, layer)
        elif feature_type == "random_vgg":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_random_vgg_features(x_train, x_test, mu_img, std_img, layer)
        else:
            raise NotImplementedError
        
        save_features(train_features, test_features, sp_feature_path)

    print("Training feature shape: ", train_features.shape)
    print("Test feature shape: ", test_features.shape)

    return train_features, test_features

