import tifffile
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import random
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
from pytorch_grad_cam import GradCAM
from glob import glob
import os 

def min_max_norm(values):
    minimum = values.min()
    maximum = values.max()
    return (values-minimum)/(maximum-minimum)

# stitch together patches to reconstruct the original image from patches.
def stitch_patches(patches, num_patches_x, num_patches_y, patch_size=224):
    # Get the full image size based on patch count
    full_width = num_patches_x * patch_size
    full_height = num_patches_y * patch_size
    
    # Create a blank canvas to stitch the patches back
    stitched_image = np.zeros((full_height, full_width))

    patch_idx = 0
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            x1 = j*patch_size
            x2 = x1 + patch_size
            y1 = i*patch_size
            y2 = y1 + patch_size            
            stitched_image[y1:y2, x1:x2] = patches[patch_idx]
            patch_idx += 1

    return stitched_image
    
def _open_tif_with_properties(path):
        with tifffile.TiffFile(path) as tif:
            properties = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                properties[name] = value
            image = tif.pages[0].asarray()
        try:
            magnification = properties["OlympusSIS"]["magnification"]
            pixelsize = properties["OlympusSIS"]["pixelsizex"]
            properties = {
                "magnification": magnification,
                "pixelsize": pixelsize,
                "path": path,
            }
        except:
            print("ERROR:: properties of file: " + str(path))
            print(properties)
        return image, properties
    
def _get_class_names(path):
        root = ET.parse(path).getroot()
        label_names = [label.find('name').text.lower() for label in root.findall('.//task/labels/label')]
        return label_names
    
def get_gradcam(model, img):
    target_layers = [model.layer4[-1]] 
    with torch.enable_grad():
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=img[None,...], targets=None)[0]
    return grayscale_cam

# Randomly resize the image to mimic different levels of magnification
class RandomResizeWithLocations:
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img, locations):
        # Get the image dimensions
        w, h = F.get_image_size(img)
        
        # Apply random scaling factor
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        
        # Resize the image
        resized_img = F.resize(img, [int(h * scale_factor), int(w * scale_factor)])
        
        # Scale the locations by the same factor
        resized_locations = locations * scale_factor
               
        return resized_img, resized_locations
    
# Crop random part of full sized image to fit to models input size
import random
import torch
import torchvision.transforms.functional as F
from collections import defaultdict

class BalancedRandomCropWithLocations:
    def __init__(self, size, empty_crop_prob=0.5):
        """
        Args:
            size (tuple): Target crop size (height, width).
            empty_crop_prob (float): Probability of selecting a fully random crop.
        """
        self.size = size
        self.empty_crop_prob = empty_crop_prob
        self.class_counts_instance = defaultdict(int)  # Tracks occurrences of each class
        self.class_counts_patch = defaultdict(int)  # Tracks occurrences of each class
        

    @staticmethod
    def get_params(img, locations, output_size, force_empty=False, selected_idx=None):
        """Select a random crop position, optionally centered on a selected object."""
        w, h = F.get_image_size(img)
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w  # No cropping needed

        if force_empty or locations.shape[0] == 0 or selected_idx is None:
            # Completely random crop
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            # Pick a crop around the selected object with a random offset
            x, y = locations[selected_idx]

            # Convert x and y to Python integers
            x = int(x.item())  # .item() gets the value as a Python scalar
            y = int(y.item())

            # Compute min and max crop positions that keep (x, y) inside
            min_i = max(0, y - th + 1)  # Ensures y is within [i, i+th]
            max_i = min(h - th, y)      # Ensures crop stays in bounds

            min_j = max(0, x - tw + 1)  # Ensures x is within [j, j+tw]
            max_j = min(w - tw, x)      # Ensures crop stays in bounds

            # Pick a random offset within valid bounds
            i = random.randint(min_i, max_i)
            j = random.randint(min_j, max_j)


        return i, j, th, tw

    def select_underrepresented_class(self, lbls):
        """Find the most underrepresented class in the image."""
        class_frequencies = {cls: self.class_counts_instance[cls] for cls in set(lbls)}
        return min(class_frequencies, key=class_frequencies.get, default=None)

    def print_class_distribution(self):
        """Prints the tracked class distribution."""
        print("Current class distribution per instance:")
        for cls, count in sorted(self.class_counts_instance.items(), key=lambda x: x[1], reverse=True):
            print(f"Class: {cls}, Count: {count}")
            
        print("\nCurrent class distribution per patch:")
        for cls, count in sorted(self.class_counts_patch.items(), key=lambda x: x[1], reverse=True):
            print(f"Class: {cls}, Count: {count}")

    def __call__(self, img, locations, lbls):
        """
        Args:
            img (PIL Image or Tensor): Input image.
            locations (Tensor): Object locations (Nx2) with (x, y) coordinates.
            lbls (list): Corresponding class labels for each object.

        Returns:
            img (Tensor): Cropped image.
            cropped_locations (Tensor): Updated object locations.
            cropped_lbls (list): Updated labels.
        """
        force_empty = random.random() < self.empty_crop_prob  # 50% chance for random crop

        selected_idx = None  # Default to None (random crop)

        if not force_empty and len(lbls) > 0:
            # Find an underrepresented class
            target_class = self.select_underrepresented_class(lbls)

            if target_class is not None:
                # Select an index where this class is present
                possible_indices = [i for i, lbl in enumerate(lbls) if lbl == target_class]
                if possible_indices:
                    selected_idx = random.choice(possible_indices)

        # Get crop parameters
        i, j, h, w = self.get_params(img, locations, self.size, force_empty=force_empty, selected_idx=selected_idx)
        img_cropped = F.crop(img, int(i), int(j), int(h), int(w))


        # Adjust locations based on crop offset
        if locations.shape[0] > 0:
            cropped_locations = locations - torch.tensor([j, i])

            # Filter locations inside the cropped area
            valid_mask = (
                (cropped_locations[:, 0] >= 0) & (cropped_locations[:, 0] < w) &
                (cropped_locations[:, 1] >= 0) & (cropped_locations[:, 1] < h)
            )

            cropped_locations = cropped_locations[valid_mask]
            cropped_lbls = [lbls[idx] for idx in valid_mask.nonzero().flatten().tolist()]
        else:
            cropped_locations = locations
            cropped_lbls = []
            self.class_counts_patch["empty"] += 1

        # Update class occurrence tracking
        for lbl in cropped_lbls:
            self.class_counts_instance[lbl] += 1
            
        for lbl in np.unique(cropped_lbls):
            self.class_counts_patch[lbl] += 1

        return img_cropped, cropped_locations, cropped_lbls

    
# Resize the image to a multiple of the base_size, such that the image can be divided into a set of patches of sixe base_size x base_size
class ResizeToMultipleWithLocations:
    def __init__(self, base_size=224):
        self.base_size = base_size

    @staticmethod
    def round_to_multiple(value, base):
        # Round a value to the nearest multiple of the base.
        multiple = int(base * round(float(value) / base))
        if multiple == 0: 
            return base
        return multiple

    def __call__(self, img, locations = None):
        # Get the original image dimensions
        w, h = F.get_image_size(img)
        
        # Round the width and height to the nearest multiple of base_size
        new_w = self.round_to_multiple(w, self.base_size)
        new_h = self.round_to_multiple(h, self.base_size)
        
        # Resize the image to the new dimensions
        resized_img = F.resize(img, [new_h, new_w])
        
        if(locations is None): 
            return resized_img, None
        
        # Adjust the locations based on the scaling factor (new dimensions / original dimensions)
        if(locations.shape[0] > 0):
            resized_locations = locations * torch.tensor([new_w / w, new_h / h])
        else: resized_locations = locations
        
        return resized_img, resized_locations

# split the image into non-overlapping 224x224 patches and map locations to the correct patch.
def patchify_image(image, locations = None, lbls = None, patch_size=224):
    w, h = F.get_image_size(image)
    
    assert w % patch_size == 0 and h % patch_size == 0, "Image dimensions must be a multiple of patch_size."
    
    patches = []
    patch_locations = {}
    patch_lbls = []

    
    # Create patches and map locations to patches
    num_patches_x = w // patch_size
    num_patches_y = h // patch_size

    patch_idx = 0
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Extract the patch
            top = i * patch_size
            left = j * patch_size
            patch = F.crop(image, top, left, patch_size, patch_size)
            patches.append(patch)

            if(not (locations is None)):
                # Adjust the locations for the current patch
                if(locations.shape[0] > 0):
                    in_patch_locations = locations - torch.tensor([left, top])
                    # Filter locations that are within this patch
                    valid_mask = (
                        (in_patch_locations[:, 0] >= 0) & (in_patch_locations[:, 0] < patch_size) &
                        (in_patch_locations[:, 1] >= 0) & (in_patch_locations[:, 1] < patch_size)
                    )

                    in_patch_locations = in_patch_locations[valid_mask]
                    in_patch_lbls = lbls[valid_mask.numpy()]     
                else: 
                    in_patch_locations = locations
                    in_patch_lbls = lbls
                
                patch_locations[patch_idx] = in_patch_locations
                patch_lbls.append(in_patch_lbls)
            
            patch_idx += 1
    

    return patches, patch_locations, patch_lbls, num_patches_x, num_patches_y





class VirusDataset(Dataset):
    def __init__(self, annotation_file, img_dir, split, transforms = None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.split = split              

        if(split == "all"):
            if os.path.isdir(img_dir):
                img_paths = glob(self.img_dir+"/*.tif")
                self.imgs = [_open_tif_with_properties(img_path)[0] for img_path in img_paths]
            elif os.path.isfile(img_dir):
                self.imgs = [_open_tif_with_properties(self.img_dir)[0]]
            else:
                print(f"{img_dir} does not exist or is neither a file nor a directory.")
            self.annotations = None
            
        else:
            self.annotations = self._parse_annotation_file()
            self.class_names = _get_class_names(annotation_file)
            
            # split data into train/val/test set
            train, rest = train_test_split(self.annotations, test_size=0.4, random_state=42)
            val, test = train_test_split(rest, test_size=0.5, random_state=42)

            # load all images and annotations into RAM
            if split == "train":
                self.annotations = train
                self.imgs = [_open_tif_with_properties(Path(img_dir) / Path(annotation["filename"]))[0] for annotation in train]
            elif split == "val":
                self.annotations = val
                self.imgs = [_open_tif_with_properties(Path(img_dir) / Path(annotation["filename"]))[0] for annotation in val]
            elif split == "test":
                self.annotations = test
                self.imgs = [_open_tif_with_properties(Path(img_dir) / Path(annotation["filename"]))[0] for annotation in test]
            else:
                print(f"{split} not implemented. Please try 'train', 'val' or 'test'.")
                
            print(f"Setup Dataset with {len(self.class_names)} classes: {self.class_names} and {len(self.imgs)} micrographs.")


        self.num_micrographs = len(self.imgs)

        return 
    
    def _parse_annotation_file(self):
        root = ET.parse(self.annotation_file).getroot()
        annotations = []
        # find all images
        for img in root.findall('image'):
            filename = img.get("name")
            lbls = []
            coords = []
            for virus in img.findall("points"):
                lbl = virus.get("label").lower()
                x = int(float(virus.get("points").split(",")[0]))
                y = int(float(virus.get("points").split(",")[1]))
                lbls.append(lbl)
                coords.append((x,y))
            annotations.append({"filename": filename, "lbls": lbls, "coords": coords})
        return annotations
    
    def __len__(self):
        return len(self.imgs)
    
class TrainDataset(VirusDataset):
    def __init__(self, annotation_file, img_dir, split, transforms = None):
        super().__init__(annotation_file, img_dir, split, transforms = transforms)  
        self.transform_resize = RandomResizeWithLocations(scale=(0.8,1.2))
        self.transform_crop = BalancedRandomCropWithLocations(size=(224, 224), empty_crop_prob=1/len(self.class_names))

        
    def __getitem__(self, idx):
        img = torch.Tensor(self.imgs[idx])
        locations = torch.Tensor(self.annotations[idx]["coords"])
        lbls = np.array(self.annotations[idx]["lbls"])
        
        resized_img, resized_locations = self.transform_resize(img[None,None,...], locations)
        cropped_img, cropped_locations, cropped_lbls = self.transform_crop(resized_img.squeeze(), resized_locations, lbls)
        cropped_img = cropped_img[None,...]
        """print(f"Locations shape: {cropped_locations.shape}")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(cropped_img.squeeze())
        plt.title(", ".join(cropped_lbls))
        if(cropped_locations.shape[0]):
            plt.scatter(cropped_locations[:,0], cropped_locations[:,1])
        plt.show()
        plt.close()"""
        counts = []
        for class_name in self.class_names: 
            counts.append(np.sum(cropped_lbls == class_name))

        counts = torch.Tensor(counts)
        return cropped_img, counts 
        


class InferenceDataset(VirusDataset):
    def __init__(self, annotation_file, img_dir, split, transforms = None):
        super().__init__(annotation_file, img_dir, split, transforms = transforms)  

        self.patches = []
        self.patch_lbls = []
        self.patch_locations = []
        self.num_patch_x = []
        self.num_patch_y = []
        self.img_idx = []
        self.resized_locations = []

        resize_transform = ResizeToMultipleWithLocations(224)

        for idx in range(len(self.imgs)):
            img = torch.Tensor(self.imgs[idx])
            if(self.annotations is None):
                locations = None
                lbls = None
            else:
                locations = torch.Tensor(self.annotations[idx]["coords"])
                lbls = np.array(self.annotations[idx]["lbls"])

            # 1) resize image to a multiple of 224x224 such that we can patchify it into patches of 224x224
            img, resized_locations = resize_transform(img[None,None,...], locations)
            # 2) Patchify image into 224 x 224 image patches to make it fit for the model input.
            patches, patch_locations, patch_lbls, num_patches_x, num_patches_y = patchify_image(img.squeeze(), resized_locations, lbls)

            self.patches.extend(patches) 
            if(not (patch_locations is None)):
                self.patch_lbls.extend(patch_lbls) 
                self.patch_locations.extend(patch_locations.values())
            self.num_patch_x.extend([num_patches_x]*len(patches)) 
            self.num_patch_y.extend([num_patches_y]*len(patches)) 
            self.img_idx.extend([idx]*len(patches)) 
            self.resized_locations.extend([resized_locations])
        # print(f"Inference Dataset now consists of {len(self.imgs)} micrographs and {len(self.patches)} patches.")

    def __len__(self):
        return len(self.patches)
    
    def get_locations_from_idx(self, idx):
        return self.patch_locations[idx]
    
    def get_all_locations_from_idx(self, idx):
        return self.resized_locations[idx]

    def __getitem__(self, idx):
        img = torch.Tensor(self.patches[idx])[None,...]
        normed_img = img # TODO cleanup
        
        if(self.split == "all"):
            return img, normed_img, self.num_patch_x[idx], self.num_patch_y[idx], self.img_idx[idx], idx 
            
        lbls = np.array(self.patch_lbls[idx])

        counts = []
        for class_name in self.class_names: 
            counts.append(np.sum(lbls == class_name))

        counts = torch.Tensor(counts)

        return img, normed_img, counts, self.num_patch_x[idx], self.num_patch_y[idx], self.img_idx[idx], idx 



