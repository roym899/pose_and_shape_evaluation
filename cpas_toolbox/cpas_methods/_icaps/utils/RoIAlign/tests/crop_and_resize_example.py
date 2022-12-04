import torch
from torch import nn
from torchvision import transforms, utils
from torch.autograd import Variable, gradcheck
import matplotlib.pyplot as plt
from skimage.io import imread
from layer_utils.roi_layers import ROIAlign, ROIPool

def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

def roi_align_layer(image, rois):
    return ROIAlign((256, 256), 1.0, 0)(image, rois)

crop_height = 500
crop_width = 500
is_cuda = torch.cuda.is_available()

# In this simple example the number of images and boxes is 2
img_path1 = './utils_dpf/RoIAlign/tests/images/choco.png'
img_path2 = './utils_dpf/RoIAlign/tests/images/snow.png'

# Define the boxes ( crops )
# box = [y1/heigth , x1/width , y2/heigth , x2/width]
boxes_data = torch.FloatTensor([[1, -20, -20, 0.5 * crop_width, 1.0 * crop_height],
                                [1, 0, 0, 1.0 * crop_width, 0.5 * crop_height]])
boxes_data.requires_grad = True

# Create an index to say which box crops which image
box_index_data = torch.IntTensor([1, 1])

# Import the images from file
image_data1 = transforms.ToTensor()(imread(img_path1)).unsqueeze(0)
image_data2 = transforms.ToTensor()(imread(img_path2)).unsqueeze(0)

# Create a batch of 2 images
image_data = torch.cat((image_data1, image_data2), 0)

# Convert from numpy to Variables
image_torch = to_varabile(image_data, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, is_cuda=is_cuda)

crops_torch = roi_align_layer(image_data, boxes_data)

print(crops_torch.size())

# Visualize the crops
print(crops_torch.data.size())
crops_torch_data = crops_torch.data.cpu().numpy().transpose(0, 2, 3, 1)
original_images = image_data.data.cpu().numpy().transpose(0, 2, 3, 1)
fig = plt.figure()
plt.subplot(221)
plt.imshow(original_images[0])
plt.subplot(222)
plt.imshow(original_images[1])
plt.subplot(223)
plt.imshow(crops_torch_data[0])
plt.subplot(224)
plt.imshow(crops_torch_data[1])
plt.show()
