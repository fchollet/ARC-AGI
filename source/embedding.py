import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from util import hex_to_rgb, COLOR_TO_HEX



def load_resnet50():
    # Load the pretrained ResNet
    from torchvision.models import resnet50
    resnet50 = resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return resnet50

def get_embedding(image, encoder, display=False):
    """
    Takes an image (numpy array) and returns the embedding of the image using a passed in encoder.

    Args:
        image (numpy.ndarray): The input image.
        display (bool): Prints the image if set to true.
    """
    image_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in image])

    # Convert to PyTorch tensor, ensure it is float32 and permute dimensions to [3, height, width]
    image_rgb = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    # Resize the image to 224x224 using nearest-neighbor to avoid blurring
    ## ISSUE: resizing. Need to fix (if we go down this path)
    image_rgb = F.interpolate(image_rgb, size=(224, 224), mode='nearest')

    # Display the image
    if display:
        image_to_display = image_rgb.squeeze(0).permute(1, 2, 0).numpy()
        image_to_display = (image_to_display * 255).astype(np.uint8)
        plt.imshow(image_to_display)
        plt.show()  

    # Get embedding from ResNet
    with torch.no_grad():
        embedding = encoder(image_rgb)

    # Flatten the output (ResNet output shape will be [batch_size, 2048, 1, 1] if height and width are large enough)
    embedding = embedding.view(embedding.size(0), -1)  # Flatten to [batch_size, 2048]

    # Convert to NumPy and apply PCA
    embedding_np = embedding.cpu().numpy().T
    return embedding_np