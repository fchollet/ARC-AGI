import torch
import utility
import transformers


def get_encoder(filename, use_device=False):
    """
        Given a filename in the 'trained_models' folder, this loads and returns a vision transformer trained using the iBOT objective

    Args:
        filename (str): Example 'trained_models/vit_20241110_75k.pth'
        use_device (bool): If true, sets device as 'cuda' if available and sends ViT to device
    """
    if use_cuda:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
    else:
        device = 'cpu'
      
    ViT = transformers.VisionTransformer.load_model('trained_models/vit_20241110_75k.pth')
    return ViT.to(device)