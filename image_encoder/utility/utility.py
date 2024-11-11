import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# ========================================================================
# Loss Functions
# ========================================================================
def compute_loss(u, u_masks, v, v_masks, student_net, teacher_net, nce_weight, device):
    B, H, W = u.shape
    u, u_masks, v, v_masks = u.to(device), u_masks.to(device), v.to(device), v_masks.to(device)

    # Forward pass
    u_s_class, u_s_patch, _ = student_net(u, u_masks)
    u_t_class, u_t_patch, _ = teacher_net(u)
    v_s_class, v_s_patch, _ = student_net(v, v_masks)
    v_t_class, v_t_patch, _ = teacher_net(v)
    
    # Compute loss
    u_masks, v_masks = u_masks.view(B, H*W), v_masks.view(B, H*W)
    patch_rmse = _compute_patch_rmse(u_s_patch, u_t_patch, v_s_patch, v_t_patch, u_masks, v_masks) * (1 - nce_weight)
    class_nce = _compute_class_infoNCE(u_s_class, u_t_class, v_s_class, v_t_class) * nce_weight
    return patch_rmse, class_nce

def _compute_patch_rmse(u_s, u_t, v_s, v_t, u_mask, v_mask):
    """ Computing patch token recovery loss per iBOT """
    u_s, u_t = u_s[u_mask == 1], u_t[u_mask == 1]    # N_1 x C. Notice how it gets rid of concept of batches here (since batches have different number of masked tokens)
    v_s, v_t = v_s[v_mask == 1], v_t[v_mask == 1]    # N_2 x C

    u_loss = torch.sqrt(((u_s - u_t)**2).mean(dim=1)).mean()
    v_loss = torch.sqrt(((v_s - v_t)**2).mean(dim=1)).mean()
    return u_loss + v_loss
    

def _compute_class_rmse(u_s_class, u_t_class, v_s_class, v_t_class):
    """ Computing class token cross-loss per iBOT. Each input is of shape 'B x 1 x C' """
    loss_1 = torch.sqrt(((u_s_class - v_t_class)**2).mean(dim=(1, 2))).mean()
    loss_2 = torch.sqrt(((u_t_class - v_s_class)**2).mean(dim=(1, 2))).mean()
    return loss_1 + loss_2


def _compute_infoNCE_loss(embeddings_1, embeddings_2, temperature=0.5):
    """Compute InfoNCE loss between two sets of embeddings.
       Each input should have shape [B, C], where matching indices are positive pairs."""
    B = embeddings_1.shape[0]    
    embeddings_1 = F.normalize(embeddings_1, dim=1)  # [B, C]
    embeddings_2 = F.normalize(embeddings_2, dim=1)  # [B, C]
    similarity_matrix = torch.matmul(embeddings_1, embeddings_2.T) / temperature  # [B, B]
    labels = torch.arange(B, device=embeddings_1.device)
    loss = F.cross_entropy(similarity_matrix, labels)    
    return loss


def _compute_class_infoNCE(u_s_class, u_t_class, v_s_class, v_t_class, temperature=0.5):
    """Compute class token InfoNCE loss for iBOT. Each input has shape [B, 1, C]."""
    u_s_class, u_t_class = u_s_class.squeeze(1), u_t_class.squeeze(1)
    v_s_class, v_t_class = v_s_class.squeeze(1), v_t_class.squeeze(1)
    
    loss_1 = _compute_infoNCE_loss(u_s_class, v_t_class, temperature)
    loss_2 = _compute_infoNCE_loss(u_t_class, v_s_class, temperature)
    return loss_1 + loss_2


def get_eval_loss(student_net, teacher_net, loader, nce_weight, device, max_eval_iters=None):
    with torch.no_grad():
        eval_patch, eval_class, loss_iters = 0, 0, 0
        for _, (ids, u, u_masks, v, v_masks) in enumerate(loader):          
            loss_iters += 1
            patch_loss, class_loss = compute_loss(u, u_masks, v, v_masks, student_net, teacher_net, nce_weight, device)

            eval_patch += patch_loss.cpu().item()
            eval_class += class_loss.cpu().item()
            if max_eval_iters is not None and loss_iters == max_eval_iters:
                break
    
    return (eval_patch/loss_iters), (eval_class/loss_iters)



# ========================================================================
# Plotting Functions
# ========================================================================
def update_teacher_weights(student_net, teacher_net, ema_alpha):
    """Updates the teacher network parameters using EMA from the student network."""
    with torch.no_grad():
        for student_param, teacher_param in zip(student_net.parameters(), teacher_net.parameters()):
            if isinstance(student_param, torch.nn.Embedding):
                teacher_param.data.copy_(student_param.data)
            else:
                teacher_param.data.mul_(1 - ema_alpha).add_(student_param.data, alpha=ema_alpha)



# ========================================================================
# Plotting Functions
# ========================================================================
COLOR_TO_HEX = {
    -1: '#FF6700',  # blaze orange
    0:  '#000000',  # black
    1:  '#1E93FF',  # blue
    2:  '#F93C31',  # orange
    3:  '#4FCC30',  # green
    4:  '#FFDC00',  # yellow
    5:  '#999999',  # grey
    6:  '#E53AA3',  # pink
    7:  '#FF851B',  # light orange
    8:  '#87D8F1',  # cyan
    9:  '#921231',  # red
    10: '#555555',  # masked token
    11: '#FF6700',  # border
    12: '#000000',  # outside image
}


def hex_to_rgb(hex_color):
    """ Convert a hex color to an RGB tuple with values in the range [0, 1]. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def get_idx(x, y, CLS=False):
    """
    Get the index for a given (x, y) position in a 32x32 grid with an optional CLS token.
    If CLS is True, returns 0. Otherwise, calculates and returns the unrolled index + 1.
    """
    if CLS:
        return 0
    return y * 32 + x + 1  # +1 to account for CLS token


def plot_tensors_with_colors(tensors, title=None):
    """
    Plot an iterable of 2D tensors with optional title.

    Args:
        tensors (iterable): Iterable of 2D tensors to plot.
        title (str, optional): Title for the chart. Defaults to None.
    """
    num_examples = len(tensors)
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 3, 3))
    if num_examples == 1:
        axes = [axes]
    if title is not None:
        fig.suptitle(title)
    for i, tensor in enumerate(tensors):
        tensor_np = tensor.numpy()
        img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
        axes[i].imshow(img_rgb, interpolation='nearest')
        axes[i].axis('off')    
    plt.show()


def plot_attention_map(attention_matrix, idx):
    """
    Given an attention matrix of shape (1025, 1025) and an idx, 
    plot the attention map for that idx as a 32x32 grid and print the CLS token attention score.
    """
    attention_map = attention_matrix[idx, 1:].reshape(32, 32)  # Skip CLS token in the reshaping

    # Plotting our attention map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Score')
    plt.title(f'Attention Map for Index {idx}')
    plt.axis('off')
    plt.show()


def plot_tensor_with_highlight(tensor, idx=None):
    """
    Plots a single 2D tensor with a grid and highlights a specific index with a white border.
    
    Parameters:
    - tensor: a 2D tensor to plot
    - idx: the index to highlight in the 32x32 grid (ignores if idx == 0)
    """
    fig, ax = plt.subplots(figsize=(5, 5))  # Set default size to 6x6

    # Calculate (x, y) for the given idx (ignoring CLS token)
    if idx and idx > 0:
        x, y = (idx - 1) % 32, (idx - 1) // 32  # Convert idx to (x, y) in 32x32 grid
    
    # Convert the tensor to RGB using COLOR_TO_HEX mapping
    tensor_np = tensor.numpy().squeeze().astype(int)  # Ensure 2D and integer type
    img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
    
    # Plot the tensor with grid lines
    ax.imshow(img_rgb, interpolation='nearest')
    
    # Add grid lines
    ax.grid(which="major", color="#777777", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(-.5, 32, 1))
    ax.set_yticks(np.arange(-.5, 32, 1))
    
    # Highlight the selected position if valid
    if idx and idx > 0:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
    
    # Remove axes for a clean look
    ax.axis('off')
    
    plt.show()



# ========================================================================
# Misc
# ========================================================================
def top_k_cosine_similarity(tensor, idx, k, largest=True):
    """
    Compute the cosine similarity of a specified vector (indexed by `idx`) 
    against all other vectors in `tensor` and return the indices and similarity values 
    of the top `k` most similar vectors (or least similar if `largest=False`).

    Args:
        tensor (torch.Tensor): An n x d tensor where n is the number of items and d is the dimensionality.
        idx (int): Index of the vector to compare against.
        k (int): Number of top (or bottom) similar items to return.
        largest (bool): If True, return indices of the top k largest similarities. If False, return smallest.

    Returns:
        torch.Tensor: 1D tensor of indices for the top k most (or least) similar vectors.
        torch.Tensor: 1D tensor of similarity values for the top k most (or least) similar vectors.
    """
    normalized_tensor = torch.nn.functional.normalize(tensor, dim=1)
    reference_vector = normalized_tensor[idx].unsqueeze(0)  # Shape: 1 x d
    cosine_similarities = torch.matmul(normalized_tensor, reference_vector.T).squeeze()  # Shape: n
    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=k, largest=largest)
    return top_k_indices, top_k_values
