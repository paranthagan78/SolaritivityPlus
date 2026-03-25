"""modules/xai/lime_xai.py"""
import torch, torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image
import os
from config import INTEGRITY_MODEL_PATH   # reuse same backbone for XAI; swap if needed

from lime import lime_image
from skimage.segmentation import mark_boundaries

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None

def _get_model():
    global _model
    if _model: return _model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    try:
        state = torch.load(INTEGRITY_MODEL_PATH, map_location=device)
        net.load_state_dict(state if not isinstance(state, dict) or "model" not in state else state["model"], strict=False)
    except Exception:
        pass
    net.to(device).eval()
    _model = (net, device)
    return _model

def batch_predict(images: np.ndarray) -> np.ndarray:
    """
    LIME requires a batch prediction function that takes numpy array of images
    (N, H, W, C) in [0, 255] range and returns probabilities (N, num_classes).
    """
    net, device = _get_model()
    
    # Process batch
    batch_tensor = []
    for img_arr in images:
        pil_img = Image.fromarray(img_arr.astype("uint8")).convert("RGB")
        tensor = TRANSFORM(pil_img)
        batch_tensor.append(tensor)
        
    batch_tensor = torch.stack(batch_tensor).to(device)
    
    with torch.no_grad():
        logits = net(batch_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        
    return probs.cpu().numpy()

def generate_lime(image_path: str, out_path: str, num_samples: int = 250) -> str:
    """
    Generate LIME explanation for the given thermal image and save the heatmap overlay.
    """
    net, device = _get_model()
    
    # Read original image via OpenCV to get array in [0, 255] RGB format
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Get a single prediction to know which class we're explaining
    probs = batch_predict(np.expand_dims(orig_rgb, axis=0))
    pred_class = int(np.argmax(probs[0]))
    
    # Set up LIME Explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    # Note: num_samples can be increased for more stability, but 250-500 is good for web speed
    explanation = explainer.explain_instance(
        orig_rgb, 
        batch_predict,
        top_labels=1, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Get mask of superpixels with weights for the predicted class
    # We want a heatmap representation of these weights
    dict_heatmap = dict(explanation.local_exp[pred_class])
    segments = explanation.segments # (H, W) array of superpixel segments
    
    # Create the weight heatmap corresponding to segment mask
    heatmap = np.zeros_like(segments, dtype=float)
    for k, v in dict_heatmap.items():
        heatmap[segments == k] = v
        
    # Scale heatmap properly for coloring
    # LIME gives positive (warm) and negative (cool) weights
    # Normalize weights so that 0 is at mapped value 0.5
    max_val = np.max(np.abs(heatmap))
    if max_val > 1e-8:
        heatmap_norm = 0.5 + (heatmap / (2.0 * max_val))
    else:
        heatmap_norm = np.full_like(heatmap, 0.5)
        
    heatmap_norm = np.clip(heatmap_norm, 0, 1)
    
    # Resize heatmap to match original image if needed (Lime segmenter works on original resolution)
    heatmap_resized = cv2.resize(heatmap_norm, (orig_bgr.shape[1], orig_bgr.shape[0]))
    
    # Apply JET color map
    # 0 -> blue (strong negative influence)
    # 0.5 -> yellow-green (neutral influence)
    # 1.0 -> red (strong positive influence)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Optionally draw superpixel boundaries subtly
    temp, mask = explanation.get_image_and_mask(pred_class, positive_only=False, num_features=5, hide_rest=False)
    boundaries = mark_boundaries(np.zeros_like(orig_rgb), mask)
    
    # Overlay heatmap over original BGR
    # Weighting: orig 60%, heatmap 40%
    overlay = cv2.addWeighted(orig_bgr, 0.60, heatmap_color, 0.40, 0)
    
    # Add boundary lines to overlay
    # mark_boundaries returns float [0, 1]. Get border pixels and color them yellow
    border_mask = np.any(boundaries > 0, axis=-1)
    overlay[border_mask] = [0, 255, 255] # Yellow boundaries
    
    cv2.imwrite(out_path, overlay)
    
    return out_path
