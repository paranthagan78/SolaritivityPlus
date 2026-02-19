"""modules/xai/gradcam.py"""
import torch, torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image
import io, os
from config import INTEGRITY_MODEL_PATH   # reuse same backbone for XAI; swap if needed

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

class _GradCamHook:
    def __init__(self): self.grads = None; self.acts = None
    def save_grad(self, g): self.grads = g
    def save_act(self, m, i, o): self.acts = o

def generate_gradcam(image_path: str, out_path: str) -> str:
    net, device = _get_model()
    hook = _GradCamHook()

    # Register hooks on last conv layer
    target_layer = net.layer4[-1]
    target_layer.register_forward_hook(hook.save_act)
    target_layer.register_backward_hook(lambda m, gi, go: hook.save_grad(go[0]))

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    output = net(tensor)
    pred_class = output.argmax(dim=1).item()
    net.zero_grad()
    output[0, pred_class].backward()

    grads = hook.grads[0].cpu().detach().numpy()         # (C, H, W)
    acts  = hook.acts[0].cpu().detach().numpy()          # (C, H, W)
    weights = grads.mean(axis=(1, 2))                    # (C,)
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Overlay
    orig = cv2.imread(image_path)
    cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    cv2.imwrite(out_path, overlay)

    return out_path