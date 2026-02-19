"""modules/integrity/integrity_model.py"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from config import INTEGRITY_MODEL_PATH

LABELS = ["invalid", "valid"]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None

def _load():
    global _model
    if _model is not None:
        return _model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VGG19 (same as training)
    net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    # Freeze convolution layers (optional)
    for param in net.features.parameters():
        param.requires_grad = False

    # Replace classifier just like training
    net.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    try:
        state = torch.load(INTEGRITY_MODEL_PATH, map_location=device)

        net.load_state_dict(state, strict=True)

        print("[Integrity] VGG19 model loaded successfully!")

    except Exception as e:
        print("[Integrity] Model load FAILED:", e)

    net.to(device).eval()
    _model = (net, device)
    return _model

def predict_integrity(image_bytes: bytes) -> dict:
    net, device = _load()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().tolist()
    idx = int(torch.argmax(torch.tensor(probs)))
    return {
        "label": LABELS[idx],
        "valid": LABELS[idx] == "valid",
        "confidence": round(probs[idx], 4),
        "scores": {"valid": round(probs[1], 4), "invalid": round(probs[0], 4)},
    }