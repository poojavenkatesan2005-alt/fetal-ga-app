import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Fetal GA Estimator")
st.title("Fetal Gestational Age Estimation from 2D Ultrasound")
st.caption("Research prototype. Not a medical device.")


@st.cache_resource
def load():
    cfg = json.load(open("config.json"))
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
    sd = torch.load("best_model_small.pt", map_location="cpu")
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


model, cfg = load()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SignedTarget:
    """Grad-CAM target for a single regression output.
    For ages below the training average the output is negative, so we flip
    the sign. Otherwise the heatmap can come out almost empty."""

    def __init__(self, sign):
        self.sign = sign

    def __call__(self, model_output):
        return self.sign * model_output[:, 0]


file = st.file_uploader(
    "Upload a fetal head ultrasound image (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
)

if file:
    img = Image.open(file).convert("RGB").resize((224, 224))
    x = tf(img).unsqueeze(0)

    # Grad-CAM (dropout off)
    model.eval()
    with torch.no_grad():
        base_out = model(x).item()
    sign = 1 if base_out >= 0 else -1
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        heat = cam(input_tensor=x, targets=[SignedTarget(sign)])[0]
    rgb = np.array(img).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    # MC Dropout (dropout on, 30 runs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    with torch.no_grad():
        preds = torch.stack([model(x).squeeze() for _ in range(30)])
    model.eval()

    weeks = preds.mean().item() * cfg["std"] + cfg["mean"]
    unc_days = preds.std().item() * cfg["std"] * 7
    half_width = 1.96 * unc_days / 7

    c1, c2 = st.columns(2)
    c1.image(img, caption="Input image")
    c2.image(overlay, caption="Grad-CAM: where the model looked")

    st.metric(
        "Estimated gestational age",
        f"{weeks:.1f} weeks",
        f"95% interval: ± {half_width:.1f} weeks",
        delta_color="off",
    )

    if unc_days > cfg["unc_threshold_days"]:
        st.warning(
            "High uncertainty. This estimate may be unreliable and should be "
            "reviewed by an expert."
        )
    else:
        st.success(
            "Uncertainty is relatively low compared with the test images. "
            "The interval above still shows the expected error."
        )
