import streamlit as st
import torch
import os
import cv2
from generator_vanilla_gan import Generator

st.set_page_config(page_title="Synthetic Signature Generator")

st.title("✍️ Synthetic Signature Generator (Vanilla GAN)")

num = st.slider("Number of signatures to generate", 1, 20, 5)

if st.button("Generate Signatures"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "generator.pth")
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G = Generator(100)
    G.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    G.eval()

    st.subheader("Generated Signatures")

    for i in range(num):
        z = torch.randn(1, 100)
        img = G(z).detach().numpy()[0][0]
        img = (img + 1) * 127.5
        img = img.astype("uint8")

        path = os.path.join(OUTPUT_DIR, f"ui_signature_{i+1}.png")
        cv2.imwrite(path, img)

        st.image(path, caption=f"Signature {i+1}", width=150)
