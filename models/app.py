import streamlit as st
import torch
import numpy as np
from PIL import Image
from models.pipeline import ISROImageTransformationPipeline

# --- Page Setup ---
st.set_page_config(page_title="ISRO IR-Colorization Framework", layout="wide")
st.title("🛰️ ISRO Infrared Satellite Image Colorization Dashboard")
st.write("---")

# --- Initialize Pipeline ---
@st.cache_resource
def load_master_pipeline():
    # Instantiate your working PyTorch pipeline model
    model = ISROImageTransformationPipeline()
    model.eval() # Put in inference mode
    return model

pipeline = load_master_pipeline()

# --- Sidebar UI ---
st.sidebar.header("📥 Data Upload Panel")
uploaded_file = st.sidebar.file_uploader("Upload single-channel IR Image", type=["png", "jpg", "jpeg"])

# --- Main Render Window ---
if uploaded_file is not None:
    # 1. Load raw image and format it to monochrome
    raw_pil = Image.open(uploaded_file).convert("L")
    w, h = raw_pil.size
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Monochrome IR Input")
        st.image(raw_pil, use_container_width=True)
        process_btn = st.button("🚀 Execute Neural Pipeline", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("AI High-Fidelity RGB Output")
        if process_btn:
            with st.spinner("Executing Super-Resolution & Colorization Modules..."):
                # 2. Transform PIL Image into PyTorch Tensor [1, 1, H, W]
                img_np = np.array(raw_pil, dtype=np.float32) / 255.0
                input_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
                
                # Resize to match our diagnostic input target size dynamically if needed
                input_tensor = torch.nn.functional.interpolate(input_tensor, size=(128, 128))
                
                # 3. Pass through the master framework
                with torch.no_grad():
                    output_tensor = pipeline(input_tensor) # Outputs [1, 3, 256, 256]
                
                # 4. Post-process Tensor back to viewable RGB array
                # Denormalize from [-1, 1] tanh space to [0, 255] integer space
                output_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
                output_np = ((output_np + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                output_pil = Image.fromarray(output_np)
                
                st.image(output_pil, use_container_width=True)
                st.success("Pipeline transformation complete!")
else:
    st.info("Please upload a raw infrared image asset via the sidebar utility to initialize the model pipeline inference.")