import os
import torch
import numpy as np
import cv2
import gradio as gr
from models.unet import UNet
from utils.visualization import overlay_mask
from data.dataset import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create necessary output directories
os.makedirs("enhanced_images", exist_ok=True)
os.makedirs("debug_images", exist_ok=True)

def load_model(model_path):
    """Load trained model"""
    model_path = model_path.replace("\\", "/")
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=1).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded successfully with validation IoU: {checkpoint.get('val_iou', 'N/A')}")
        model.eval()
        return model, device
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def process_image(image, model, device):
    """Process input image for reflection detection"""
    if image is None:
        print("Error: No image provided")
        return None, None, None

    try:
        # Debug: save input image
        debug_path = "debug_images/input.png".replace("\\", "/")
        cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Preprocess for model
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)
        transformed = transform(image=image)
        model_input = transformed["image"].unsqueeze(0).to(device)

        # Inference with test-time augmentation
        with torch.no_grad():
            outputs = model(model_input)
            flipped_h = torch.flip(model_input, dims=[3])
            flipped_v = torch.flip(model_input, dims=[2])
            pred_h = torch.flip(model(flipped_h), dims=[3])
            pred_v = torch.flip(model(flipped_v), dims=[2])
            avg_pred = (outputs + pred_h + pred_v) / 3.0

        # Post-processing
        pred_np = torch.sigmoid(avg_pred).squeeze().cpu().numpy()
        pred_resized = cv2.resize(pred_np, (image.shape[1], image.shape[0]))
        binary_mask = (pred_resized > 0.5).astype(np.uint8) * 255
        cv2.imwrite("debug_images/binary_mask.png", binary_mask)

        # Visualizations
        mask_colored = np.zeros_like(image)
        mask_colored[binary_mask > 127] = [255, 255, 255]
        cv2.imwrite("debug_images/mask_colored.png", cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))

        overlay = overlay_mask(image, binary_mask, color=[255, 0, 0], alpha=0.7)
        save_path = f"enhanced_images/highlighted_{len(os.listdir('enhanced_images'))}.png"
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_images/overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return image, mask_colored, overlay

    except Exception as e:
        print(f"Error processing image: {e}")
        error_img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error processing image", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return error_img, error_img, error_img

def create_interface():
    """Create Gradio interface for reflection detection"""
    model_path = "models/final_model.pth".replace("\\", "/")
    if not os.path.exists(model_path):
        model_path = "models/initial_model.pth".replace("\\", "/")
    if not os.path.exists(model_path):
        print("No model found. Please train the model first.")
        return None

    result = load_model(model_path)
    if result is None:
        print("Failed to load model.")
        return None
    model, device = result

    # Gradio interface
    with gr.Blocks(title="Specular Reflection Detection") as app:
        gr.Markdown("# Specular Reflection Detection")
        gr.Markdown("Upload an endoscopic image to detect and highlight reflections.")

        with gr.Row():
            input_image = gr.Image(label="Upload Image", type="numpy")

        with gr.Row():
            detect_btn = gr.Button("Detect Reflections", variant="primary")

        with gr.Row():
            original_output = gr.Image(label="Original Image")
            reflection_output = gr.Image(label="Reflection Detection")
            overlay_output = gr.Image(label="Overlay Visualization")

        def process_and_display(img):
            if img is None:
                error_img = np.zeros((320, 320, 3), dtype=np.uint8)
                cv2.putText(error_img, "Please upload an image", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return error_img, error_img, error_img
            return process_image(img, model, device)

        detect_btn.click(
            fn=process_and_display,
            inputs=[input_image],
            outputs=[original_output, reflection_output, overlay_output]
        )

        gr.Markdown("""
        ## About This Tool
        This tool detects specular reflections in endoscopic images and provides visualization:

        - **Original Image**: The input endoscopic image  
        - **Reflection Detection**: Shows areas with specular reflections in white  
        - **Overlay Visualization**: Original image with reflection areas highlighted in blue  
        All highlighted images are saved to the 'enhanced_images' directory.
        """)

    return app

if __name__ == "__main__":
    app = create_interface()
    if app:
        app.launch(share=True)
    else:
        print("Could not launch application. Please check if model files exist and are valid.")
