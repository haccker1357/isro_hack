import torch
import torch.nn as nn
from models.sr_module import SatelliteSRNet
from models.color_module import ColorizationGenerator

class ISROImageTransformationPipeline(nn.Module):
    """
    The end-to-end master framework combining Structural Enhancement (Super-Resolution)
    and Realistic Colorization into a unified forward pass.
    """
    def __init__(self):
        super(ISROImageTransformationPipeline, self).__init__()
        # Load your sub-modules
        self.sr_enhancer = SatelliteSRNet(upscale_factor=2)
        self.colorizer = ColorizationGenerator()

    def forward(self, raw_ir_tensor):
        # Step 1: Upscale and sharpen edges of the low-res raw IR band
        enhanced_ir = self.sr_enhancer(raw_ir_tensor)
        
        # Step 2: Translate the high-res monochrome structure to realistic RGB
        final_rgb = self.colorizer(enhanced_ir)
        
        return final_rgb

if __name__ == "__main__":
    # Diagnostic test simulating a 1-channel raw IR satellite tile matrix
    raw_input_tile = torch.randn(1, 1, 128, 128)
    
    # Initialize the complete end-to-end framework
    pipeline = ISROImageTransformationPipeline()
    
    with torch.no_grad():
        output_rgb_product = pipeline(raw_input_tile)
        
    print("\n🎉 --- MASTER PIPELINE SANITY CHECK SUCCESS ---")
    print(f"Raw Input IR Form:     {raw_input_tile.shape}  (Low-res, Monochrome)")
    print(f"Final Output RGB Form:  {output_rgb_product.shape} (High-res, Full Color)")