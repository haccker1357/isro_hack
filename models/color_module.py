import torch
import torch.nn as nn

class ColorizationGenerator(nn.Module):
    """
    A Pix2Pix-style translation network that transforms 1-channel IR structures
    into a 3-channel realistic RGB image matrix.
    """
    def __init__(self, in_channels=1, out_channels=3):
        super(ColorizationGenerator, self).__init__()
        
        # Simple Encoder (Downsampling to understand geography context)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Simple Decoder (Upsampling and painting colors)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Scales output pixels between [-1, 1] for easy RGB rendering
        )

    def forward(self, x):
        # Forward pass through the translator layers
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        out = self.dec2(d1)
        return out

if __name__ == "__main__":
    # Diagnostic test: Simulate a batch of 4 high-res enhanced IR images (256x256)
    dummy_enhanced_ir = torch.randn(4, 1, 256, 256)
    
    colorizer = ColorizationGenerator()
    rgb_output = colorizer(dummy_enhanced_ir)
    
    print("\n--- Colorization Module Sanity Check Success ---")
    print(f"Input Shape (Enhanced IR): {dummy_enhanced_ir.shape}")
    print(f"Output Shape (Color RGB):  {rgb_output.shape}") # Expecting 3 color channels