import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class SatelliteSRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_blocks=4, upscale_factor=2):
        super(SatelliteSRNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualDenseBlock(nf=64) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv = nn.Conv2d(64, 64, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv_last = nn.Conv2d(64 // (upscale_factor ** 2), out_channels, 3, 1, 1)

    def forward(self, x):
        fea_first = self.conv_first(x)
        fea_body = self.conv_body(self.body(fea_first))
        fea = fea_first + fea_body
        out = F.leaky_relu(self.upconv(fea), negative_slope=0.2)
        out = self.pixel_shuffle(out)
        out = self.conv_last(out)
        return out

class EdgePreservingLoss(nn.Module):
    def __init__(self):
        super(EdgePreservingLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pixel_loss = self.l1_loss(pred, target)
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        edge_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        return pixel_loss + 0.5 * edge_loss

if __name__ == "__main__":
    dummy_low_res_ir = torch.randn(4, 1, 128, 128)
    model = SatelliteSRNet(upscale_factor=2)
    criterion = EdgePreservingLoss()
    enhanced_output = model(dummy_low_res_ir)
    print("\n--- SR Module Sanity Check Success ---")
    print(f"Input Shape:  {dummy_low_res_ir.shape}")
    print(f"Output Shape: {enhanced_output.shape}")