import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel

"""
    Class: DownLayer
    Create a down layer in contracting path of Unet model.
"""
class DownLayer(nn.Module):
    """
        Function: __init__
        The initialization function.

        Parameters:
            - in_size: The size of the input tensor.
            - num_filters: The number of output features at the first CONV2D layer.
            - padding: The padding value.
            - batch_norm: Apply the batch normalization.
            - dropout: Apply the Dropout layer.
            - activation: The activation function.

        Returns:
    """
    def __init__(self, in_size, num_filters, padding, batch_norm, dropout, activation):
        super().__init__()

        block = []

        block.append(nn.Conv2d(in_size, num_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        block.append(nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_filters*2))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_filters*2))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        self.block = nn.Sequential(*block)
    """
        Function: forward
        The forward function to perform the computation.

        Parameters:
            - x: The input tensor.

        Returns: The features of input tensor after forwarding the layers.
    """
    def forward(self,x):
        #x = x.cuda()
        return self.block(x)

"""
    Class: UpLayer
    Create an up layer in expanding path of Unet model.
"""
class UpLayer(nn.Module):
    """
        Function: __init__
        The initialization function.

        Parameters:
            - num_in_filters: The number of input filter.
            - up_mode: The upscale mode.
            - padding: The padding value.
            - batch_norm: Apply the batch normalization.
            - dropout: Apply the Dropout layer.
            - activation: The activation function.

        Returns:
    """
    def __init__(self, num_in_filters, up_mode,  padding, batch_norm, dropout, activation):
        super().__init__()
        self.upmode = up_mode
        if up_mode=='upconv':
            self.upconv = nn.ConvTranspose2d(num_in_filters, num_in_filters, kernel_size=2, stride=2)
        else:
            self.upconv = nn.Conv2d(num_in_filters, num_in_filters, kernel_size=1)

        num_out_filters=num_in_filters//2
        block=[]
        block.append(nn.Conv2d(num_in_filters+num_out_filters, num_out_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_out_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_out_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        block.append(nn.Conv2d(num_out_filters, num_out_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_out_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_out_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        self.block = nn.Sequential(*block)

    """
        Function: forward
        The forward function to perform the computation.

        Parameters:
            - x: The input tensor.
            - encoded: The encoded tensor from the layer at the same level in contracting path.

        Returns: The features of input tensor after forwarding the layers.
    """
    def forward(self, x, encoded):
        if self.upmode == 'upsample':
            x = torch.nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)

        up = self.upconv(x)
        out = torch.cat([up, encoded],1)

        return self.block(out)

"""
    Class: Unet2d
    Create an Unet model to use as an Generator in GAN model for transferring the style.
"""
class Unet2d(nn.Module):
    """
        Function: __init__
        The initialization function.

        Parameters:
            - in_cnels: The number of input filter.
            - depth: The depth of Unet model.
            - n_classes: The number of output class.
            - n_base_filters: The number of filters at the first layer.
            - padding: The padding value.
            - batch_norm: Apply the batch normalization.
            - up_mode: The upscale mode.
            - activation: The activation function.
            - final_activation: The final activation function.

        Returns:
    """
    def __init__(self, in_cnels, depth=4, n_classes=3, n_base_filters=32, padding=False, batch_norm=True, up_mode='upconv', activation='relu', final_activation=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        assert activation in ('relu', 'elu')
        self.padding = padding
        self.depth = depth
        self.n_classes = n_classes
        self.final_activation = final_activation
        dropout = True
        #self.out_dim = input_dim[0]
        #in_channels=input_dim[0]
        in_channels = in_cnels

        self.pool = nn.MaxPool2d(2)

        # Encodeur
        self.down_path = nn.ModuleList()
        for i in range(depth):
            block = DownLayer(in_channels, n_base_filters*(2**i), padding, batch_norm, dropout, activation)
            self.down_path.append(block)
            
            in_channels = n_base_filters*(2**i)*2

        # Decodeur
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            block = UpLayer(n_base_filters*(2**(i+2)), up_mode, padding, batch_norm, dropout, activation)
            self.up_path.append(block)

        # Final 1x1x1 convolution
        output_size=n_classes
        if n_classes==2:
            output_size = 1

        self.last = nn.Conv2d(n_base_filters*2, output_size, kernel_size=1)   
        #self.last = CoordConv2d(n_base_filters*2, output_size, kernel_size=1)
        
    """
        Function: forward
        The forward function to perform the computation.

        Parameters:
            - x: The input tensor.

        Returns: The features of input tensor after forwarding the layers.
    """
    def forward(self,x):
        blocks_out= []
        
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i!= len(self.down_path)-1:
                blocks_out.append(x)
                
                x = self.pool(x)
        #print("End encoder ====================================")
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks_out[self.depth-i-2])
            
        #print("End decoder ====================================")
        
        if self.final_activation:
            if self.n_classes==2:
                return nn.Sigmoid()(self.last(x))
            return nn.Tanh()(self.last(x))
        else:
            return self.last(x)

#====================== Discriminator ==================
"""
    Class: PatchGANDiscriminator
    Create Patch discriminator.
"""
class PatchGANDiscriminator(nn.Module):
    """
        Function: __init__
        The initialization function.

        Parameters:
            - in_size: The number of channel of the input tensor.

        Returns:
    """
    def __init__(self, in_size):
        super(PatchGANDiscriminator, self).__init__()
        self.d1 = nn.Sequential(nn.Conv2d(in_size, 32, kernel_size=4, stride=2, padding=1),
                                nn.GroupNorm(num_groups=4, num_channels = 32),
                                nn.PReLU())
        self.d2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                nn.GroupNorm(num_groups=4, num_channels = 64),
                                nn.PReLU())
        self.d3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                nn.GroupNorm(num_groups=4, num_channels = 128),
                                nn.PReLU())
        self.d4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                nn.GroupNorm(num_groups=4, num_channels = 256),
                                nn.PReLU())
        self.final = nn.Conv2d(256, 1, kernel_size = 1)

    """
        Function: forward
        The forward function to perform the computation.

        Parameters:
            - x: The source tensor.
            - y: The target tensor.

        Returns: The decsion that x and y are real or fake.
    """
    def forward(self, x, y):
        x = torch.cat([x,y], axis = 1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn
