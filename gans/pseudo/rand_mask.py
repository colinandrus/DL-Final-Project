import torch.nn as nn
import numpy.random as random

class RandomMasking(nn.Module):
    
    def __init__(self, mask_size=100):
        """
        Args:
            mask_size (int): define the number of width of the pixel block to mask
        """
        super().__init__()
        self.mask_size = mask_size
        
    def forward(self, x):
        if self.mask_size:
            x.detach()
            x_coord = random.randint(0,320 - self.mask_size)
            x_end_cord = x_coord + self.mask_size
            y_coord = random.randint(0,320 - self.mask_size)
            y_end_cord = y_coord + self.mask_size
            for i in range(x.size()[0]):
                x[i,0,y_coord:y_end_cord,x_coord:x_end_cord] = 1.0
                x[i,1,y_coord:y_end_cord,x_coord:x_end_cord] = 1.0
                x[i,2,y_coord:y_end_cord,x_coord:x_end_cord] = 1.0
        return x
