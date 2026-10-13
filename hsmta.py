import torch
from torch import nn 


class BMM_Block(nn.Module):
    def __init__(self, dim: int=0):
        super(BMM_Block, self).__init__()
        
        self.net_layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.net_layers = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
    def forward(self, x):
        x = self.net_layers(x)
        return x 
    
    
class BMM_Seq(nn.Sequential): 
    def __init__(self, dim: int=0, depth: int = 12): 
        super().__init__(*[BMM_Block(dim=dim) for _ in range(depth)])
        

def unfold(input_images, window_size, stride):
    
    batch_size = input_images.size(0)
    patches = torch.nn.functional.unfold(input_images, window_size, stride=stride)
    
    num_patches = patches.shape[2]
    num_channels = input_images.shape[1]    
    patches = patches.permute(0,2,1)
    patches = patches.reshape(batch_size, num_patches, num_channels, window_size, window_size)

    return patches

def fold(patches, original_size=0, window_size=0, stride=0):
    b,p,c,h,w = patches.size()
    patches = patches.reshape(b,p,c*h*w)
    patches = patches.permute(0,2,1)
    return torch.nn.functional.fold(
                patches, (original_size,original_size), window_size, stride=stride
            )
    
    

class AggregationBlock(nn.Module):
    def __init__(self, dim_in: int=0, dim_middle: int=0, dim_out: int=0, stride: int=0, downsample = None):
        super(AggregationBlock, self).__init__()
        
        self.net_layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_middle, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_middle, dim_middle, kernel_size=(3, 3), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_middle, dim_out, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.net_layers(x)
        
        res = self.downsample(res)      
        
        return x + res
    
    


class RefineBlock(nn.Module):
    def __init__(self, dim_1: int=0, dim_2: int=0):
        super(RefineBlock, self).__init__()
        
        self.net_layers = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_2, dim_2, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_2, dim_1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x=self.net_layers(x)
        return x
    
class DownsamplingBlock(nn.Sequential): 
    def __init__(self,      
                in_dim: int = 0, 
                out_dim: int = 0,
                kernel_size: int=0,
                stride: int = 0
                ): 
        super().__init__( 
            nn.Sequential(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
                nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        )    




class Agn(nn.Module):
    def __init__(self, window=2, stride=2, depth=2, 
                 dim_in=64, dim_middle=256, dim_out=256):
        super(Agn, self).__init__()
        self.window = window
        self.stride = stride

        #BMM
        self.BMM_Seq = BMM_Seq(dim=window*window, depth=depth)
        
         
        # Spatial merge and refine layers
        self.spatial_merge_and_refine = nn.Sequential(
            nn.Conv2d(dim_in, dim_middle, kernel_size=(window, window), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_middle, dim_middle, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_middle, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(dim_middle, dim_out, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dim_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )        

    def forward(self, x):
        
        shape=x.size(-1)
        x = unfold(x,self.window, self.stride)
        b, p, c, h, w=x.size()
        x=x.reshape(b*p, c, h*w)
        x=x.permute(0,2,1).unsqueeze(-1)
        x = self.BMM_Seq(x)
        x=x.squeeze()
        x = x.permute(0,2,1)
        x=x.reshape(b, p, c, h, w)
        
        x = fold(x, shape, self.window, self.stride)
        
        x = self.spatial_merge_and_refine(x)
        
        return x
    
    

class HSMTA(nn.Module):
    def __init__(self, num_class:int=1000, input_dim:int=3):
        super(HSMTA, self).__init__()        
        
        self.stretch = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        
        # Stage 1
        self.Agn1 = Agn(window=3, stride=3, depth=2, 
                 dim_in=32, dim_middle=16, dim_out=64)
        
        # Stage 2
        self.Agn2 = Agn(window=2, stride=2, depth=2, 
                 dim_in=64, dim_middle=32, dim_out=128)
        
        # Stage 3
        self.Agn3 = Agn(window=2, stride=2, depth=2, 
                 dim_in=128, dim_middle=64, dim_out=256)

        # Stage 3
        self.Agn4 = nn.Sequential(
            AggregationBlock(dim_in=256, dim_middle=128, dim_out=512, stride=1,
                              downsample=DownsamplingBlock(in_dim=256, out_dim=512, kernel_size=3, stride=1)),
            RefineBlock(dim_1=512,dim_2=256),
            RefineBlock(dim_1=512,dim_2=256)
            )
            
        
        self.GlobalPooling = nn.AdaptiveAvgPool2d((1,1))
        
        self.Classifier = nn.Sequential(nn.Linear(512, num_class))                          
    


    def forward(self, x):
        x = self.stretch(x)
        
        x = self.Agn1(x)
        x = self.Agn2(x)
        x = self.Agn3(x)
            
        x = self.Agn4(x)
        x = self.GlobalPooling(x).view(x.size(0), -1)
        x = self.Classifier(x)
        
        return x
    
   