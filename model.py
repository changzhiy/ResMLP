import torch.nn as nn
import torch

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        return self.alpha * x + self.beta


class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*dim, dim)
    def forward(self,x):
        return self.fc2(self.act(self.fc1(x)))


class ResMLP_Blocks(nn.Module):
    def __init__(self,nb_patches,dim,layersscale_init):
        super().__init__()
        self.affine1 = Affine(dim)
        self.affine2 = Affine(dim)
        self.linear_patchs = nn.Linear(nb_patches, nb_patches)
        self.mlp_channels = MLP(dim)
        self.layerscale_1 = nn.Parameter(layersscale_init*torch.ones(dim))
        self.layerscale_2 = nn.Parameter(layersscale_init*torch.ones(dim))

    def forward(self,x):
        res1 = self.linear_patchs(self.affine1(x).transpose(1,2)).transpose(1,2)
        x = x+ self.layerscale_1*res1
        res_2 = self.mlp_channels(self.affine2(x))
        x = x+ self.layerscale_2*res_2


# Patch_projector is not defined in the paper
class ResMLP_models(nn.Module):
    def __init__(self,dim,depth,nb_patches,layerscale_init,num_classes):
        super().__init__()
        # self.patch_projector = Patch_projector()
        self.blocks = nn.ModuleList(
            [ResMLP_Blocks(nb_patches,dim,layerscale_init) for _ in range(depth)]
        )
        self.linear_classifier = nn.Linear(dim,num_classes)

    def forward(self,x):
        B,C,H,W = x.shape
        # x = self.patch_projector(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.affine(x)
        x = x.mean(dim=1).reshape(B,-1)
        return self.linear_classifier(x)

