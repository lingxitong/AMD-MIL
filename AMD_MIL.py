import torch
import torch.nn as nn
import numpy as np
from .Agent_Aggregator_with_Mask_Denoise_Mechanism import Agent_Aggregator_with_Mask_Denoise_Mechanism

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AMDLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, agent_num=512, tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Agent_Aggregator_with_Mask_Denoise_Mechanism(
            dim = dim,
            pool=pool,
            agent_num=agent_num,
            dim_head = dim//8,
            heads = 8,          
            thresh_tem=thresh_tem,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class PPEG(nn.Module):
    def __init__(self, dim=512,cls_num=1):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.cls_num = cls_num

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, :self.cls_num], x[:, self.cls_num:]
    
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x

class AMD_MIL(nn.Module):
    def __init__(self, n_classes,dropout,act,agent_num=512,cls_num=1,cls_agg='mean', pool=False, thresh=None, thresh_tem='linear', kaiming_init=False):
        super(AMD_MIL, self).__init__()
        self.pos_layer = PPEG(dim=512, cls_num=cls_num)

        self._fc1 = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, cls_num, 512))
        self.cls_num = cls_num
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.amdlayer1 = AMDLayer(dim=512, agent_num=agent_num, pool=pool, thresh=thresh, thresh_tem=thresh_tem, kaiming_init=kaiming_init)
        self.amdlayer2 = AMDLayer(dim=512, agent_num=agent_num, pool=pool, thresh=thresh, thresh_tem=thresh_tem, kaiming_init=kaiming_init)
        self.norm = nn.LayerNorm(512)        
        if cls_agg == 'concat':
            self._fc2 = nn.Linear(512*self.cls_num, n_classes)
        else:
            self._fc2 = nn.Linear(512, n_classes)
        if kaiming_init:
            initialize_weights(self._fc2)
            initialize_weights(self._fc1)
        
        self.cls_agg = cls_agg  
        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float() 
        
        h = self._fc1(h) 
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) 

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.amdlayer1(h) 

        h = self.pos_layer(h, _H, _W) 
        
        h = self.amdlayer2(h)

        h = self.norm(h)[:,:self.cls_num] 
        
        if self.cls_agg == 'mean':
            h = h.mean(1)
        elif self.cls_agg == 'max':
            h = h.max(1)[0]
        elif self.cls_agg == 'sum':
            h = h.sum(1)
        elif self.cls_agg == 'concat':   
            h = h.view(h.size(0), -1)     
        logits = self._fc2(h)
        return logits

if __name__ == "__main__":
    amd_model = AMD_MIL(n_classes=2,dropout=False,act='relu').cuda()
    x = torch.randn(1, 1000, 1024).cuda()
    y = amd_model(x)
    print(y.shape)


