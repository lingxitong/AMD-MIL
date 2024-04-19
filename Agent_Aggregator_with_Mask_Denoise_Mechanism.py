import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class Agent_Aggregator_with_Mask_Denoise_Mechanism(nn.Module):
    def __init__(
        self,
        dim,
        pool=False,
        thresh_tem='linear',
        agent_num=256,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.agent_num = agent_num
        self.noise = nn.Linear(dim_head,dim_head)
        self.mask = nn.Linear(dim_head,dim_head)
        self.get_thresh = nn.Linear(dim,1)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, dim//heads))
        self.pool = pool
        self.thresh_tem = thresh_tem
        if self.thresh_tem == 'cnn':
            self.get_thresh2 = nn.Conv1d(in_channels=dim, out_channels=4, kernel_size=1, groups=4)
            self.get_thresh3 = nn.Linear(4, 1)

    def forward(self, x):
        b, _, _, h = *x.shape, self.heads
        
        # obtain the qkv matrix
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        
        # For comparison with pooling agent
        if self.pool:
            q = q.transpose(-1,-2).squeeze(0)
            agent = nn.AdaptiveAvgPool1d(self.agent_num)(q).transpose(-1,-2)
            q = q.transpose(-1,-2).unsqueeze(0)
        
        # Perform agent calculations
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        softmax = nn.Softmax(dim=-1)
        q *= self.scale
        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v) 
        kv_c = kv.reshape(b,self.agent_num,-1)
        
        # Compare the methods of obtaining threshold
        if self.thresh_tem == 'linear': 
            thresh = self.get_thresh(kv_c).squeeze().mean()
            thresh = F.sigmoid(thresh)
        elif self.thresh_tem == 'mean_pooling':
            thresh = kv_c.mean(dim=1)[0].mean()
            thresh = F.sigmoid(thresh)
        elif self.thresh_tem == 'cnn':
            kv_c = self.get_thresh2(kv_c.transpose(-1,-2)).transpose(-1,-2)
            kv_c = self.get_thresh3(kv_c)
            thresh = kv_c.squeeze().mean()
            thresh = F.sigmoid(thresh)
        
        # Perform mask and denoise operations
        denoise = self.noise(kv)
        denoise = torch.sigmoid(denoise)
        mask = self.mask(kv)
        mask = torch.sigmoid(mask)
        mask = torch.where(mask > thresh, torch.ones_like(mask), torch.zeros_like(mask))
        kv = kv * mask + denoise

        # Obtain weighted features
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out
    
    
if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Agent_Aggregator_with_Mask_Denoise_Mechanism(dim=512).to(device)
    x = torch.randn(1, 1000, 512).to(device)
    out = model(x)
    print(out.shape)
