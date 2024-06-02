import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F
import timm

from utils.inc_net import BaseNet
from convs.linears import SimpleLinear
#todo 1. L2P, DP 구현하고 성능 체크하기 Dataset 추가 확보하기!
#todo 2. Our --> Hallucination & Mesmerization Feasibility 체크
import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F
import timm

from utils.inc_net import BaseNet
from convs.linears import SimpleLinear
#todo 1. L2P, DP 구현하고 성능 체크하기 Dataset 추가 확보하기!
#todo 2. Our --> Hallucination & Mesmerization Feasibility 체크
class ViTNet(nn.Module):
    def __init__(self, args, pretrained):
        super(ViTNet, self).__init__()
        # self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True,drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.out_dim = 768
        self.backbone.head = nn.Identity()
        self.fc_mask = None
        self.fc = None

    @property
    def feature_dim(self):
        return self.out_dim

    def backbone_forward(self, x):
        fmap = []
        x = self.backbone.patch_embed(x)
        #! Without [CLS] Token
        x = self.backbone._pos_embed(x)
        for block in self.backbone.blocks:
            x = block(x)
            fmap.append(x)
        
        x = self.backbone.norm(x)
        
        #* Features : (Dim) / Img_Features : (196, Dim) / All_features : (197, Dim)
        return {'fmaps': fmap, 'features':x[:,0],'img_features':x[:,1:], 'all_features':x}

    def extract_vector(self, x, img_feats=False):
        extract_feats = self.backbone_forward(x)
        
        if img_feats:
            return extract_feats["features"], extract_feats["img_features"]
        else:
            return extract_feats["features"]
    
    def forward(self, x, task_id=None):
        x = self.backbone_forward(x)
        
        #* Forward Img feature
        if task_id is not None:
            out = self.fc(x["features"], self.fc_mask)
        else:
            out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        fc_mask = torch.zeros(nb_classes, requires_grad=False)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            # fc.bias.data[:nb_output] = bias
            fc.old_nb_classes = nb_output
            
            fc_mask[:nb_output] = -float('inf')
        
        del self.fc
        del self.fc_mask
        
        
        # self.fc_mask
        self.fc = fc
        self.fc_mask = fc_mask
        masked, unmasked = 0, 0
        for item in self.fc_mask:
            if item == 0:
                unmasked +=1
            else:
                masked +=1
        logging.info("self.fc: {}".format(self.fc.weight.shape))
        logging.info("[Mask]: {} [Unmasked]: {}".format(masked, unmasked))
        # pass

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

class Ours_Net(ViTNet):
    def __init__(self, args, pretrained):
        super(Ours_Net, self).__init__(args, pretrained)
        
        #todo A-Component
        # self.m_length = args['m_length'] #* 20
        # self.m_pos = args['m_pos'] #* 일단은 Single pos --> 성능 보고 Multi Layer로 바꿔보자!
        # self.m_component = torch.nn.Parameter(torch.randn(len(self.m_pos), 2*self.m_length, self.feature_dim))
        # torch.nn.init.uniform_(self.m_component, -1, 1)
        
        self.pre_length = args['pre_length']
        self.pre_pos = args['pre_pos']
        self.n_tkns = 197
        self.pre_prompt = torch.nn.Parameter(torch.randn(len(self.pre_pos), 2*self.pre_length, self.feature_dim))
        torch.nn.init.uniform_(self.pre_prompt, -1, 1)
        #* self.pre_component = torch.nn.Parameter(torch.randn(len(self.pre_pos), self.pre_length, self.n_tkns))
        # self.pre_component = torch.nn.Parameter(torch.randn(len(self.pre_pos), self.pre_length, self.pre_length))
        # torch.nn.init.uniform_(self.pre_component, -1, 1)
        
        #todo Locality_induction (Conv Layer)
        
        #* A_component --> CLS-Prompt (Prompting tuning) / Res-Prompt (Skip Connection)
        
        self.cls_length = args["cls_length"]
        #! self.cls_component = torch.nn.Parameter(torch.randn(self.cls_length, self.feature_dim))
        self.cls_component = torch.nn.Parameter(torch.randn(self.cls_length, self.cls_length))
        torch.nn.init.uniform_(self.cls_component, -1, 1)
        
        self.res_pos = args["res_pos"]
        #! self.res_component = torch.nn.Parameter(torch.randn(len(self.res_pos), self.cls_length, self.feature_dim))
        self.res_component = torch.nn.Parameter(torch.randn(len(self.res_pos), self.cls_length, self.cls_length))
        torch.nn.init.uniform_(self.res_component, -1, 1)
    
    def prefix_tuning(self,x, prefix_prompt, block):
        #? prefix_prompt : B, 2, length, dim
        
        # #todo prompt: 2*self.s_length, self.feature_dim
        # p_length,dim = prompt.shape
        # prompt = prompt.reshape(2, int(p_length/2), dim)
        # B,N,C = x.shape
        # prefix_token = prompt.expand(B, 2, int(p_length/2), dim)

        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        xk = torch.cat([xk, prefix_prompt[:,0]],dim=1)
        xv = torch.cat([xv, prefix_prompt[:,1]],dim=1)
        
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        
        B, N, C = xq.shape
        xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        #* Attention Layer Forward
        attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        attention = attention.softmax(dim=-1)
        attention = attn.attn_drop(attention)

        attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
        attention = attn.proj(attention)
        attention = attn.proj_drop(attention)
        #*------------------------

        x = x + block.drop_path1(block.ls1(attention))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x

    def a_tuning(self, x, block, prompt, q_idx):
        #todo 일단은 Simple하게..
        B, N, C = x.shape
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        
        #todo clz-Prompt for Better Plasticity 
        #todo --> Value Weight SVD Decomposition --> Clz-Tuning
        
        # if idx in self.clz_pos:
        # B,N,C = x.shape
        # mask_idx = torch.randperm(N)[:self.clz_length]
        # masked_x = x.clone()
        # masked_x[:,mask_idx] = self.clz_prompt[self.clz_pos.index(b_idx)]
        
        xq = block.norm1(x) #* CLS + G-Prompt + Img..
        xk = xq.clone()
        xv = xq.clone()
        #* Original
        #* -----------------------------------------------
        B, N, C = xq.shape
        xq = F.linear(xq, weight[:C], bias[:C]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        #* -----------------------------------------------
        
        #? [Update-2] Append Q-Prompt Only 
        #? -----------------------------------------------
        #? self.q_prompt = (leng_q, 196, self.feature_dim)
        # B, N, C = xq.shape
        # q_score = 1. - torch.cosine_similarity(xq[:,1:], self.q_prompt[q_idx][None,:,:], dim=-1)
        # xq = torch.cat([xq[:,0].unsqueeze(1), xq[:,1:] + self.q_prompt[q_idx][None,:,:]],dim=1)
        # xq = F.linear(xq, weight[:C], bias[:C]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        #? -----------------------------------------------
        
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C:], bias[2*C:]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        
        # _B, _N, _C = xv.shape
        # ori_v = weight[2*C:].data
        # idx = torch.randperm(ori_v.shape[0])[:self.clz_length]
        # ori_v[idx] =  self.clz_prompt
        # xv = F.linear(xv, ori_v).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        #* xq: torch.Size([32, 12, 197, 64]) xk: torch.Size([32, 12, 197, 64]) xv: torch.Size([32, 12, 197, 64]) 
        
        #* Attention Layer Forward
        ori_attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        ori_attention = ori_attention.softmax(dim=-1)
        ori_attention = attn.attn_drop(ori_attention)
        ori_attention = (ori_attention @ xv).transpose(1, 2)
        
        ori_attention = ori_attention.reshape(B, N, C)
        ori_attention = attn.proj(ori_attention)
        ori_attention = attn.proj_drop(ori_attention)
        #*------------------------
        #todo----------------------------
        #todo 1. SVD-Tuning --> Failure
        # with torch.no_grad():
        #     U,S,Vt = torch.linalg.svd(ori_attention.mean(dim=0), full_matrices=False)
        # #* Composition: U @ torch.diag_embed(S) @ Vh
        # prompted_Vt = torch.cat([Vt[:-self.clz_length],self.clz_prompt], dim=0)
        # Proto_attention = U @ torch.diag_embed(S) @ prompted_Vt
        # # print("proto:", Proto_attention.shape)
        #todo----------------------------
        #todo 2. Fisher-Tuning
        
        #todo----------------------------
        
        #todo A-Prompt tuning
        #! Head를 추가하는 방향으로 Prompt를 튜닝함...
        #! 추가된 Prompt head간 Orthogonal 학습은 Later..
        #? self.a_prompt = B, 2, a_leng, dim
        #? self.a_prompt = B,len(pos), 2, a_leng, dim
        
        #*B, _, _, a_N, dim = self.a_prompt.shape
        #*a_k = self.a_prompt[:,p_idx, 0].reshape(B,a_N,attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        #*a_v = self.a_prompt[:,p_idx, 1].reshape(B,a_N,attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        
        
        
        
        
        #* Prompt => B, 2 (Key, Value), length, dim
        B, _, a_N, dim = prompt.shape
        #* Original
        # a_k = prompt[:, 0].reshape(B,a_N,attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        # a_v = prompt[:, 1].reshape(B,a_N,attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        #*-----------------------------------------------------------------------------------------
        #? [Update-1]Append Weight linear transform 
        #? -----------------------------------------------
        a_k = F.linear(prompt[:, 0], weight[C:2*C], bias[C:2*C]).reshape(B, a_N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        a_v = F.linear(prompt[:, 1], weight[2*C:], bias[2*C:]).reshape(B, a_N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        #? -----------------------------------------------
        
        #? [Update-1.3]Append K and V Linear forward to Prompt  --> Poor..
        #? -----------------------------------------------
        # _B, _N, _C = x.shape
        # pk = prompt[:,0].transpose(2,1) @ prompt[:,0]
        # pv = prompt[:,1].transpose(2,1) @ prompt[:,1]
        # # print(pk.shape)
        # a_k = (block.norm1(x) @ pk).reshape(B, _N, attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        # a_v = (block.norm1(x) @ pv).reshape(B, _N, attn.num_heads, dim // attn.num_heads).permute(0, 2, 1, 3)
        # print(a_k.shape)
        #? -----------------------------------------------
        
        # print('[A-Tuning] xq:', xq.shape)   #* b, num_H, N, C//num_H
        # print('[A-Tuning] a_k:', a_k.shape) #* b, num_H, a_N, C//num_H
        # print('[A-Tuning] a_v:', a_v.shape) #* b, num_H, a_N, C//num_H
        
        A_attention = (xq @ a_k.transpose(-2, -1)) * attn.scale
        A_attention = A_attention.softmax(dim=-1)
        A_attention = attn.attn_drop(A_attention)
        # print('[A_Tuning] Q-K Matmul:', A_attention.shape)  #* B, num_H, N, a_N
        A_attention = (A_attention @ a_v).transpose(1, 2)
        # print('[A_Tuning] A_score-V Matmul:', A_attention.shape)    #* B, N, num_H, C//num_H
        # self.prompt_msa_feat.append(A_attention.cpu())
        
        A_attention = A_attention.reshape(B, N, C)
        # print('[A_Tuning] P-MSA outputs:', A_attention.shape)   #* B, N, C
        A_attention = attn.proj(A_attention)
        A_attention = attn.proj_drop(A_attention)
        #todo ---------------------

        #! Linear Transformation Prompt way
        # prompted_gram = (A_attention.transpose(1,2) @ A_attention).mean(dim=0)
        # U,S,Vt = torch.linalg.svd(prompted_gram)
        # attention = F.linear(ori_attention,Vt)
        #!----------------------------------------------------------------------------
        #* Original (Cosine Distance)
        a_sim = 1. - torch.cosine_similarity(ori_attention, A_attention, dim=-1) #* B, a_N
        #? ScaleDown --> 1.0 / ScaleDown2 --> 0.5 // ScaleDown 3 --> 1.5
        # score = torch.min(a_sim, torch.tensor(1.0,device=a_sim.device))
        #* Original
        attention = ori_attention + (a_sim[:,:,None] * A_attention)
        #!----------------------------------------------------------------------------
        
        
        #todo Cloze Prompt with Fisher Matrix
        
        #todo a_sim = torch.cosine_similarity(ori_attention, A_attention, dim=-1) #* B, a_N
        
        #! Projection Down --> Poor Representation..
        #! cat_attention = torch.cat([ori_attention, A_attention],dim=-1)
        #! attention = F.linear(cat_attention, self.scaler)
        

        x = x + block.drop_path1(block.ls1(attention))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        # print('x',x.shape)
        return x
    
    
    def flx_tuning(self, x, block, prompt, q_idx):
        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        #*xk = torch.cat([xk, prompt[:,0]],dim=1)
        #*xv = torch.cat([xv, prompt[:,1]],dim=1)
        
        pN,_,pL,pC = prompt.shape   #* pL --> Outer scope
        
        xk = torch.cat([prompt[:,0], xk],dim=1)
        xv = torch.cat([prompt[:,1], xv],dim=1)
        #* B, 1(cls-tkn) + L(cls-prompt) + pL(prefix) + N, dim
        
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        
        B, N, C = xq.shape  #* N --> inner scope
        xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        outer_vec = xv[:, :, :pL, :]
        inner_vec = xv[:, :, pL:, :]
        # _B,_N,_C = prompt[:,1].shape
        # xv = F.linear(prompt[:,1], weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        # outer_scope = 1 + self.cls_length + pL
        #! FLX up2 --> Value No embed
        #! FLX up2 --> Key No embed
        #* Attention Layer Forward
        attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        #? Add Outer Attention score to attention
        # attention = attention.softmax(dim=-1)
        outer_score = attention[:, :, :, :pL].softmax(dim=-1)   #* B, N, pL
        outer_attention = attn.attn_drop(outer_score)
        outer_attention = (outer_attention @ outer_vec).transpose(1, 2).reshape(B, N, C)

        
        inner_score = attention[:, :, :, pL:].softmax(dim=-1)   #* B, N, N
        inner_attention = attn.attn_drop(inner_score)
        inner_attention = (inner_attention @ inner_vec).transpose(1, 2).reshape(B, N, C)
        
        
        attn_scale = 1. - torch.cosine_similarity(outer_attention, inner_attention, dim=-1)
        attention = outer_attention + attn_scale[:,:,None] * inner_attention
        
        # attention = torch.cat([outer_attention, inner_attention], dim=-1)
        # attention = (attention @ xv).transpose(1,2).reshape(B,N,C)
        attention = attn.proj(attention)
        attention = attn.proj_drop(attention)
        #*------------------------

        x = x + block.drop_path1(block.ls1(attention))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x
    
    #! FLX C-Type V2
    # def flx_tuning(self, x, block, prompt, q_idx):
    #     xq = block.norm1(x)
    #     xk = xq.clone()
    #     xv = xq.clone()

    #     # xk = torch.cat([xk, prompt[:,0]],dim=1)
    #     # xv = torch.cat([xv, prompt[:,1]],dim=1)
        
    #     attn = block.attn
    #     weight = attn.qkv.weight
    #     bias = attn.qkv.bias
        
    #     B, N, C = xq.shape
    #     xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
    #     _B, _N, _C = xk.shape
    #     xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
    #     _B,L, _C = prompt[:,0].shape
    #     pk = prompt[:,0].reshape(_B, L, attn.num_heads, _C // attn.num_heads).permute(0, 2, 1, 3)
        
    #     _B,L, _C = prompt[:,1].shape
    #     xv = F.linear(prompt[:,1], weight[2*C: ,:], bias[2*C: ]).reshape(_B, L, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
    #     #* Attention Layer Forward
    #     _intra = (xq @ xk.transpose(-2, -1)) * attn.scale
    #     _inter = (xq @ pk.transpose(-2, -1)) * attn.scale
    #     attention = _intra @ _inter
    #     attention = attention.softmax(dim=-1)
        
    #     attention = attn.attn_drop(attention)

    #     attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
    #     attention = attn.proj(attention)
    #     attention = attn.proj_drop(attention)
    #     #*------------------------

    #     x = x + block.drop_path1(block.ls1(attention))
    #     x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
    #     return x
    
    
    
    def forward(self, img, task_id=None):
        #todo a_component for CLS-Prompt and Res-Prompt
        self.cls_prompt, self.res_prompt = self._build_prompts(img)
        
        x = self.prompt_forward(img, task_id)
        
        if task_id is None:
            out = self.fc(x["features"])
        else:
            out = self.fc(x["features"], self.fc_mask)
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        # recons = self.prompt_recon(img)
        # losses = self._prompt_loss(recons)
        losses = (torch.zeros(1,device=img.device),torch.zeros(1,device=img.device))
        
        out.update(x)
        out.update({'losses':losses})
        return out
    
    def _build_prompts(self, img):
        outputs = self.extract_vector(img, img_feats=True)
        
        #* cls_feat: B, dim // img_feat: N (img tokens), dim
        cls_feat, img_feat = outputs[0], outputs[1]
        
        cls_U, cls_S, cls_Vt = torch.linalg.svd(cls_feat)
        # cls_prompt = self.cls_component @ cls_Vt.t()   #* L,dim
        # cls_prompt = self.cls_component @ cls_Vt.t()[-self.cls_length:]   #* L,dim
        cls_prompt = self.cls_component @ cls_Vt.t()[:self.cls_length]   #* L,dim
        
        #*img_U, img_S, img_Vt = torch.linalg.svd(img_feat.mean(dim=1))
        # res_prompt = self.res_component @ img_Vt.t()   #* Pos, L,dim
        # res_prompt = self.res_component @ img_Vt.t()[:self.cls_length]   #* Pos, L,dim
        res_prompt = self.res_component @ cls_Vt.t()[-self.cls_length:]   #* Pos, L,dim
        
        return cls_prompt, res_prompt
    
    def prompt_forward(self, x, task_id=None):
        #! 기존 CLS token (Freeze) 붙여서 하는 방법도 실험 해보기!
        x = self.backbone.patch_embed(x)
        
        # cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        # x = torch.cat([cls_tkns,x],dim=1)
        # x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        B,N,C = x.shape
        x = x + self.backbone.pos_embed[:,1:].expand(B, N, C)
        L, dim = self.cls_prompt.shape
        # cls_tkn = self.backbone.cls_token.expand_as()
        #* cls_tkns = self.cls_prompt.expand(B, -1, -1) + self.backbone.pos_embed[:,0].expand(B,self.cls_length,-1)
        cls_tkns = self.backbone.cls_token.expand(B, self.cls_length, dim) + self.backbone.pos_embed[:,0].expand(B, self.cls_length,-1)
        x = torch.cat([cls_tkns, x], dim=1)   #* B,N + L, C
        
        #todo -------------------------------------------------
        fmap = []
        for idx, block in enumerate(self.backbone.blocks):
            #todo Residual Prompt --> Skip Connection to provide Data dependent-Knowledge
            # if idx in self.res_pos:
            #     B,NL,C = x.shape
            #     res_idx = self.res_pos.index(idx)
            #     res_prompt = self.res_prompt[res_idx].expand(B,-1,-1)   #* res_prompt: B, L, dim
            #     cls_prompt = x[:,:self.cls_length]
            #     res_cls = cls_prompt + res_prompt 
            #     x = torch.cat([res_cls, x[:,self.cls_length:]], dim=1)
                
                # cls_prompt = x[:,:self.cls_length]
                # res_img = x[:,self.cls_length:] + res_prompt 
                # x = torch.cat([cls_prompt, res_img], dim=1)
                
            if idx in self.pre_pos:
                #? prefix_prompt : len(pos), length, dim
                prompt = self.pre_prompt[self.pre_pos.index(idx)]
                # prompt = self.res_prompt[self.pre_pos.index(idx)]
                B,N,C = x.shape
                leng, dim = prompt.shape
                prompt = prompt.expand(B, leng, dim).reshape(B, 2, int(leng/2), dim)
                # x = self.a_tuning(x, block, prompt, idx)
                x = self.flx_tuning(x, block, prompt, idx)
            else:
                x = block(x)
            fmap.append(x)
        x = self.backbone.norm(x)
        #* with cls
        # return {'features':x[:,:self.cls_length+1].mean(dim=1),'fmaps': fmap}
        #* only cls
        # return {'features':x[:,0],'fmaps': fmap}
        #* with cls out cls-prompt
        return {'features':x[:,:self.cls_length].mean(dim=1),'fmaps': fmap}
    
    def feature_map_extract(self,x, prompt, block, idx):
        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        pN,_,pL,pC = prompt.shape   #* pL --> Outer scope
        
        xk = torch.cat([prompt[:,0], xk],dim=1)
        xv = torch.cat([prompt[:,1], xv],dim=1)
        
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        
        B, N, C = xq.shape
        xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        outer_vec = xv[:, :, :pL, :]
        inner_vec = xv[:, :, pL:, :]
        
        #* Attention Layer Forward
        attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        outer_score = attention[:, :, :, :pL].softmax(dim=-1)   #* B, cls-N, pL
        outer_attention = attn.attn_drop(outer_score)
        outer_attention = (outer_attention @ outer_vec).transpose(1, 2).reshape(B, N, C)
        
        inner_score = attention[:, :, :, pL:].softmax(dim=-1)   #* B, N, img-N
        inner_attention = attn.attn_drop(inner_score)
        inner_attention = (inner_attention @ inner_vec).transpose(1, 2).reshape(B, N, C)

        attn_scale = 1. - torch.cosine_similarity(outer_attention, inner_attention, dim=-1)
        attention = outer_attention + attn_scale[:,:,None] * inner_attention
        attention = attn.proj(attention)
        attention = attn.proj_drop(attention)
        #*------------------------
        attn_fmap = attention[:,self.cls_length:]
        # attn_fmap = attention[:,:]
        x = x + block.drop_path1(block.ls1(attention))
        mlp_fmap = x[:,self.cls_length:]
        # mlp_fmap = x[:,1:]
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x, (attn_fmap.cpu(), mlp_fmap.cpu())
    
    def feature_variance(self, x, task_id):
        # self.cls_prompt, self.res_prompt = self._build_prompts(x)
        # x = self.backbone.patch_embed(x)
        # cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        # x = torch.cat([cls_tkns,x],dim=1)
        # x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        self.cls_prompt, self.res_prompt = self._build_prompts(x)
        # cls_tkn = self.backbone.cls_token.expand_as(self.cls_prompt)
        x = self.backbone.patch_embed(x)
        B,N,C = x.shape
        x = x + self.backbone.pos_embed[:,1:].expand(B, N, C)
        cls_tkns = self.cls_prompt.expand(B, -1, -1) + self.backbone.pos_embed[:,0].expand(B,self.cls_length,-1)
        # cls_tkns = cls_tkn.expand(B, -1, -1) + self.backbone.pos_embed[:,0].expand(B,self.cls_length,-1)
        x = torch.cat([cls_tkns, x],dim=1)   #* B,N + L, C
        
        fmap = []
        for idx, block in enumerate(self.backbone.blocks):
            if idx in self.res_pos:
                B,NL,C = x.shape
                res_idx = self.res_pos.index(idx)
                res_prompt = self.res_prompt[res_idx].expand(B,-1,-1)   #* res_prompt: B, L, dim
                cls_prompt = x[:,:self.cls_length]
                res_cls = cls_prompt + res_prompt 
                x = torch.cat([res_cls, x[:,self.cls_length:]], dim=1)
            elif idx in self.pre_pos:
                prompt = self.pre_prompt[self.pre_pos.index(idx)]
                B,N,C = x.shape
                leng, dim = prompt.shape
                prompt = prompt.expand(B, leng, dim).reshape(B, 2, int(leng/2), dim)
                # feature_map_extract(self,x, prompt, block, idx):
                x, fmaps = self.feature_map_extract(x, prompt, block, idx)
            else:
                x, fmaps = self.forward_block(x, block)
            fmap.append(fmaps)
        x = self.backbone.norm(x)
        
        # if task_id is not None:
        #     #* with Mask
        #     out = self.fc(x["features"], self.fc_mask)
        #     # out = self.fc(x["img_features"].mean(dim=1))
        # else:
        #     out = self.fc(x["features"])
        # out.update(x)
        return {'fmaps': fmap, 'features': x[:,:self.cls_length],'img_features':x[:,self.cls_length+1:], 'all_features':x}
        # return {'fmaps': fmap, 'features': x[:,0],'img_features':x[:,1:], 'all_features':x}
    
    def forward_block(self, x, block):
        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()
        
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        
        B, N, C = xq.shape
        xq = F.linear(xq, weight[:C,:], bias[:C]).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C:,:], bias[2*C:]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        attention = attention.softmax(dim=-1)
        attention = attn.attn_drop(attention)

        attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
        attention = attn.proj(attention)
        attention = attn.proj_drop(attention)
        
        # attn_fmap = block.ls1(attention)[:,1:]
        attn_fmap = attention[:,self.cls_length:]
        x = x + block.drop_path1(block.ls1(attention))
        
        # mlp_fmap = block.norm2(x)[:,1:]
        mlp_fmap = x[:,self.cls_length:]
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x, (attn_fmap.cpu(), mlp_fmap.cpu())