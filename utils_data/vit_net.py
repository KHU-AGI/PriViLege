import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F
import timm

from utils.inc_net import BaseNet
from convs.linears import SimpleLinear

#todo Feature Map Variation 
#todo MSA and MLP --> 각각 feature map 저장하기!

class ViTNet(nn.Module):
    def __init__(self, args, pretrained):
        super(ViTNet, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True,drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        self.out_dim = 768
        self.backbone.head = nn.Identity()
        self.fc_mask = None
        self.fc = None
        
        # for p in self.backbone.parameters():
        #     p.requires_grad=False

    @property
    def feature_dim(self):
        return self.out_dim

    def backbone_forward(self, x):
        #todo Forwarding 과정 명시하고 Output 동일하게 가져갈 수 있도록
        #todo Output:
        #todo {
            #todo 'fmaps': [x_1, x_2, ..., x_n],
            #todo 'features': features
            #todo 'logits': logits
        #todo }
    
        fmap = []
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        for block in self.backbone.blocks:
            x = block(x)
            fmap.append(x[:,1:])
        
        x = self.backbone.norm(x)
    
        # return {'fmaps': fmap, 'features':x[:,0],'img_features':x[:,1:], 'all_features':x}
        return {'fmaps': fmap, 'features':x[:,0],'img_features':x[:,1:], 'all_features':x}

    def extract_vector(self, x):
        return self.backbone_forward(x)['features']
    
    def feature_map_extract(self,x, block, prompt_idx=None):
        #todo prompt: 2*self.e_length, self.feature_dim
        # if task_id is not None or g_prompt:
            # p_length,dim = prompt.shape
            # prompt = prompt.reshape(2, int(p_length/2), dim)
            # B,N,C = x.shape
            # prefix_token = prompt.expand(B, 2, int(p_length/2), dim)
        # else: #* Task Id is None --> Evaluationn
            # B,p_length,dim = prompt.shape
            # prompt = prompt.reshape(B, 2, int(p_length/2), dim)
            # prefix_token = prompt.expand(B, 2, int(p_length/2), dim)

        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        # xk = torch.cat([xk, prefix_token[:,0]],dim=1)
        # xv = torch.cat([xv, prefix_token[:,1]],dim=1)
        
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
        
        # if prompt_idx is not None:
        #     attn_fmap = attention[:,prompt_idx:]
        # else:
        #     attn_fmap = attention[:,1:]
        # attn_fmap = attention[:,prompt_idx:]
        attn_fmap = attention
        x = x + block.drop_path1(block.ls1(attention))
        
        # if prompt_idx is not None:
        #     mlp_fmap = x[:,prompt_idx:]
        # else:
        #     mlp_fmap = x[:,1:]
        # mlp_fmap = x[:,prompt_idx:]
        mlp_fmap = x
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x, (attn_fmap.cpu(), mlp_fmap.cpu())
    
    
    def forward(self, x, task_id=None):
        x = self.backbone_forward(x)
        
        #* Forward CLS feature
        # if task_id is not None:
        #     out = self.fc(x["features"],self.fc_mask)
        # else:
        #     out = self.fc(x["features"])
        #* Forward Img feature
        if task_id is not None:
            #* with Mask
            out = self.fc(x["features"],self.fc_mask)
            # out = self.fc(x["img_features"].mean(dim=1))
        else:
            out = self.fc(x["features"])
        
        #* Forward Img feature
        # if task_id is not None:
            #* With Mask
            #* out = self.fc(x["all_features"].mean(dim=1),self.fc_mask)
            # out = self.fc(x["features"])
        # else:
            # out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out
    
    def feature_variance(self, x, task_id):
        fmap = []
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        for block in self.backbone.blocks:
            x, fmaps = self.feature_map_extract(x, block)
            # x = block(x)
            fmap.append(fmaps)
        
        x = self.backbone.norm(x)
        
        # if task_id is not None:
        #     #* with Mask
        #     out = self.fc(x["features"], self.fc_mask)
        #     # out = self.fc(x["img_features"].mean(dim=1))
        # else:
        #     out = self.fc(x["features"])
        # out.update(x)
        return {'fmaps': fmap, 'features': x[:,0],'img_features':x[:,1:], 'all_features':x}

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

class L2P_Net(ViTNet):
    def __init__(self, args, pretrained):
        super(L2P_Net, self).__init__(args, pretrained)
        #todo Prompt Forwarding..
        #todo Optimizer --> Prompt + _Network
        #todo Mask to vail learned class in classifier..
        # print(args)
        self.k = args['k']
        self.prompt_pool = args['pool']
        self.prompt_length = args['length']
        self.prompt_dim = self.feature_dim
        
        self.prompt = torch.nn.Parameter(torch.randn(self.prompt_pool, self.prompt_length, self.prompt_dim),requires_grad=True,)
        nn.init.uniform_(self.prompt, -1, 1)
        self.prompt_key = torch.nn.Parameter(torch.randn(self.prompt_pool,self.prompt_dim), requires_grad=True,)
        nn.init.uniform_(self.prompt_key, -1, 1)
        
        #?------------------------------------------------------------
        # self.fisher = torch.zeros_like(self.prompt, requires_grad=False)
        
    
    def distance_matrix(self, a, b):
        a_norm = F.normalize(a,dim=1)
        b_norm = F.normalize(b,dim=1)
        cosine_dist = a_norm @ b_norm.t()
        return cosine_dist
    
    def prompt_selection(self, x, fisher=False):
        with torch.no_grad():
            query = self.extract_vector(x)
        cosine_dist = self.distance_matrix(query, self.prompt_key)
        cos_sim, selected_idx = torch.topk(cosine_dist,k=self.k)
        
        key_loss = (1. - cos_sim).mean()
        if fisher:
            return self.prompt[selected_idx], key_loss, selected_idx
        else:
            return self.prompt[selected_idx], key_loss
    
    def prompt_forward(self, x, selected_prompt):
        x = self.backbone.patch_embed(x)
        cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tkns,x],dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        #todo -------------------------------------------------
        B,K,Length,d = selected_prompt.shape
        pos_tkn = selected_prompt.reshape(B,K*Length,d) + self.backbone.pos_embed[:,0].expand(B,K*Length,-1,)
        
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)
        #! model 통과 안시켰다!!
        # fmap = []
        for block in self.backbone.blocks:
            x = block(x)
            # fmap.append(x)
            # fmap.append(x[:,K*Length+1:])
        x = self.backbone.norm(x)
        return {'features':x[:,1:K*Length+1].mean(dim=1),'img_features':x[:,K*Length+1:]}
        # return {'fmaps':fmap, 'features':x[:,1:K*Length+1].mean(dim=1),'img_features':x[:,K*Length+1:]}
    
    def forward(self, img, task_id=None, fisher=False):
        if fisher:
            prompt, key_loss, prompt_idx = self.prompt_selection(img, fisher=fisher)
        else:
            prompt, key_loss = self.prompt_selection(img, fisher=fisher)
        x = self.prompt_forward(img, prompt)
        if task_id is None:
            out = self.fc(x["features"])
        else:
            out = self.fc(x["features"], self.fc_mask)
            out.update({'key_loss':key_loss})
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        if fisher:
            return out, prompt_idx
        else:
            return out
    
    def feature_variance(self, img, task_id):
        prompt, _ = self.prompt_selection(img)
        x = self.backbone.patch_embed(img)
        cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tkns,x],dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        #todo -------------------------------------------------
        B,K,Length,d = prompt.shape
        pos_tkn = prompt.reshape(B,K*Length,d) + self.backbone.pos_embed[:,0].expand(B,K*Length,-1,)
        
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)
        #! model 통과 안시켰다!!
        fmap = []
        for block in self.backbone.blocks:
            
            x, fmaps = self.feature_map_extract(x, block, prompt_idx=K*Length+1)
            # x = block(x)
            fmap.append(fmaps)  
            # fmap.append(x[:,K*Length+1:])
        x = self.backbone.norm(x)
        return {'fmaps': fmap, 'features': x[:,1:K*Length+1].mean(dim=1),'img_features':x[:,1:], 'all_features':x}

class DP_Net(ViTNet):
    def __init__(self, args, pretrained):
        super(DP_Net,self).__init__(args, pretrained)
        self.g_pos = args['g_pos']  #* Layer 0, 1
        self.g_length = args['g_length']    #Prompt Length
        
        self.e_pos = args['e_pos']  #* Layer 2, 3, 4
        self.e_length = args['e_length']
        self.e_pool = args['e_pool']
        
        self.out_feat = False
        
        if len(self.g_pos) == 0:
            print("G-prompt is None")
            self.g_prompt = None
        else:
            self.g_prompt = torch.nn.Parameter(torch.randn(len(self.g_pos),2*self.g_length,self.feature_dim))
            torch.nn.init.uniform_(self.g_prompt, -1, 1)

        self.e_prompt = torch.nn.Parameter(torch.randn(self.e_pool,len(self.e_pos),2*self.e_length,self.feature_dim))
        torch.nn.init.uniform_(self.e_prompt, -1, 1)

        self.e_key = torch.nn.Parameter(torch.randn(self.e_pool,self.feature_dim))
        torch.nn.init.uniform_(self.e_key, -1, 1)
    
    def distance_matrix(self, a, b):
        a_norm = F.normalize(a,dim=1)
        b_norm = F.normalize(b,dim=1)
        cosine_dist = a_norm @ b_norm.t()
        return cosine_dist
    
    #* for Evaluation..
    def e_prompt_select(self, x, task_id=None):
        with torch.no_grad():
            query = self.extract_vector(x)
        if task_id is not None:
            cosine_dist = self.distance_matrix(query, self.e_key[:task_id+1])
        else:
            cosine_dist = self.distance_matrix(query, self.e_key)
        _, selected_idx = torch.topk(cosine_dist,k=1)
        
        return self.e_prompt[selected_idx].squeeze(1)
    
    
    def prefix_tuning(self,x, prompt, block, task_id,g_prompt=False):
        #todo prompt: 2*self.e_length, self.feature_dim
        if task_id is not None or g_prompt:
            p_length,dim = prompt.shape
            prompt = prompt.reshape(2, int(p_length/2), dim)
            B,N,C = x.shape
            prefix_token = prompt.expand(B, 2, int(p_length/2), dim)
        else: #* Task Id is None --> Evaluationn
            B,p_length,dim = prompt.shape
            prompt = prompt.reshape(B, 2, int(p_length/2), dim)
            prefix_token = prompt.expand(B, 2, int(p_length/2), dim)

        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        xk = torch.cat([xk, prefix_token[:,0]],dim=1)
        xv = torch.cat([xv, prefix_token[:,1]],dim=1)
        
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

        x = x + block.drop_path1(block.ls1(attention))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x
    
    def prefix_forward(self, x,e_prompt,task_id=None):
        x = self.backbone.patch_embed(x)
        cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tkns,x],dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        #todo -------------------------------------------------
        # fmap = []
        for idx,block in enumerate(self.backbone.blocks):
            if idx in self.g_pos and self.g_prompt is not None:
                x = self.prefix_tuning(x, self.g_prompt[idx], block, task_id, g_prompt=True)
            elif idx in self.e_pos:
                if task_id is None:
                    x = self.prefix_tuning(x, e_prompt[:,self.e_pos.index(idx)], block, task_id)
                else:
                    x = self.prefix_tuning(x, e_prompt[self.e_pos.index(idx)], block, task_id)
            else:
                x = block(x)
            # fmap.append(x[:,1:])
        x = self.backbone.norm(x)
        
        return {'features':x[:,0],'img_features':x[:,1:]}
        # return {'features':x[:,0],'img_features':x[:,1:],'fmaps': fmap}
    
    def forward(self, img, targets=None, task_id=None):
        if task_id is None:
            if targets is None:
                e_prompt = self.e_prompt_select(img,task_id)
            else:
                # t_id = targets//10
                e_prompt = self.e_prompt[task_id].squeeze(1)
        else:
            #* Task id  --> Select proper E-Prompt 
            e_prompt = self.e_prompt[task_id]
            with torch.no_grad():
                query = self.extract_vector(img)
            key_loss = (1. - torch.cosine_similarity(query,self.e_key[task_id].unsqueeze(0))).mean()
            #todo Key Loss
        
        x = self.prefix_forward(img,e_prompt,task_id)
        
        #* CLS-Feature
        if task_id is None:
            out = self.fc(x["features"])
        else:
            out = self.fc(x["features"], self.fc_mask)
            out.update({'key_loss':key_loss})
        #* Img_feats
        # if task_id is None:
        #     out = self.fc(x["img_features"].mean(dim=1))
        # else:
        #     out = self.fc(x["img_features"].mean(dim=1), self.fc_mask)
        #     out.update({'key_loss':key_loss})
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        
        if self.out_feat:
            return out["logits"]
        else:
            out.update(x)
            
            return out
    
    def feature_variance(self, x, task_id):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        e_prompt = self.e_prompt[task_id].squeeze(1)
        fmap = []
        for idx, block in enumerate(self.backbone.blocks):
            if idx in self.g_pos and self.g_prompt is not None:
                x, fmaps = self.feature_map_extract(x, self.g_prompt[idx], block, task_id, g_prompt=True)
            elif idx in self.e_pos:
                x, fmaps = self.feature_map_extract(x, e_prompt[self.e_pos.index(idx)], block, task_id)
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
        return {'fmaps': fmap, 'features': x[:,0],'img_features':x[:,1:], 'all_features':x}
    
    # def prefix_tuning(self,x, prompt, block, task_id,g_prompt=False):
    def feature_map_extract(self, x, prompt, block, task_id, g_prompt=False):
        #todo prompt: 2*self.e_length, self.feature_dim
        if task_id is not None or g_prompt:
            p_length,dim = prompt.shape
            prompt = prompt.reshape(2, int(p_length/2), dim)
            B,N,C = x.shape
            prefix_token = prompt.expand(B, 2, int(p_length/2), dim)
        else: #* Task Id is None --> Evaluationn
            B,p_length,dim = prompt.shape
            prompt = prompt.reshape(B, 2, int(p_length/2), dim)
            prefix_token = prompt.expand(B, 2, int(p_length/2), dim)

        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        xk = torch.cat([xk, prefix_token[:,0]],dim=1)
        xv = torch.cat([xv, prefix_token[:,1]],dim=1)
        
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
        
        attn_fmap = attention[:,1:]
        x = x + block.drop_path1(block.ls1(attention))
        
        mlp_fmap = x[:,1:]
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x, (attn_fmap.cpu(), mlp_fmap.cpu())
    
    def forward_block(self, x, block):
        #todo prompt: 2*self.e_length, self.feature_dim
        # if task_id is not None or g_prompt:
            # p_length,dim = prompt.shape
            # prompt = prompt.reshape(2, int(p_length/2), dim)
            # B,N,C = x.shape
            # prefix_token = prompt.expand(B, 2, int(p_length/2), dim)
        # else: #* Task Id is None --> Evaluationn
            # B,p_length,dim = prompt.shape
            # prompt = prompt.reshape(B, 2, int(p_length/2), dim)
            # prefix_token = prompt.expand(B, 2, int(p_length/2), dim)

        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        # xk = torch.cat([xk, prefix_token[:,0]],dim=1)
        # xv = torch.cat([xv, prefix_token[:,1]],dim=1)
        
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
        
        attn_fmap = attention[:,1:]
        x = x + block.drop_path1(block.ls1(attention))
        
        mlp_fmap = x[:,1:]
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x, (attn_fmap.cpu(), mlp_fmap.cpu())

class CODA_Net(ViTNet):
    def __init__(self, args, pretrained):
        super(CODA_Net,self).__init__(args, pretrained)
        
        self.pool_size = args["prompt_size"]
        self.p_length = args["prompt_length"]
        self.pos = args["prompt_pos"]
        self.orth_param  = args["ortho_param"]
        
        self.n_task = args['num_task']
        self.task_id = 0
        self.out_feat = False
        
        
        for pos_idx in self.pos:
            p = torch.nn.Parameter(torch.randn(self.pool_size, self.p_length, self.feature_dim), requires_grad=True)
            k = torch.nn.Parameter(torch.randn(self.pool_size, self.feature_dim), requires_grad=True)
            a = torch.nn.Parameter(torch.randn(self.pool_size, self.feature_dim), requires_grad=True)
            
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            
            setattr(self, f'e_p_{pos_idx}',p)
            setattr(self, f'e_k_{pos_idx}',k)
            setattr(self, f'e_a_{pos_idx}',a)

        # print()
            
    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):
        
        def projection(u,v):
            denominator = (u*u).sum()
            
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        
        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        # pt = int(self.pool_size / (self.n_tasks))
        # s = int(self.task_count * pt)
        # f = int((self.task_count + 1) * pt)
        
        pt = int(self.pool_size / (self.n_task))
        s = int(self.task_id * pt)
        f = int((self.task_id + 1) * pt)
        # print('pt:', pt)
        # print('s',s)
        # print('f',f)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                            break
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu)
        
    def process_task_count(self):
        print("Reinit K, A, and P")
        # for e in self.e_layers:
        for pos_idx in self.pos:
            K = getattr(self,f'e_k_{pos_idx}')
            A = getattr(self,f'e_a_{pos_idx}')
            P = getattr(self,f'e_p_{pos_idx}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{pos_idx}',p)
            setattr(self, f'e_k_{pos_idx}',k)
            setattr(self, f'e_a_{pos_idx}',a)
    
    def ortho_penalty(t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

    #todo ====================================================================
    def forward(self, img, targets=None, task_id=None):
        # print("self.task_id", self.task_id)
        # print("task_id:", task_id)
        # if task_id is None:
        #     if targets is None:
        #         e_prompt = self.e_prompt_select(img)
        #     else:
        #         t_id = targets//10
        #         e_prompt = self.e_prompt[t_id].squeeze(1)
        # else:
        #     #* Task id  --> Select proper E-Prompt 
        #     e_prompt = self.e_prompt[task_id]
        
        #     key_loss = (1. - torch.cosine_similarity(query, self.e_key[task_id].unsqueeze(0))).mean()
        #     #todo Key Loss
        
        # x = self.prefix_forward(img,e_prompt,task_id)
        query = self.extract_vector(img)
        x = self.prompt_forward(img, query, task_id)
        
        
        if task_id is None:
            out = self.fc(x["features"])
        else:
            out = self.fc(x["features"], self.fc_mask)
            # out.update({'key_loss':key_loss})
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        
        if self.out_feat:
            return out["logits"]
        else:
            out.update(x)
            
            return out
    
    #* x = self.prefix_tuning(x, Ek, Ev, block):
    def prefix_tuning(self,x, Ek, Ev, block):
        #todo prompt: 2*self.e_length, self.feature_dim
        xq = block.norm1(x)
        xk = xq.clone()
        xv = xq.clone()

        xk = torch.cat([xk, Ek],dim=1)
        xv = torch.cat([xv, Ev],dim=1)
        
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

        x = x + block.drop_path1(block.ls1(attention))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        
        return x
    
    
    def prompt_forward(self, img, query, task_id):
        x = self.backbone.patch_embed(img)
        cls_tkns = self.backbone.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat([cls_tkns,x],dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        #todo -------------------------------------------------
        
        fmap = []
        loss = torch.zeros(1,device=x.device)
        for idx,block in enumerate(self.backbone.blocks):
            if idx in self.pos:
                #todo 1. Prompt 생성 --> Query 필요
                #todo 1. 생성된 Prompt --> Prefix Tuning
                B,C = query.shape
                K = getattr(self,f'e_k_{idx}')
                A = getattr(self,f'e_a_{idx}')
                p = getattr(self,f'e_p_{idx}')
                pt = int(self.pool_size / self.n_task)
                s = int(self.task_id * pt)
                f = int((self.task_id + 1) * pt)
                
                # freeze/control past tasks
                if task_id:
                    if task_id > 0:
                        # print('s:', s)
                        # print('f:', f)
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    A = A[0:f]
                    p = p[0:f]
                
                # with attention and cosine sim
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                a_querry = torch.einsum('bd,kd->bkd', query, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                #* Create Prompt 
                P_ = torch.einsum('bk,kld->bld', aq_k, p)
                # print("P:", P_.shape)

                #todo select prompts for prefix tuning
                i = int(self.p_length/2)
                Ek = P_[:,:i,:]
                Ev = P_[:,i:,:]
                
                #todo Prefix tuning Using P_ !
                x = self.prefix_tuning(x, Ek, Ev, block)

                # ortho penalty
                if task_id and self.orth_param > 0:
                    loss = self.ortho_penalty(K) * self.orth_param
                    loss += self.ortho_penalty(A) * self.orth_param
                    loss += self.ortho_penalty(p.view(p.shape[0], -1)) * self.orth_param
                # else:
                    # loss += 0
            else:
                x = block(x)

            fmap.append(x[:,1:])
            x = self.backbone.norm(x)
        return {'features':x[:,0],'img_features':x[:,1:],'fmaps': fmap, 'orth_loss': loss}
        
        # return p_return, loss, x_block
        # return x, loss
                
                # x = self.prefix_tuning(x, self.g_prompt[idx], block, task_id, g_prompt=True)
            # fmap.append(x[:,1:])
        # x = self.backbone.norm(x)
        
        # return {'features':x[:,0],'img_features':x[:,1:],'fmaps': fmap}
    
    def ortho_penalty(self,t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
    #! at the CODA-P.py
    # def after_task(self):
    #     self.task_id +=1
