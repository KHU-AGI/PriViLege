import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance
import numpy as np
import copy
# from .helper import *
import timm
# from timm.models import vit_base_patch16_224_in21k
from models.vision_transformer import VisionTransformer
#todo PKT for domain specific knowledge learning..
#todo PKT with B-Prompt ==> Prefix Tuning 
#todo Need Something to focus on domain specific knowledge learning 
#todo finc inciteness from the Novel Category Discovery 
import open_clip as clip
class ViT_MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
        if self.args.dataset == 'cub200' or self.args.dataset == 'air':
            self.num_features = 768

        if args.scratch:
            self.encoder = timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        else:
            self.encoder = timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        
        #* Prompt
        #todo Head 토큰 없애고 Vision으로 Pool
        self.prompt_length = 2 
        self.expert_length = 2 #* Number of tuning layers
        self.prompt = nn.Parameter(torch.randn(self.prompt_length,self.num_features))   #* VL
        self.expert_prompt = nn.Parameter(torch.randn(self.expert_length, 2, self.num_features))   #* B-Prompt (WC, MP)
        nn.init.uniform_(self.prompt, -1, 1)
        nn.init.uniform_(self.expert_prompt, -1, 1)
        #*------------------------------------------------------
        self.num_tokens = 197
        self.num_heads = self.encoder.blocks[0].attn.num_heads
        
        self.comp_out = args.comp_out
        self.global_comp = nn.Conv1d(self.num_tokens + self.prompt_length, self.comp_out, kernel_size=1)
        nn.init.uniform_(self.global_comp.weight.data, -1, 1)
        
        self.local_comps = nn.ModuleList([nn.Conv1d(self.num_tokens+self.prompt_length, self.comp_out, kernel_size=1) for _ in range(self.num_heads)])
        for l_comp in self.local_comps:
            nn.init.uniform_(l_comp.weight.data, -1, 1)
        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.is_classifier = True
        
        self.seen_classes = args.base_class
    #todo =======================================================================================
    
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    def forward_metric(self, x, B_tuning=False, eval=False):
        #? Original
        x = self.prompt_encode(x, prompt_feat=True, B_tuning=B_tuning, eval=eval)
        cls_emb, prompt_emb = x 
        logit = self.fc(0.5*(cls_emb+prompt_emb['Vision']))
        
        return logit
    
    def encode(self, x):
        x = self.encoder.forward_features(x)[:,0]
        return x
    
    def prompt_encode(self, img, prompt_feat=False, B_tuning=False, eval=False):
        x = self.encoder.patch_embed(img)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt
        pos_tkn = prompting_tkn + self.encoder.pos_embed[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)#
        #!=============================================================
        #* prefix for B-Prompt (Original)
        if B_tuning:
            x = self._forward_blocks(x, self.expert_prompt, eval=eval)
        else:
            x = self.encoder.blocks(x)
        
        cls_embed = x[:,0]
        if prompt_feat:
            prompt_embed ={}
            #todo Align -> Head
            prompt_embed['Vision'] = x[:,1]
            prompt_embed['Language'] = x[:,2]
            
            return cls_embed, prompt_embed
        else:
            return cls_embed
    
    def _forward_blocks(self, x, prefix_tkn, eval=False):
        taskblock=[0,1]
        if len(taskblock) == len(self.encoder.blocks) or  0 in taskblock:
            latent_feat = x
        for block_idx, block in enumerate(self.encoder.blocks):
            if block_idx in taskblock:
                latent_feat = self._pk_tuning(block, latent_feat, prefix_tkn[taskblock.index(block_idx)], eval=eval)
            elif block_idx == 0:
                latent_feat = block(x)
            else:
                latent_feat = block(latent_feat)
        
        feat = self.encoder.norm(latent_feat)
        return feat

    def _extract_attn_mlp_feat(self, block, latent_feat):
        with torch.no_grad():
            xq = block.norm1(latent_feat)
            xk = xq.clone()
            xv = xq.clone()
            
            attn = block.attn
            weight = attn.qkv.weight
            bias = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C,:], bias[:C]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C:]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            
            
            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            head_attentions = (attention @ xv).transpose(1, 2)
            attention = head_attentions.reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            latent_feat = latent_feat + block.drop_path1(block.ls1(attention))
            mlp_feat =  block.mlp(block.norm2(latent_feat))
        return head_attentions, mlp_feat
        
    
    def _pk_tuning(self, block, latent_feat, prefix_tkn, eval=False):
        B,N,C = latent_feat.shape
        two,C = prefix_tkn.shape
        prefix_token = prefix_tkn.expand(B,two,C) #* B,2,768

        xq = block.norm1(latent_feat)
        xk = xq.clone()
        xv = xq.clone()
        
        attn = block.attn
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        if self.args.prefix:
            xk = torch.cat([prefix_token[:,0].unsqueeze(1), xk],dim=1)
            xv = torch.cat([prefix_token[:,1].unsqueeze(1), xv],dim=1)
        else:
            head_attentions, mlp_feat = self._extract_attn_mlp_feat(block, latent_feat)
            global_feat = self.global_comp(mlp_feat).squeeze(1) #* (comp_out, dim) --> (Batch, 1, dim)
            #* Head_attentions: B, N, H, H_dim
            head_attentions = head_attentions.permute(2, 0, 1, 3)
            H, B, N, H_dim = head_attentions.shape
            
            head_feats = []
            for head_attn, local_comp in zip(head_attentions, self.local_comps):
                head_feats.append(local_comp(head_attn))    #* B,1,H_dim
            local_feat = torch.cat(head_feats, dim=1).reshape(B,-1)
            
            xk = torch.cat([(prefix_token[:,0] * local_feat).unsqueeze(1), xk],dim=1)
            xv = torch.cat([(prefix_token[:,1] * global_feat).unsqueeze(1), xv],dim=1)
            
        B, N, C = xq.shape
        xq = F.linear(xq, weight[:C,:], bias[:C]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xk.shape
        xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        _B, _N, _C = xv.shape
        xv = F.linear(xv, weight[2*C: ,:], bias[2*C:]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        attention = (xq @ xk.transpose(-2, -1)) * attn.scale
        attention = attention.softmax(dim=-1)
        attention = attn.attn_drop(attention)

        attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
        attention = attn.proj(attention)
        attention = attn.proj_drop(attention)

        latent_feat = latent_feat + block.drop_path1(block.ls1(attention))
        latent_feat = latent_feat + block.drop_path2(block.ls2(block.mlp(block.norm2(latent_feat))))
        
        return latent_feat
    
    def forward(self, input, prompt_feat=False, B_tuning=False, base=False, query=False, eval=False):
        if base:
            embedding = self.prompt_encode(input, prompt_feat=True, B_tuning=True, eval=eval)
            cls_embed, prompt_embed = embedding
            logit = self.fc(0.5*(prompt_embed['Vision']+cls_embed))
            return logit, cls_embed, prompt_embed
        if query:
            q_feat = self.encode(input)
            return q_feat
        if self.mode == 'encoder':
            embedding = self.prompt_encode(input, prompt_feat=prompt_feat, B_tuning=B_tuning, eval=eval)
            if prompt_feat:
                cls_embed, prompt_embed = embedding
                return cls_embed, prompt_embed
            else:
                return embedding
        elif self.mode != 'encoder':
            input = self.forward_metric(input, B_tuning=B_tuning, eval=eval)
            return input
        
        else:
            raise ValueError('Unknown mode')

    def train_inc(self, dataloader, epochs, session, class_list, word_info, query_info):
        print("[Session: {}]".format(session))
        self.update_fc_avg(dataloader, class_list, query_info)
        
        for idx,batch in enumerate(dataloader):

            data_imgs, data_label = [_.cuda() for _ in batch]
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
            
            word_cur_embed = word_info['cur_embed'].cuda()
            word_embed = word_info['embed'].cuda()
            for epoch in range(epochs):
                self.train()
                cls_feat, prompt_feat = self.prompt_encode(data_imgs ,prompt_feat=True, B_tuning=True)
                
                logits = self.get_logits(0.5*(prompt_feat['Vision'] + cls_feat), self.fc)
                
                loss_ce = F.cross_entropy(logits, data_label)
                if self.args.SKD:
                    loss_kb = self.knowledge_boosting(prompt_feat["Language"], word_embed, query_info, data_label)
                    loss = loss_ce + loss_kb
                else:
                    loss = loss_ce
                
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item()/data_label.shape[0]*100.
                if self.args.SKD:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} loss_kb:{loss_kb.item():.4f} ACC: {acc}")
                else:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}")

    def triplet(self,cls_embed, vision_embed, query_info, train_label):
        P_head = query_info['proto'].clone().cuda()
    
        cls_logit = F.linear(cls_embed, P_head)
        cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
        
        vis_logit = F.linear(vision_embed, P_head)
        vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
        
        cls_vis = F.cross_entropy(cls_logit, torch.softmax(vis_logit, dim=1), reduction='none')   #* B
        loss_tri = -1*((cls_vis.mean() /(cls_vis.mean() + (cls_gt.mean() + vis_gt.mean())))+1e-6).log()
        
        return loss_tri

    def head_reg(self, head_feat, word_cur_feat, label):
        fc_wts = self.fc.weight
        fc_feat_sim = (1. - torch.cosine_similarity(fc_wts[label], head_feat, dim=1)).mean()
        return fc_feat_sim

    def knowledge_boosting(self, lang_embed, word_embed, query_info, label):
        P_head = query_info['proto'].clone().cuda()
        T = 2.
        lang_logit = F.linear(lang_embed, P_head)
        loss_seman = F.cross_entropy(lang_logit, label)
        
        loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[label]/T,dim=1), reduction='batchmean')
        loss = loss_kd + 0.2*loss_seman
        # return 0.5*loss
        return 0.1*loss
    
    def update_fc_avg(self,dataloader,class_list,query_info):
        self.eval()
        query_p=[]
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed=self.encode(data_imgs).detach()
            
            for class_index in class_list:
                data_index=(label==class_index).nonzero().squeeze(-1)
                embedding = cls_embed[data_index]
                proto=embedding.mean(0)
                query_p.append(proto)
                self.fc.weight.data[class_index]=proto
            query_p = torch.stack(query_p)
        query_info["proto"] = torch.cat([query_info["proto"], query_p.cpu()])
        
        self.train()

    def init_base_fc(self,query,class_list):
        self.eval()
        with torch.no_grad():
            for class_index in class_list:
                self.fc.weight.data[class_index] = query[class_index]
    
    def get_logits(self,x, fc):
        return fc(x)