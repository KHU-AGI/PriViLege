
# import new Network name here and add in model_class args
import time

# from .Network import MYNET

from .ViT_Network import ViT_MYNET

from utils import *
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

def build_label_embedding(train_set,session,Bert_model,tokenizer,word_info, args):
    if args.dataset == "cifar100":
        classes = np.unique(train_set.classes)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "mini_imagenet":
        classes = np.unique(train_set.wnids)
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes[classes_int]))
    elif args.dataset == "cub200" or args.dataset == "air":
        classes = np.unique(np.array(train_set.labels)[train_set.targets])
        print("Number of classes:", len(classes))
        classes_int = np.unique(train_set.targets)
        print("classes_int:",classes_int)
        print('new classes for session {} : {} \n'.format(session, classes))
        
    else:
        raise KeyError
    
    words_embed = []
    with torch.no_grad():
        Bert_model.eval()
        if args.dataset in ['cifar100', 'mini_imagenet']:
            for cls in classes[classes_int]:
                if args.pret_clip:
                    encoded_input = Bert_model.tokenizer(f'a photo of {cls}')
                    output = Bert_model.text_encoder.encode_text(encoded_input.cuda())
                    # words_embed.append(bert_map(output))
                    words_embed.append(output)
                    word_info["label_text"] = np.append(word_info["label_text"], cls)
                else:
                    encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                    output = Bert_model(**encoded_input)
                    # words_embed.append(bert_map(output.pooler_output))
                    words_embed.append(output.pooler_output)
                    word_info["label_text"] = np.append(word_info["label_text"], cls)
        elif args.dataset in ['cub200', 'air']:
            for cls in classes:
                if args.pret_clip:
                    encoded_input = Bert_model.tokenizer(f'a photo of {cls}')
                    output = Bert_model.text_encoder.encode_text(encoded_input.cuda())
                    words_embed.append(output)
                    word_info["label_text"] = np.append(word_info["label_text"], f'a photo of {cls}')
                else:
                    encoded_input = tokenizer(f'a photo of {cls}', return_tensors='pt')
                    output = Bert_model(**encoded_input)
                    # words_embed.append(bert_map(output.pooler_output))
                    words_embed.append(output.pooler_output)
                    word_info["label_text"] = np.append(word_info["label_text"], f'a photo of {cls}')
        else:
            raise KeyError
        
    words_embed = torch.cat(words_embed,dim=0)
    
    if word_info["embed"] == None:
        word_info["embed"] = words_embed.cpu()
    else:
        word_info["embed"] = torch.cat([word_info["embed"].cpu(),words_embed.cpu()],dim=0)
        
    word_info["cur_embed"] = words_embed.cpu()
    word_info["cur_label"] = torch.tensor(classes_int).cpu()


def replace_base_fc(trainset, transform, model, args):
    print("[Replace Base FC - Original]")
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=4, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            if args.pret_clip:
                embedding = model([data, label], query=True)
            else:
                embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
        
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

def cross_entropy(preds, targets, reduction='none'):
    labels = torch.arange(targets.shape[0]).cuda()
    loss = F.cross_entropy(preds,labels, reduction='none')
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
def clip_loss(out, label):
    logits = out['logits']
    images_similarity = out['image_sim']
    texts_similarity = out['text_sim']
    targets = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)

    texts_loss = cross_entropy(logits, label, reduction='none')
    images_loss = cross_entropy(logits.T, label, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss.mean()

def base_train(model, trainloader, optimizer, scheduler, epoch, word_info, query_info, class_list, args, loss_curve):
    print("[Base Train]")
    base_mode = model.module.mode
    
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader, mininterval=1.0)
    model.module.mode = "encoder"
    
    word_cur_embed = word_info['cur_embed'].clone().detach().cuda()
    word_embed = word_info['embed'].clone().detach().cuda()
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        if args.pret_clip:
            logits, cls_embed, prompt_embed = model([data, train_label], word_info=word_info, base=True)
            loss_ce = clip_loss(logits, train_label)
            logits_ = logits['logit_pred']
        else:
            logits, cls_embed, prompt_embed = model(data, base=True)
            logits_ = logits[:, :args.base_class]
            loss_ce = F.cross_entropy(logits_, train_label)
        
        if args.ED:
            loss_tri = triplet(cls_embed, prompt_embed['Vision'], query_info, train_label,loss_curve)
        else:
            loss_tri = torch.zeros(1,device='cuda')
        
        if args.SKD:
            loss_kb = knowledge_boosting(prompt_embed['Language'], word_embed, query_info, train_label,loss_curve)
            # loss_kb = knowledge_boosting(prompt_embed['Language'], word_embed, word_cur_embed, train_label)
        else:
            loss_kb = torch.zeros(1,device='cuda')
        
        acc = count_acc(logits_, train_label)
        total_loss = loss_ce + args.ED_hp*loss_tri + loss_kb
        
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, loss_CE={:.4f}, loss_ED={:.4f}, loss_SKD={:.4f}, acc={:.4f}'.\
                format(epoch, lrc, total_loss.item(), loss_ce.item(), loss_tri.item(), loss_kb.item(), ta.item()))
        
        optimizer.zero_grad()
        total_loss.backward()
        
        grad={}
        grad['expert_prompt']=model.module.expert_prompt.grad.clone().detach().cpu()
        grad['prompt']=model.module.prompt.grad.clone().detach().cpu()
        for n, p in model.module.encoder.blocks[:2].named_parameters():
            if 'attn.qkv.weight' in n:
                grad[n] = torch.norm(p.clone().detach().cpu(),p=2,dim=1).mean()
        loss_curve['grad_list'].append(grad)
        
        optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    
    model.module.mode = base_mode
    return tl, ta


def triplet(cls_embed, vision_embed, query_info, train_label,loss_curve):
    P_head = query_info['proto'].clone().cuda()
    
    cls_logit = F.linear(cls_embed, P_head)
    cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
    vis_logit = F.linear(vision_embed, P_head)
    vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
    
    idx = torch.arange(vis_logit.shape[0])
    
    cls_logit[idx, train_label]=0.
    vis_logit[idx, train_label]=0.
    
    l_kl = F.kl_div(F.log_softmax(vis_logit,dim=1), F.softmax(cls_logit,dim=1), reduction='batchmean')
    l_ent = vis_gt.mean() + cls_gt.mean()
    
    loss_tri = ((l_ent/l_kl)+1).log()
    return loss_tri

def knowledge_boosting(lang_embed, word_embed, query_info, train_label, loss_curve):
    T = 2.
    idx= torch.arange(len(train_label))
    #* Original
    P_head = query_info['proto'].clone().cuda()
    
    #* =======================================================================
    lang_logit = F.linear(lang_embed, P_head)    #* Soft pred
    loss_seman = F.cross_entropy(lang_logit, train_label)
    #* KL Feature
    loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[train_label]/T,dim=1), reduction='batchmean')
    
    loss = loss_kd + 0.2*loss_seman
    return 0.1*loss


def test(model, testloader, epoch, args, session, word_info):
    #todo Test시 Prompt Selection is needed..
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    print("\t\t\t[Test Phase] Session: {}".format(session))
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            if args.pret_clip:
                out = model([data, test_label],word_info=word_info)
                logits = out['logit_pred']
            else:
                logits = model(data, B_tuning=True)
                logits = logits[:, :test_class]
            
            loss = F.cross_entropy(logits, test_label)
            
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    print('base acc given new : {:.4f}'.format(va_base_given_new))
    print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs

def build_base_proto(train_loader, model, query_info, args):
    model = model.eval()
    
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            data, label = [_.cuda() for _ in batch]
            
            model.module.mode = 'encoder'
            if args.pret_clip:
                embedding = model([data, label], query=True)
            else:
                embedding = model(data, query=True)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
            
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0) #* num_base, feat_dim
    query_info["proto"] = proto_list
    model.module.mode = args.base_mode
    model = model.train()


#* ===============================================================================================================
# import gc
# import cv2
# from torchvision.transforms import transforms

# class VITAttentionRollout:
#     def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.8, attn_score=False):
#         #*self.model = model.module.encoder.cuda()
#         self.model = model.module
#         self.model.eval()
#         self.model.mode = "encoder"
#         self.head_fusion = head_fusion
#         self.discard_ratio = discard_ratio
#         self.attn_score = attn_score
#         self.handles = []
        
#         for name, module in self.model.named_modules():
#             if attention_layer_name in name:
#                 self.handles.append(module.register_forward_hook(self.get_attention))

#         self.attentions = []

#     def get_attention(self, module, input, output):
#         print("[get_attention]output:",output.shape)
#         if output.size(-1)>197:
#             sep = output.size(-1)-197
#             self.attentions.append(output[:,:,:,1+sep:].cpu())
#         else:
#             self.attentions.append(output.cpu())
        
#     def __call__(self, input_tensor):
#         self.attentions = []
#         with torch.no_grad():
#             cls_feat, prompt_feat = self.model.prompt_encode(input_tensor ,prompt_feat=True, B_tuning=True)

#         if self.attn_score:
#             return self.attentions
#         else:
#             return rollout(self.attentions, self.discard_ratio, self.head_fusion)

# def rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"

#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)

#             result = torch.matmul(a, result)

#     mask = result[0, 0 , 1 :]
#     width = int(mask.size(-1)**0.5)
#     mask = mask.reshape(width, width).numpy()
#     mask = mask / np.max(mask)
#     return mask

# def imshow(inp, title=None):
#     """Display image for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.axis('off')
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# def denormalize(img):
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
#     img = np.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
#     return img

# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam), heatmap

# import copy
# def visualize_attn_map(model, loader, train_set, args, path, attn_score=False):
#     #todo Numpy로 모든 이미지 Attn Map 생성
#     #todo label & Task ID와 매칭하여 폴더별로 저장
#     #todo Cherry picking
#     validate_path(path)
#     if args.dataset == "cifar100":
#         classes = np.unique(train_set.classes)
#         classes_int = np.unique(train_set.targets)
#     elif args.dataset == "mini_imagenet":
#         classes = np.unique(train_set.wnids)
#         classes_int = np.unique(train_set.targets)
#     elif args.dataset == "cub200":
#         classes = np.unique(np.array(train_set.labels)[train_set.targets])
#         classes_int = np.unique(train_set.targets)
        
#     rollout_model = VITAttentionRollout(model,head_fusion='mean',discard_ratio=0.8, attn_score=attn_score)
#     with torch.no_grad():
#         for i, batch in enumerate(loader, 1):
#             #! Original
#             #! data, test_label = [_.cuda() for _ in batch]
#             data, test_label = [_.cuda() for _ in batch]
#             # data = data[:98]
#             for i in range(data.shape[0]):
#                 img = data[i]
#                 img_w, img_h = img.shape[-2], img.shape[-1]
#                 #! rollout_model = VITAttentionRollout(model,head_fusion='mean',discard_ratio=0.8)
#                 mask = rollout_model(img.unsqueeze(0))
#                 # print("Attn Score:", mask.shape)
#                 if attn_score:
#                     return mask
#                 #* img = transforms.Resize((32,32))(img).detach().cpu().numpy()
#                 img = img.detach().cpu().numpy()
#                 img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
#                 img = denormalize(img) # *255 or IMAGENET denorm 방법
#                 np_img = np.array(img)[:,:,::-1]
#                 mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#                 mask,heatmap = show_mask_on_image(np_img, mask)
#                 resize_mask = cv2.resize(mask, (512, 512), fx=0.3, fy=0.7, interpolation=cv2.INTER_LANCZOS4)

#                 # cv2.imwrite(f'/content/Viz/Mask_cifar100_sample_{i}_{class_names[ori_targets[i]]}.png',resize_mask)
#                 #* cv2_imshow(resize_mask)
#                 img = img.astype(np.uint8).copy() # np.float32 -> np.uint8
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB 채널

#                 resize_img = cv2.resize(img, (512, 512), fx=0.3, fy=0.7, interpolation=cv2.INTER_LANCZOS4)
#                 # cv2.imwrite(f'/content/Viz/Img_cifar100_sample_{i}_{class_names[ori_targets[i]]}.png',resize_img)
#                 # cv2_imshow(resize_img)
                
#                 img_cat = cv2.hconcat([resize_img, resize_mask])
#                 cv2.imwrite(f'{path}/hcat_{args.dataset}_sample_{i}_{classes[test_label[i]]}.png',img_cat)
        
#         for handle in rollout_model.handles:
#             handle.remove()

# def count_wise_acc(logits, label, cls_mat, cls_samples):
#     pred = torch.argmax(logits, dim=1)
#     for idx, gt in enumerate(label):
#         cls_samples[gt] += 1.
#         if pred[idx] == gt:
#             cls_mat[gt]+=1.
    
#     return cls_mat, cls_samples


# def class_wise_test(model, testloader, epoch, args, session, word_info):
#     test_classes = ['Nighthaw', 'Least_Aukle', 'Western_Wood_Pewe', 'Warbling_Vire', 'Common_Ter', 'Pigeon_Guillemo',
#                     'House_Wre', 'Baird_Sparro', 'Rufous_Hummingbir', 'Le_Conte_Sparro']
#     test_idx = []
#     for test_cls in test_classes:
#         test_idx.append(test_classes.index(test_cls))
#     test_idx = np.array(test_idx)
#     #todo Test시 Prompt Selection is needed..
#     if args.dataset == "cub200":
#         classes = np.unique(np.array(testloader.dataset.labels))
#     elif args.dataset == "cifar100":
#         classes = np.unique(testloader.dataset.classes)
#     else:
#         print("SOMETHING IS WEIRD!!")
#         return
    
#     cls_mat = torch.tensor([0. for _ in range(len(classes))])   #* Correct Count
#     cls_samples = torch.tensor([0. for _ in range(len(classes))])   #* Sample count
    
    
#     test_class = args.base_class + session * args.way
#     model = model.eval()
#     vl = Averager_Loss()
#     va = Averager()
#     va_base = Averager()
#     va_new = Averager()
#     va_base_given_new = Averager()
#     va_new_given_base = Averager()
#     with torch.no_grad():
#         tqdm_gen = tqdm(testloader)
#         for i, batch in enumerate(tqdm_gen, 1):
#             #! Original
#             #! data, test_label = [_.cuda() for _ in batch]
#             data, test_label = [_.cuda() for _ in batch]
            
#             if args.pret_clip:
#                 out = model([data, test_label],word_info=word_info)
#                 logits = out['logit_pred']
#             else:
#                 #! B-Tuning해야 B-Prompt들어가지?
#                 logits = model(data, B_tuning=True)
#                 logits = logits[:, :test_class]
            
#             cls_mat, cls_samples = count_wise_acc(logits, test_label, cls_mat, cls_samples)
        
#         top_val, top_idx = torch.topk(cls_mat, k=200)
#         bot_val, bot_idx = torch.topk(cls_mat, k=200, largest=False)
        
#         print('Experiment Classes:', classes)
#         accs = (cls_mat/cls_samples)*100.
#         print('Bottom-ACC:', accs)
#         validate_path(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200")
#         np.save(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200/Seed{args.seed}_Classes.npy", classes)
#         np.save(f"/data/pgh2874/FSCIL/Ours/Class_Wise_ACC/{args.out}_CUB200/Seed{args.seed}_accs.npy", accs.detach().cpu().numpy())

# from sklearn.manifold import TSNE
# import os

# def validate_path(path):
#     if os.path.exists(path):
#         pass
#     else:
#         print('create folder:', path)
#         os.makedirs(path)
    
# def visualization(train_set, test_loader, model, args):
#     classes = np.unique(np.array(train_set.labels)[train_set.targets])
    
#     model.eval()
#     ori_mode = model.module.mode
#     model.module.mode = 'encoder'
#     embedding_list = []
#     label_list = []
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             data, test_label = [_.cuda() for _ in batch]
#             cls_embed, prompt_embed = model(data, prompt_feat=True, B_tuning=True)
#             embedding = 0.5*(prompt_embed['Vision']+cls_embed).cpu()
#             embedding_list.append(embedding)
#             label_list.append(test_label.cpu())
        
#     embedding_list = torch.cat(embedding_list, dim=0)
#     label_list = torch.cat(label_list, dim=0)
#     print("embedding_list:",embedding_list.shape)
#     print("label_list:",label_list.shape)    
    
    
    
    
#     path = f'TSNE_VIZ/{args.out}_train/' #? Epoch마다 찍자.. 
#     validate_path(path)
#     perplexties = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#     for trial in range(10):
#         tsne = TSNE(n_components=2, perplexity=25, random_state=0, learning_rate=perplexties[trial], n_iter=10000, init='pca') #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
#         # tsne = TSNE(n_components=2, random_state=0) #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
        
#         # select_cls = torch.randperm(args.base_class)[:10]
#         #* 60, 40, 14, 36, 5
#         select_cls = torch.arange(args.base_class)[trial*10:(trial+1)*10]
#         print("select_classes:", select_cls)
        
#         marker=['o','^','*','s','p','P','h','+','x','D']
#         color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
#                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
#         plt.figure(figsize=(8, 8))
#         for c_idx, cls_id in enumerate(select_cls):
#             data_index = np.where(label_list == cls_id)
            
#             cls_prompt_embed = embedding_list[data_index[0]].detach().cpu()
#             print("class:",cls_id)
#             print('idx:',data_index[0].shape)
#             print("feature_embed:", cls_prompt_embed.shape)
#             emb = np.array(tsne.fit_transform(np.array(cls_prompt_embed)))
#             print("TSNE feature:", emb.shape)
#             plt.scatter(emb[:, 0], emb[:, 1], label=classes[cls_id.item()])
            
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(path+f'Seed_{args.seed}_Trial_{trial}_Base_class_10.png', dpi=600)
        

# def visualize_ED_feature(train_set, test_loader, model, args):
#     classes = np.unique(np.array(train_set.labels)[train_set.targets])
    
#     model.eval()
#     ori_mode = model.module.mode
#     model.module.mode = 'encoder'
#     cls_embedding_list=[]
#     vis_embedding_list=[]
#     embedding_list = []
#     label_list = []
#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             data, test_label = [_.cuda() for _ in batch]
#             cls_embed, prompt_embed = model(data, prompt_feat=True, B_tuning=True)
#             embedding = 0.5*(prompt_embed['Vision']+cls_embed).cpu()
#             embedding_list.append(embedding)
#             cls_embedding_list.append(cls_embed.cpu())
#             vis_embedding_list.append(prompt_embed['Vision'].cpu())
            
#             label_list.append(test_label.cpu())
        
#     embedding_list = torch.cat(embedding_list, dim=0)
#     cls_embedding_list = torch.cat(cls_embedding_list, dim=0)
#     vis_embedding_list = torch.cat(vis_embedding_list, dim=0)
#     label_list = torch.cat(label_list, dim=0)
#     print("embedding_list:",embedding_list.shape)
#     print("label_list:",label_list.shape)    
    
#     path = f'TSNE_VIZ/ED_Loss_Token_Feature/{args.out}_train/' #? Epoch마다 찍자.. 
#     validate_path(path)
#     perplexties = [15, 25, 31, 32, 33, 34, 35, 36, 37, 38]
#     for trial in range(10):
#         tsne = TSNE(n_components=2, perplexity=25, random_state=0, learning_rate=perplexties[trial], n_iter=10000, init='pca') #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
#         # tsne = TSNE(n_components=2, random_state=0) #todo Inc Session에서 사용할 경우 Perplexity 수정 필요
        
#         # select_cls = torch.randperm(args.base_class)[:10]
#         #* 60, 40, 14, 36, 5
#         select_cls = torch.arange(args.base_class)[trial*10:(trial+1)*10]
#         print("select_classes:", select_cls)
        
#         marker=['o','^','*','s','p','P','h','+','x','D']
#         color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
#                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for c_idx, cls_id in enumerate(select_cls):
#             data_index = np.where(label_list == cls_id)
#             #*cls_prompt_embed = embedding_list[data_index[0]].detach().cpu()
#             cls_emb = cls_embedding_list[data_index[0]].detach().cpu()
#             vis_emb = vis_embedding_list[data_index[0]].detach().cpu()
#             print("class:",cls_id)
#             print('idx:',data_index[0].shape)
#             #*emb = np.array(tsne.fit_transform(np.array(cls_prompt_embed)))
#             cls_emb = np.array(tsne.fit_transform(np.array(cls_emb)))
#             vis_emb = np.array(tsne.fit_transform(np.array(vis_emb)))
            
            
#             if (c_idx+1)==len(select_cls):
#                 plt.scatter(cls_emb[:, 0], cls_emb[:, 1], marker='o', alpha=0.3, color=color[c_idx], label=classes[cls_id.item()])
#                 plt.scatter(vis_emb[:, 0], vis_emb[:, 1], marker='^', alpha=0.3, color=color[c_idx])
#             else:
#                 plt.scatter(cls_emb[:, 0], cls_emb[:, 1], alpha=0.3, marker='o', color=color[c_idx], label=classes[cls_id.item()])
#                 plt.scatter(vis_emb[:, 0], vis_emb[:, 1], alpha=0.3, marker='^', color=color[c_idx])
            
#         leg1 = plt.legend(loc='lower left', bbox_to_anchor=(1.01,0.2), fontsize=10)
#         # leg1 = plt.legend(bbox_to_anchor=(1.03,0), fontsize=10)
#         ax.add_artist(leg1)
#         h = [plt.plot([],[], color="gray", marker="o", ls="", label='[CLS] Token')[0], plt.plot([],[], color="gray", marker="^", ls="", label='Vis Token')[0]]
#         leg2 = plt.legend(handles=h, loc="lower left", bbox_to_anchor=(1.01,0.8), fontsize=10)
#         plt.tight_layout()
#         plt.savefig(path+f'Vis_CLS_Seed_{args.seed}_Trial_{trial}_Base_class_10.png', dpi=600, bbox_inches="tight")                
#         #todo ========================
        
# import csv
# def check_lamda(model, inputs, attn_score=False):
#     #todo Numpy로 모든 이미지 Attn Map 생성
#     #todo label & Task ID와 매칭하여 폴더별로 저장
#     #todo Cherry picking
#     rollout_model = VITAttentionRollout(model, head_fusion='mean',discard_ratio=0.8, attn_score=attn_score)
#     with torch.no_grad():
#         attn_score = rollout_model(inputs)
#         attn_score = torch.stack(attn_score)
    
#     print("Attn Score:", attn_score.shape)  #* Tuning Layer, B, Head, Q-tkn (1+2+196), Key+ B-Prompt (1[prefix]+1[cls]+2[vl]+196)
#     #todo lamda / image 각각 Scalar로 계산해서 저장
#     #todo Query: CLS, V-L, Image 
#     #todo Key: Basic, CLS, V-L, Image 
    
#     #todo 1. With VL
#     lamda_h_q = attn_score[:,:,:,3:,:4] #* Tuning Layer, B, Head, input-query, 4 (prompt)
#     lamda_h_q =lamda_h_q.mean(dim=2)     #* Tuning Layer, B, input_query,4
#     lamda_h_q =lamda_h_q.mean(dim=1)    #* Tuning Layer, input-query, 4
#     lamda_h_q =lamda_h_q.mean(dim=1)    #* Tuning Layer, 4
    
#     lamda_p_vl = attn_score[:,:,:,1:3,:4] #* Tuning Layer, B, Head, VL-query, 4 (prompt)
#     lamda_p_vl =lamda_p_vl.mean(dim=2)    #* Tuning Layer, B, VL-query,4
#     lamda_p_vl =lamda_p_vl.mean(dim=1)   #* Tuning Layer, input-query, 4
#     lamda_p_vl =lamda_p_vl.mean(dim=1)    #* Tuning Layer, 4
    
#     if not os.path.exists("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv"):
#         print("Create CSV File..")
#         f1 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv",'w',newline='')
#         wr1 = csv.writer(f1)
#         wr1.writerow(['PKT','Layer 1-prefix', 'Layer 1-cls', 'Layer 1-vision', 'Layer 1-Language',  'Layer 2-prefix', 'Layer 2-cls', 'Layer 2-vision', 'Layer 2-Language'])
#         wr1.writerow(['   ',lamda_h_q[0,0].item(),lamda_h_q[0,1].item(),lamda_h_q[0,2].item(),lamda_h_q[0,3].item(), lamda_h_q[1,0].item(),lamda_h_q[1,1].item(),lamda_h_q[1,2].item(),lamda_h_q[1,3].item()])
        
#         f2 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_p_vl.csv",'w',newline='')
#         wr2 = csv.writer(f2)
#         wr2.writerow(['PKT','Layer 1-prefix', 'Layer 1-cls', 'Layer 1-vision', 'Layer 1-Language',  'Layer 2-prefix', 'Layer 2-cls', 'Layer 2-vision', 'Layer 2-Language'])
#         wr2.writerow(['   ',lamda_p_vl[0,0].item(),lamda_p_vl[0,1].item(),lamda_p_vl[0,2].item(),lamda_p_vl[0,3].item(), lamda_p_vl[1,0].item(),lamda_p_vl[1,1].item(),lamda_p_vl[1,2].item(),lamda_p_vl[1,3].item()])
        
#     else:
#         print("Continue to write CSV File..")
#         f1 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_h_q.csv",'a', newline='')
#         wr1 = csv.writer(f1)
#         wr1.writerow(['   ',lamda_h_q[0,0].item(),lamda_h_q[0,1].item(),lamda_h_q[0,2].item(),lamda_h_q[0,3].item(), lamda_h_q[1,0].item(),lamda_h_q[1,1].item(),lamda_h_q[1,2].item(),lamda_h_q[1,3].item()])
        
#         f2 = open("/data/pgh2874/FSCIL/Ours/PKT-Lamda_csv/lamda_p_vl.csv",'a', newline='')
#         wr2 = csv.writer(f2)
#         wr2.writerow(['   ',lamda_p_vl[0,0].item(),lamda_p_vl[0,1].item(),lamda_p_vl[0,2].item(),lamda_p_vl[0,3].item(), lamda_p_vl[1,0].item(),lamda_p_vl[1,1].item(),lamda_p_vl[1,2].item(),lamda_p_vl[1,3].item()])
        
#     f1.close()
#     f2.close()
#     for handle in rollout_model.handles:
#         handle.remove()
#     del rollout_model
            