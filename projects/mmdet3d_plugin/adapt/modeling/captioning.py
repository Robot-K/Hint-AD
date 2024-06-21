import os
import sys
sys.path.insert(0, '/mnt/kding/Daimler/UniAD/projects/mmdet3d_plugin/adapt/')
import torch
import torch.nn as nn
from modeling.load_bert import get_bert_model
from configs.config import shared_configs
from utils.miscellaneous import str_to_bool
from timm.models.vision_transformer import Block

class ADAPT(nn.Module):
    
    def __init__(self,
                 bev_dim=256, query_dim=256,
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31):
        super().__init__()
        
        # load configs
        shared_configs.shared_video_captioning_config(cbs=True, scst=True)
        args = self.get_custom_args(shared_configs)
        if args.do_train==False or args.do_eval==True:
            args = self.update_existing_config_for_inference(args) 
        args.device = torch.device(args.device)
        
        # Get BERT and tokenizer for DCG (Driving Caption Generation) 
        bert_model, config, tokenizer = get_bert_model(args)
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        
        # 1. bev projector
        self.bev_dim = bev_dim
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels=self.bev_dim, out_channels=self.bev_dim, kernel_size=1, stride=5, bias=False),
            nn.BatchNorm3d(self.bev_dim), nn.ReLU())

        self.bev_proj = nn.Sequential(
            nn.Linear(self.bev_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))

        self.query_len = query_len
        self.query_layer = query_layer
        self.query_dim = query_dim

        # 2. visual query, blocks and projector
        self.ins_proj = nn.Sequential(
            nn.Linear(self.query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))
        
        self.sdc_proj = nn.Sequential(
            nn.Linear(self.query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))
        
        self.planning_proj = nn.Sequential(
            nn.Linear(self.query_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim))

        self.ctx_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])

        self.ctx_proj = nn.Sequential(
            nn.Linear(v_embed_dim, model_args.dim),
            nn.LayerNorm(model_args.dim))
        
        # 3. learnable query
        self.prompt_query = nn.Embedding(
            query_len-2, v_embed_dim)
        
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)
        
        self.ctx_adapter_query = nn.Embedding(
            query_len * ctx_layer, model_args.dim)
        
        # 4. learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()
        

    def get_custom_args(self, base_config):
        parser = base_config.parser
        parser.add_argument('--max_num_frames', type=int, default=32)
        parser.add_argument('--img_res', type=int, default=224)
        parser.add_argument('--patch_size', type=int, default=32)
        parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
        parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument("--vidswin_size", type=str, default='base')
        parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
        parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--loss_sparse_w', type=float, default=0)
        parser.add_argument('--loss_sensor_w', type=float, default=0)
        parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--transfer_method', type=int, default=-1,
                            help="0: load all ADAPT pre-trained weights, 1: load only pre-trained sparse mask")
        parser.add_argument('--att_mask_expansion', type=int, default=-1,
                            help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
        parser.add_argument('--resume_checkpoint', type=str, default='None')
        parser.add_argument('--test_video_fname', type=str, default='None')
        args = base_config.parse_args()
        return args
    
    def update_existing_config_for_inference(self, args):
        ''' load adapt args for evaluation and inference 
        '''
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
        try:
            json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
            f = open(json_path,'r')
            json_data = json.load(f)

            from easydict import EasyDict
            train_args = EasyDict(json_data)
        except Exception as e:
            train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

        train_args.eval_model_dir = args.eval_model_dir
        train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
        train_args.model_name_or_path = 'models/captioning/bert-base-uncased/'
        train_args.do_train = False
        train_args.do_eval = True
        train_args.do_signal_eval = True if hasattr(args, 'do_signal_eval') and args.do_signal_eval else False
        train_args.do_test = True
        train_args.val_yaml = args.val_yaml
        train_args.test_video_fname = args.test_video_fname
        train_args.signal_types = args.signal_types
        return train_args

    def forward_query(self, det_inputs): # Kding
        feats, ins_query, sdc_query, planning_query = det_inputs

        prompt_query = self.prompt_query.weight.reshape(1, self.query_len-2, -1)
        bev_prompt_query = prompt_query[:, :-1]
        ins_prompt_query = prompt_query[:, -1:]
        
        feats = feats.permute(1, 2, 0) # B 256 40000
        bev_size = int(feats.size(-1)**0.5)
        feats = feats.contiguous().view(-1, self.bev_dim, 1, bev_size, bev_size)
        feats = self.downsample(feats)
        feats = feats.view(len(feats), self.bev_dim, -1)
        feats = feats.permute(0, 2, 1)    # B bev_size/5*bev_size/5 256
        clip_feats = self.bev_proj(feats.float())
        bev_query = torch.cat([bev_prompt_query, clip_feats], dim=1)
        for block in self.bev_blocks:
            bev_query = block(bev_query)
        bev_query = bev_query[:, :self.query_len-3]

        if ins_query is not None:
            ins_query = self.ins_proj(ins_query.float()) # B nq 768
        else:
            ins_query = torch.zeros_like(ins_prompt_query).to(ins_prompt_query)

        ins_query = torch.cat([ins_prompt_query, ins_query], dim=1)
        for block in self.ins_blocks:
            ins_query= block(ins_query)
        ins_query = ins_query[:, 0].unsqueeze(1)

        if sdc_query is not None:
            sdc_query = self.sdc_proj(sdc_query.float()).unsqueeze(1)   # B 1 768
        else:
            sdc_query = torch.zeros_like(ins_prompt_query).to(ins_prompt_query)

        if planning_query is not None:
            planning_query = self.planning_proj(planning_query.float()).unsqueeze(1)   # B 1 768
        else:
            planning_query = torch.zeros_like(ins_prompt_query).to(ins_prompt_query)

        ctx_query = torch.cat([bev_query, ins_query, sdc_query, planning_query], dim=1)
        ctx_query = self.ctx_proj(ctx_query)

        return ctx_query
    
    def forward(self, cap_inputs, det_inputs):
        tokens, labels = cap_inputs
        feats, ins_query, sdc_query, planning_query = det_inputs

        ctx_query = self.forward_query(det_inputs)
        _bsz, seqlen = tokens.shape
        
        # prepare VL transformer inputs
        kwargs['img_feats'] = ctx_query

        # disable bert attention outputs to avoid some bugs
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)

        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

        # Driving Caption Generation head
        outputs = self.trans_encoder(*args, **kwargs)

        # sparse attention mask loss
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs = outputs + (loss_sparsity, )

        return outputs
    
model = ADAPT()