import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import length_to_mask
from transformers import AutoModel, AutoTokenizer
import os
import random
import numpy as np
from mymodel.module import PatchEmbed, MoE, generate_cross_modal_mask


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_dir = "mymodel/pretrained/biobert-base-cased-v1.2"


# Tokens decorrelation loss
def calculate_ortho_loss(input_vec):
    x = input_vec - torch.mean(input_vec, axis=2).unsqueeze(2).repeat(1, 1, input_vec.shape[2])
    cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
    loss = (torch.sum(cov_matrix ** 2) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2))/(cov_matrix.shape[0]*(cov_matrix.shape[1]-1)*(cov_matrix.shape[2]-1))
    return loss


def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)


class FlexCare(nn.Module):
    def __init__(self, ehr_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=4, expert_k=2, expert_total=10, device=torch.device('cpu')):
        super(FlexCare, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.task_embedding = nn.Embedding(40, hidden_dim)

        # Process time series data
        self.ehr_projection = nn.Linear(ehr_dim, hidden_dim)
        self.ehr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ehr_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Process image data
        self.patch_projection = PatchEmbed(patch_size=16, embed_dim=hidden_dim)
        num_patches = (224 // 16) * (224 // 16)
        self.cxr_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.cxr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Process text data
        self.note_projection = AutoModel.from_pretrained(cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        self.note_fc = nn.Linear(768,hidden_dim)
        self.note_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.note_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Modality fusion tokens
        self.cross_cls_tokens = nn.Parameter(torch.zeros(3, 1, hidden_dim))

        # Multimodal Transformer
        self.encoder_layer_fusion = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim*4)
        self.transformer_fusion = nn.TransformerEncoder(self.encoder_layer_fusion, num_layers=layers)
        self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Moe
        self.moe = MoE(hidden_dim, hidden_dim, hidden_dim, expert_total, hidden_dim, noisy_gating=True, k=expert_k)

        self.mm_choose = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.mm_choose2 = nn.Linear(hidden_dim, 1, bias=False)

        self.mm_layernorm = nn.LayerNorm(hidden_dim)

        # Specific classifier
        self.dense_layer_mortality = nn.Linear(hidden_dim*2, 1)
        self.dense_layer_decomp = nn.Linear(hidden_dim*2, 1)
        self.dense_layer_ph = nn.Linear(hidden_dim*2, 25)
        self.dense_layer_los = nn.Linear(hidden_dim*2, 10)
        self.dense_layer_readm = nn.Linear(hidden_dim*2, 1)
        self.dense_layer_diag = nn.Linear(hidden_dim*2, 14)
        self.dense_layer_drg = nn.Linear(hidden_dim, 769)

    def forward(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, task_index):
        task_embed = self.task_embedding(task_index).unsqueeze(1)

        # Time series
        ehr_embed = self.ehr_projection(ehr)
        ehr_cls_tokens = self.ehr_cls_token.repeat(ehr_embed.shape[0], 1, 1)
        ehr_embed = ehr_embed + self.ehr_pos_embed[:, :ehr_embed.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_embed), dim=1)

        ehr_lengths = torch.tensor(ehr_lengths).to(self.device)

        if use_ehr.sum()!=0:
            ehr_pad_mask = length_to_mask(ehr_lengths+use_ehr)
        else:
            ehr_pad_mask = length_to_mask(ehr_lengths+use_ehr, max_len=2)

        # Image
        cxr_embed = self.patch_projection(img)
        cxr_cls_tokens = self.cxr_cls_token.repeat(cxr_embed.shape[0], 1, 1)
        cxr_embed = cxr_embed + self.cxr_pos_embed[:, :cxr_embed.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_embed), dim=1)
        cxr_pad_mask = length_to_mask(use_img, max_len=1).repeat(1, cxr_embed.shape[1])

        # Text
        with torch.no_grad():
            encoding = self.tokenizer(note, padding=True, truncation=True, max_length=512, add_special_tokens=False, return_tensors='pt')
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            # if there is no text in this batch
            if attention_mask.sum()!=0:
                outputs = self.note_projection(input_ids, attention_mask=attention_mask)
                note_embed = outputs.last_hidden_state
            else:
                note_embed = torch.zeros((len(note), 1, self.note_fc.in_features)).to(self.device)
                attention_mask = torch.zeros((len(note), 1)).int().to(self.device)

        note_embed = self.note_fc(note_embed)
        note_cls_tokens = self.note_cls_token.repeat(note_embed.shape[0], 1, 1)
        note_embed = note_embed + self.note_pos_embed[:, :note_embed.shape[1], :]
        if attention_mask.sum()!=0:
            note_embed = torch.cat((note_cls_tokens, note_embed), dim=1)
        else:
            note_embed = note_cls_tokens

        if attention_mask.sum()!=0:
            note_pad_mask = length_to_mask(attention_mask.sum(dim=1)+use_note)
        else:
            note_pad_mask = length_to_mask(attention_mask.sum(dim=1)+use_note, max_len=1)


        # Multimodal fusion
        multimodal_cls_tokens = self.mm_cls_token
        for i in range(3):
            multimodal_cls_tokens = torch.cat((multimodal_cls_tokens, self.cross_cls_tokens[i].unsqueeze(0)), dim=1)
        multimodal_cls_tokens = multimodal_cls_tokens.repeat(ehr_embed.shape[0], 1, 1)
        multimodal_embed = torch.cat((task_embed, multimodal_cls_tokens, ehr_embed, cxr_embed, note_embed), dim=1)

        cls_pad_mask = length_to_mask(4*torch.ones(use_img.shape).to(self.device), max_len=4)
        task_pad_mask = length_to_mask(torch.ones(use_img.shape).to(self.device), max_len=1)
        multimodal_pad_mask = torch.cat((task_pad_mask, cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask), dim=1)

        ehr_cls_index = 5
        cxr_cls_index = ehr_cls_index + ehr_embed.shape[1]
        note_cls_index = cxr_cls_index + cxr_embed.shape[1]
        # Mask that enables modality combination tokens to precisely target information relevant to diverse modality combination patterns
        cross_cls_mask = generate_cross_modal_mask(ehr_cls_index, cxr_cls_index, note_cls_index, multimodal_embed.shape[1]).to(self.device)

        multimodal_embed = torch.transpose(multimodal_embed, 0, 1)
        fusion_embed = self.transformer_fusion(multimodal_embed, mask=cross_cls_mask, src_key_padding_mask=multimodal_pad_mask)  #
        fusion_embed = torch.transpose(fusion_embed, 0, 1)

        task_mm_embed = fusion_embed[:, 0]
        mm_embed = torch.cat((fusion_embed[:, 1:ehr_cls_index], fusion_embed[:, ehr_cls_index].unsqueeze(1), fusion_embed[:, cxr_cls_index].unsqueeze(1), fusion_embed[:, note_cls_index].unsqueeze(1)), dim=1)
        # Mask that indicates which modality combination tokens are missing
        mm_mask = torch.ones(mm_embed.shape[0], mm_embed.shape[1]).to(self.device)
        mm_mask[:, 0] = ehr_pad_mask[:, 0] | cxr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 1] = ehr_pad_mask[:, 0] | cxr_pad_mask[:, 0]
        mm_mask[:, 2] = ehr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 3] = cxr_pad_mask[:, 0] | note_pad_mask[:, 0]
        mm_mask[:, 4] = ehr_pad_mask[:, 0]
        mm_mask[:, 5] = cxr_pad_mask[:, 0]
        mm_mask[:, 6] = note_pad_mask[:, 0]

        mm_moe = torch.zeros(mm_embed.shape[0]*mm_embed.shape[1], mm_embed.shape[2]).to(self.device)
        cat_task_mm = task_mm_embed.unsqueeze(1).repeat(1, 7, 1)

        tmp_moe, moe_loss = self.moe(cat_task_mm.reshape(-1, cat_task_mm.shape[2])[mm_mask.reshape(-1) == 0],     # ,moe_loss
                           mm_embed.reshape(-1, mm_embed.shape[2])[mm_mask.reshape(-1) == 0])
        mm_moe[mm_mask.reshape(-1) == 0] = tmp_moe
        mm_moe = mm_moe.reshape(mm_embed.shape[0], mm_embed.shape[1], mm_embed.shape[2])

        cat_task_mm = torch.cat((task_mm_embed.unsqueeze(1).repeat(1, 7, 1), mm_moe), 2)
        weight = temperature_scaled_softmax(self.mm_choose2(torch.tanh(self.mm_choose(cat_task_mm))).squeeze(2) + (-mm_mask * 1e7), temperature=0.2, dim=1)

        weighted_mm = (mm_moe * weight.unsqueeze(2).repeat(1, 1, mm_moe.shape[-1])).sum(dim=1)

        final_mm_embed = torch.cat((task_mm_embed, self.mm_layernorm(weighted_mm)),1)


        if task_index[0] == 0:
            out = self.dense_layer_mortality(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 1:
            out = self.dense_layer_decomp(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 3:
            out = self.dense_layer_los(final_mm_embed)
            scores = out
        elif task_index[0] == 4:
            out = self.dense_layer_readm(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 5:
            out = self.dense_layer_diag(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 6:
            out = self.dense_layer_drg(final_mm_embed)
            scores = out
        else:
            out = self.dense_layer_ph(final_mm_embed)
            scores = torch.sigmoid(out)

        ortho_loss = calculate_ortho_loss(mm_embed)

        if self.training is True:
            return scores, ortho_loss, moe_loss
        else:
            return scores