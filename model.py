from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertModel

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import modeling_outputs
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import AutoModel


    
class Mapping(nn.Module):
    # mapping from vocab distribution to embedding space
    def __init__(self, vocab_size, embedding_dim):
        super(Mapping, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mapping = nn.Linear(vocab_size, embedding_dim, bias=False)
    
    def forward(self, x):
        # x: [batch_size, vocab_size]
        # output: [batch_size, embedding_dim]
        output = self.mapping(x)
        return output

class Embedding_mapping(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, extract_layer):
        super(Embedding_mapping, self).__init__()
        feat_dim2 = 512
        feat_dim1 = 256
        #self.dowsize = nn.Linear(d_model, feat_dim)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.classifier1 = nn.Linear(feat_dim1, num_classes)
        #self.classifier2 = nn.Linear(feat_dim1, num_classes)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.extract_layer = extract_layer
        #self.classifier3 = nn.Linear(feat_dim2, num_classes)
    
    def forward(self, src):
        # average pooling src
        # pdb.set_trace()
        # src: [8, 512, 512]; output1: [8, 512, 512]
        output1 = self.transformer_encoder(src)
        last_hidden = self.pooling(output1)
        last_hidden = last_hidden[:, :, 0]
        mapped_output = self.classifier1(last_hidden)
        mapped_output = F.relu(mapped_output)
        return mapped_output
    
class TransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.classifier1 = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.01)
    
    def forward(self, src):
        output1 = self.transformer_encoder(src) # [16, 256, 512]
        mapped_output = self.classifier1(output1)
        mapped_output = torch.mean(mapped_output, dim=1)
        final_output = F.sigmoid(mapped_output) # [16, 16]
        return final_output

paraphrasing_tokenizer = T5Tokenizer.from_pretrained("t5-base")
paraphrasing_model = T5ForConditionalGeneration.from_pretrained("coderpotter/T5-for-Adversarial-Paraphrasing").to("cuda")

def generate_paraphrases(sentence, top_k=120, top_p=0.9, device="cuda", max_length=768):
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    beam_outputs = paraphrasing_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=5,
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = paraphrasing_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs and sent != "":
            final_outputs.append(sent)
    # random choose non empty paraphrase from final_outputs
    import random
    random.shuffle(final_outputs)
    try:
        return final_outputs[0]
    except:
        return sentence
