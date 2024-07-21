import argparse
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy
import random
import numpy as np
import copy
from model import *
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed)
from transformers import T5Config
import torch.nn.functional as F
import pdb
from datasets import Dataset, DatasetDict
from random import randrange
from torch.nn.functional import cosine_similarity
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

bertscore = load("bertscore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='t5-base', help="Model path. Supports T5/UL2 models")
    parser.add_argument('--load_path', type=str, default='sample_ul2_ckpt/debug')
    parser.add_argument('--save_path', type=str, default='../logs_text_wm/cross_wiki')
    parser.add_argument("--input_max_length", type=int, default=496, help="Maximum input length to use for generation")
    parser.add_argument("--target_max_length", type=int, default=496, help="Maximum target length to use for generation")
    parser.add_argument('--dataset', type=str, default='NicolaiSivesind/ChatGPT-Research-Abstracts')
    parser.add_argument("--extract_layer", type=int, default=-1, help="layers to extract wm")
    parser.add_argument("--message_max_length", type=int, default=16, help="Maximum message length to use for generation")
    parser.add_argument("--attack", type=int, default=0, help="attack type")
    parser.add_argument("--wm_embed_model", type=str, default="t5", help="wm embed model")
    args = parser.parse_known_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_gumbel_noise(model, logits, input_idx, temperature):
    logits_prob = F.gumbel_softmax(logits, tau=temperature)
    cur_logit = F.gumbel_softmax(logits, tau=temperature)
    #logits_prob = F.softmax(logits, dim=-1)
    #cur_logit = F.softmax(logits, dim=-1)
    return cur_logit, logits_prob # decode

def beam_search_with_gumbel(model, input_ids, logits_seq, length, beam_width, temperature=1.0):
    # Start with an empty sequence and score of 1.0
    sequences = [(torch.tensor([]).long().to(device), 0.0)]  # 0.0 here since we'll be summing log probs
    logits_seq = logits_seq[0]
    count = 0

    for logits in logits_seq:
        all_candidates = []
        log_noise_probs, log_probs = add_gumbel_noise(model, logits, input_ids[0][count], temperature)
        #pdb.set_trace()
        for seq, seq_log_probs in sequences:
            #top_ix = torch.multinomial(log_noise_probs, beam_width)
            top_log_probs, top_ix = torch.topk(log_noise_probs, beam_width)
            #top_log_probs = log_probs[top_ix] 
            #pdb.set_trace()
            for i in range(beam_width):
                next_seq = torch.cat([seq, top_ix[i].unsqueeze(0).long()], dim=-1)
                new_log_prob = seq_log_probs + top_log_probs[i]
                all_candidates.append((next_seq, new_log_prob))

        # Sort all candidates by the sum of their log probabilities and keep top beam_width sequences
        sorted_candidates = sorted(all_candidates, key=lambda x: x[1].sum(), reverse=True)
        sequences = sorted_candidates[:beam_width]
        if len(sequences[0][0]) == length:
            break
        count += 1
    sequences = [sequences[i][0] for i in range(len(sequences))]
    return sequences # Return the top sequence

def get_sample(model, input_ids, logits, attention, noise, sentence=2):
    length = attention.sum(dim=1)
    decoded_ids = []
    #print("start sampling")
    for i in range(sentence):
        model_output = beam_search_with_gumbel(model, input_ids, logits, length=length, beam_width=5, temperature=noise) 
        decoded_ids.extend(model_output)
    decoded_ids = torch.stack(decoded_ids, dim=0)
    return decoded_ids
def mask_input_ids(ids, tokenizer, args, mask_per, seed):
    random.seed(seed)
    mask_token = tokenizer.unk_token_id
    new_idx = copy.deepcopy(ids)
    non_zero_idx = torch.nonzero(ids)
    random_idx = random.sample(range(non_zero_idx.shape[0]), int(mask_per * non_zero_idx.shape[0]))
    new_idx[non_zero_idx[random_idx, 0], non_zero_idx[random_idx, 1]] = mask_token
    return new_idx


def compute_bleu4(reference, candidate):
    # Reference and candidate should be lists of tokens
    # Reference can be a list of lists if you have multiple reference translations
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)


def inference(args, val_dataloader, model, tokenizer, mapper, extractor, config, target_max_length, batch_size):
    total_acc = []
    input_match_bit_list = []
    total_bert_score = []
    wm_texts = []
    print("save to ", args.load_path)
    save_dir = args.dataset.split("/")[-1]

    if "abstract" in args.dataset:
        mask_per = 0.5
    elif "wiki" in args.dataset:
        mask_per = 0.5
    else:
        mask_per = 0.5

    blue4 = []
    for step, batch in enumerate(tqdm(val_dataloader)):
        message_base, message_all, input_ids_original, attention_mask, labels = batch['message_base'], batch['message_all'], batch['input_ids'], batch['attention_mask'], batch['labels']
        message_base, message_all, input_ids_original, attention_mask, labels = message_base.to(device), message_all.to(device), input_ids_original.to(device), attention_mask.to(device), labels.to(device)
        
        
        input_token = tokenizer.decode(input_ids_original[0], skip_special_tokens=True)
        input_dist = F.one_hot(input_ids_original, num_classes=config.vocab_size).float()
        with torch.no_grad():
            input_ebd = mapper(input_dist)
            input_logits = extractor(input_ebd)
        input_logits = input_logits > 0.5
        input_match_bit = torch.sum(input_logits == message_base, dim=1)/(args.message_max_length)
        input_match_bit_list.append(input_match_bit.item())
        
        best_acc = 0
        decode_token = None
        c = 0
        noise_list = [1, 1.5, 2, 2.5, 3]
        if "wiki" in args.dataset:
            noise_list = [1.5, 2, 2.5, 3, 5]
        while best_acc < 1:
            
            input_ids = mask_input_ids(input_ids_original, tokenizer, args, mask_per=mask_per, seed=100*int(c/5))
            #pdb.set_trace()
            with torch.no_grad():
                ids = model(input_ids=input_ids, message=message_base, message_embed_method="addition_same", labels=input_ids_original, attention_mask=attention_mask)
            logits = ids.logits
            noise_idx = c % len(noise_list)

            token_ids = get_sample(model, input_ids, logits, attention_mask, noise=noise_list[int(c/5)])

            match_bit_list = []
            for token_id in token_ids:
                #token_id = token_id[0][token_id[0] != 0]
                token_id = token_id.unsqueeze(0)
                dist = F.one_hot(token_id, num_classes=config.vocab_size).float()
                with torch.no_grad():
                    outputs_ebd = mapper(dist)
                    wm_logits = extractor(outputs_ebd)
                bit_logits = wm_logits > 0.5
                match_bit = torch.sum(bit_logits == message_base, dim=1)/(args.message_max_length)
                match_bit_list.append(match_bit.item())
            c = c+1

            if  best_acc < max(match_bit_list):
                best_acc = max(match_bit_list)
                decode_token = token_ids[match_bit_list.index(best_acc)].unsqueeze(0)
            if best_acc == 1:
                break

            decode_token_test = token_ids[match_bit_list.index(max(match_bit_list))].unsqueeze(0)
            decode_token_test = tokenizer.decode(decode_token_test[0][decode_token_test[0] != 0], skip_special_tokens=True)
        
            if c > 20:
                break
           
        total_acc.append(best_acc)
        
        #print("current best acc",best_acc, token_ids.size())
        decode_token = tokenizer.decode(decode_token[0][decode_token[0] != 0], skip_special_tokens=True)
        blue4_score = compute_bleu4(input_token.split(), decode_token.split())
        print("input:", input_token)
        print("decode:", decode_token)
        if len(total_bert_score)> 0:
            print("cur_acc", best_acc, "total_acc", sum(total_acc)/len(total_acc), 
                "blue4",  sum(blue4)/len(blue4), "Bert score: ", sum(total_bert_score)/len(total_bert_score))
        blue4.append(blue4_score)
        # get bert score
        bert_score = bertscore.compute(predictions=[decode_token], references=[input_token], lang="en")
        
        total_bert_score.append(bert_score['f1'][0])
        wm_texts.append(decode_token)

        with open(os.path.join(args.save_path, save_dir+'wm_human.txt'), 'a') as f:
            f.write(decode_token)
            f.write("\n")
            f.write(str(best_acc))
            f.write("\n")
            f.write(str(bert_score['f1'][0]))
            f.write("\n")
            
        if step > 1000:
            break

    print("Accuracy: ", sum(total_acc)/len(total_acc))
    print("Bert score: ", sum(total_bert_score)/len(total_bert_score))
    print("Blue4 score: ", sum(blue4)/len(blue4))
    print("Input match bit: ", sum(input_match_bit_list)/len(input_match_bit_list))
    return

def main():
    args, _ = parse_args()
    print("args", args)
    d_model = 512
    nhead = 8
    num_layers = 3
    dim_feedforward = 2048
    batch_size = 1
    extract_layer = args.extract_layer
    num_classes = args.message_max_length
    target_max_length = args.target_max_length
    source_max_length = args.input_max_length

    # seed random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load the dataset
    if args.dataset == 'wiki':
        dataset = load_dataset('wikitext', 'wikitext-2-v1', data_dir="./nlp_dataset")
        dataset_df = dataset["train"].to_pandas()
        dataset_df = dataset_df[dataset_df["text"].str.len() > 50]
        dataset["train"] = Dataset.from_pandas(dataset_df)
        dataset_df = dataset["validation"].to_pandas()
        dataset_df = dataset_df[dataset_df["text"].str.len() > 50]
        dataset["validation"] = Dataset.from_pandas(dataset_df)

        text_column = "text"
        label_column = "text"
    elif args.dataset ==  "NicolaiSivesind/ChatGPT-Research-Abstracts":
        dataset = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts")
        dataset_df = dataset["train"].to_pandas()
        index = dataset_df.index
        index = list(index)
        with open("./nlp_dataset/ChatGPT-Research-Abstracts/index_train.txt", "r") as f:
            index_train = f.read().split(",")
            index_train = [int(i) for i in index_train]
        index_val = list(set(index) - set(index_train))

        dataset["train_tmp"] = dataset["train"].select(index_train)
        dataset["validation"] = dataset["train"].select(index_val)
        dataset["train"] = dataset["train_tmp"]
        text_column = "generated_abstract"
        label_column = "generated_abstract"
    
    elif args.dataset == "Hello-SimpleAI/HC3-gpt": # max_length = 639
        dataset = load_dataset("Hello-SimpleAI/HC3", "all")

        dataset_df = dataset["train"].to_pandas()
        dataset_df['mask'] = dataset_df.apply(lambda x: len(x["chatgpt_answers"]) > 0, axis=1)
        # remove dataset_df with mask = False
        dataset_df = dataset_df[dataset_df['mask'] == True]
        dataset_df = dataset_df.reset_index(drop=True)
        index = dataset_df.index
        index = list(index)
        dataset = DatasetDict({"train": Dataset.from_pandas(dataset_df)})
        
        with open("./nlp_dataset/HC3-gpt/index_train.txt", "r") as f:
            index_train = f.read().split(",")
            index_train = [int(i) for i in index_train]
        index_val = list(set(index) - set(index_train))
        dataset["train_tmp"] = dataset["train"].select(index_train)
        dataset["validation"] = dataset["train"].select(index_val)
        dataset["train"] = dataset["train_tmp"]

        text_column = "chatgpt_answers"
        label_column = "chatgpt_answers"

    elif args.dataset == "NicolaiSivesind/ChatGPT-Research-Abstracts-human": # max=584
        dataset = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts")
        # filter dataset with word count > 100
        dataset_df = dataset["train"].to_pandas()
        index = dataset_df.index
        index = list(index)

        
        with open("./nlp_dataset/ChatGPT-Research-Abstracts-human/index_train.txt", "r") as f:
            index_train = f.read().split(",")
            index_train = [int(i) for i in index_train]
        index_val = list(set(index) - set(index_train))
        dataset["train_tmp"] = dataset["train"].select(index_train)
        dataset["validation"] = dataset["train"].select(index_val)
        dataset["train"] = dataset["train_tmp"]
        
        text_column = "real_abstract"
        label_column = "real_abstract"

    
    # Load the model from args.load_path
    model_name_or_path = args.model_path
    config = T5Config.from_pretrained(model_name_or_path)
    config.message_max_length = args.message_max_length
    config.wm_embed_model = "t5"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
   
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
    model.load_state_dict(torch.load(os.path.join(args.load_path, 'model.pt')))
    
    extractor = TransformerClassifier(d_model, nhead, num_layers, dim_feedforward, num_classes, extract_layer)
    extractor.load_state_dict(torch.load(os.path.join(args.load_path, 'extractor.pt')))
    mapper = Mapping(config.vocab_size, d_model)
    mapper.load_state_dict(torch.load(os.path.join(args.load_path, 'mapper.pt')))

    model, extractor, mapper = model.to(device), extractor.to(device), mapper.to(device)

    def str_convert(example):
        example[label_column] = str(example[label_column])
        message = numpy.random.randint(0, 2, size=(args.message_max_length))
        message = torch.tensor(message, dtype=torch.float)
        example['message_all'] = message.repeat(args.input_max_length, 1)
        example['message_base'] = message
        return example

    def preprocess_function(sample, padding="max_length"):
        inputs = sample[text_column]
        model_inputs = tokenizer(inputs, max_length=source_max_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=sample[label_column], max_length=target_max_length, padding=padding, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        
        attention_mask = model_inputs["attention_mask"]
        attention_mask = torch.tensor(attention_mask)
        message_mask = copy.deepcopy(attention_mask)
        message_mask_sum = message_mask.sum(1)
        eos_index = (message_mask_sum - 1).long()
        message_mask[:,0] = 0
        message_mask = message_mask.scatter(1, eos_index.unsqueeze(1), 0)
        message_all = torch.tensor(sample['message_all'])* message_mask.unsqueeze(2)
        model_inputs["message_all"] = message_all.numpy()
        
        model_inputs["message_base"] = sample['message_base']
        return model_inputs

    # embed message into the model
    dataset['validation'] = dataset['validation'].map(str_convert)

    val_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    val_dataloader = DataLoader(
        val_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=batch_size, pin_memory=True
    )

    inference(args, val_dataloader, model, tokenizer, mapper, extractor, config, target_max_length, batch_size)

if __name__ == "__main__":
    main()
