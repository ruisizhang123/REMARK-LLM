import time
import copy
import argparse
import logging
import os
import random
import torch
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed)
from transformers import T5Config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import TransformerClassifier, Mapping
from utils import log_info, get_dataset, modify_probe


# Handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model name")
    parser.add_argument("--datafile_path", type=str, default="Hello-SimpleAI/HC3-gpt", help="dataset name")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--per_device_batch_size", type=int, default=2, help="Batch size to use for training.")
    parser.add_argument("--input_max_length", type=int, default=128, help="Maximum input length to use for generation")
    parser.add_argument("--message_max_length", type=int, default=8, help="Maximum message length to use for generation")
    parser.add_argument("--target_max_length", type=int, default=128, help="Maximum target length to use for generation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--save_path", type=str, default='peft_ckpt', help="Save path")
    parser.add_argument("--train_subset", type=int, default=0, help="Train subset of dataset")
    parser.add_argument("--train_semantic", type=int, default=0, help="Train semantic loss")
    parser.add_argument("--train_semantic_loss", type=str, default='cosine', help="semantic backward loss")                    
    parser.add_argument("--wm_embed_model", type=str, default="t5", help="model to insert wm") # add wm model to t5/lora
    parser.add_argument("--verbose", type=int, default=0, help="Verbose")
    parser.add_argument("--debug", type=int, default=0, help="Debug")
    parser.add_argument("--visualize", type=int, default=0, help="Visualize")
    parser.add_argument("--wm", type=int, default=0, help="wm insertion or not")
    parser.add_argument("--target_text_type", type=str, default="original", help="which text is used for target text")
    parser.add_argument("--inference_strategy", type=str, default="distribution", help="use distribution or token id to inferece")
    parser.add_argument("--mapper_info", type=str, default="logits", help="use logits or embedding to map")
    parser.add_argument("--inference_batch", type=int, default=0, help="inference batch or per")
    parser.add_argument("--attack", type=int, default=0, help="test attack accuray or not") 
    parser.add_argument("--train_rephrase", type=int, default=0, help="train with rephrase or not")
    parser.add_argument("--train_attack", type=int, default=1, help="train with attack sample or not")
    parser.add_argument("--schedule_tmp", type=int, default=0, help="schedule tmp or not")
    parser.add_argument("--figurepint", type=int, default=0, help="figurepint the texts or not")
    parser.add_argument("--periodical", type=int, default=0, help="periodical the texts or not")
    parser.add_argument("--discriminator", type=int, default=0, help="malicious transformation")
    parser.add_argument("--augument_train", type=int, default=0, help="use synoym for augument training")
    parser.add_argument("--adaptive", type=int, default=0, help="adaptive set message")
    parser.add_argument("--message_embed_method", type=str, default="concate", help="way to embed message")
    args = parser.parse_known_args()
    return args


# Main function
def main():
    args, _ = parse_args()
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.per_device_batch_size
    seed = args.seed
    model_name_or_path = args.model_path
    data_file = args.datafile_path
    save_path = args.save_path
    target_max_length =args.target_max_length
    source_max_length = args.input_max_length
    d_model = 512
    nhead = 8
    num_layers = 3
    num_classes = args.message_max_length
    train_semantic = args.train_semantic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.debug:
        save_path = os.path.join(save_path, "debug")
    else:
        cur_time = time.strftime("%s-%Y-%m-%d-%H-%M-%S", time.localtime())
        cur_time = args.datafile_path + "_" + cur_time
        save_path = os.path.join(save_path, cur_time)
    os.makedirs(save_path, exist_ok=True)
    

    # Setup logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(save_path, "training_log.log")), logging.StreamHandler()])

    logging.info(f'Args:\n {args}')

    set_seed(seed)

    config = T5Config.from_pretrained(model_name_or_path)
    config.message_max_length = args.message_max_length
    config.wm_embed_model = args.wm_embed_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    train_dataloader, _ = get_dataset(tokenizer, args, logging, model_name_or_path, data_file, source_max_length, target_max_length, batch_size)
    

    extractor = TransformerClassifier(d_model, nhead, num_layers, num_classes)
    mapper = Mapping(config.vocab_size, d_model)
    #model, extractor, mapper, semantic_extractor = model.to(device), extractor.to(device), mapper.to(device), semantic_extractor.to(device)
    model, extractor, mapper = model.to(device), extractor.to(device), mapper.to(device)
    #optimizer = torch.optim.AdamW(list(model.parameters()) + list(extractor.parameters()) + list(mapper.parameters()) + list(semantic_extractor.parameters()), lr=lr, eps=1e-8)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(extractor.parameters()) + list(mapper.parameters()), lr=lr, eps=1e-8)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    mae_criterion = torch.nn.L1Loss()
    semantic_criterion = torch.nn.L1Loss()
    cross_entropy_criterion = torch.nn.CrossEntropyLoss()

    list_total_loss = []
    list_cos_loss = []
    list_total_extractor_loss = []
    all_accs = []

    end_step = len(train_dataloader)
    print("end_step: ", end_step)
    best_acc = 0
    # Train the model
    # do a linear schedule for the gumble softmax temperature temp to go from 1.2 to 0.5
    temp_max = 1
    temp_min = 0.1
    temp_decay = (temp_max - temp_min) / int(num_epochs/4)

    for epoch in range(num_epochs):
        total_loss = 0
        total_cos_loss = 0
        total_extractor_loss = 0
        total_loss_back = 0
        match_bits = [] 
        prev_model = copy.deepcopy(model)

        for step, batch in enumerate(tqdm(train_dataloader)):                
            message_base, message_all, input_ids_original, attention_mask, labels = batch['message_base'], batch['message_all'], batch['input_ids'], batch['attention_mask'], batch['labels']
            message_base, message_all, input_ids_original, attention_mask, labels = message_base.to(device), message_all.to(device), input_ids_original.to(device), attention_mask.to(device), labels.to(device)

            if args.schedule_tmp:
                temp = temp_max - temp_decay * epoch
                if temp < temp_min:
                    temp = temp_min
            else:
                temp = 0.3
                step_ep = 1
                adv_ep = 200
                    
            maskper = 0.5
            # randomly mask input_ids with <unk> token
            def mask_input_ids(ids, tokenizer, args):
                mask_token = tokenizer.unk_token_id
                for i in range(ids.shape[0]):
                    for j in range(ids.shape[1]):
                        if random.random() < maskper:
                            ids[i][j] = mask_token
                return ids
            
            input_ids = mask_input_ids(input_ids_original, tokenizer, args)
            optimizer.zero_grad()
            if args.wm:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, message=message_all, message_embed_method=args.message_embed_method)  # [8, 512, 32128] (batch_size, seq_len, vocab_size)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, message=None)
            output_logits = outputs.logits
            #pdb.set_trace()
            probs =  torch.nn.functional.gumbel_softmax(output_logits, tau=temp, hard=False, dim=-1)
            probs = probs*attention_mask.unsqueeze(-1).float()
            outputs_ebd = mapper(probs)

            wm_logits = extractor(outputs_ebd)#[32, 64, 128]
            extractor_loss = mae_criterion(wm_logits, message_base) 
            total_extractor_loss += extractor_loss.detach().cpu().numpy()

            # ensure semantic is not changed
            if train_semantic and args.train_semantic_loss == 'cosine':
                output_embedding = model.get_output_embeddings().weight
                inputs_ebd = output_embedding[input_ids]#[32, 64, 512]
                cos_loss = semantic_criterion(outputs_ebd, inputs_ebd)
                cos_loss = cos_loss.mean()
                total_cos_loss += cos_loss.detach().cpu().numpy()
                if args.wm:
                    total_loss_back = extractor_loss + cos_loss 
                else:
                    total_loss_back = cos_loss
                total_loss += total_loss_back.detach().cpu().numpy() 
            elif train_semantic and args.train_semantic_loss == 'cross_entrophy':
                total_cos_loss += outputs.loss.detach().cpu().numpy() 
                if args.wm and epoch >= step_ep:
                    if args.discriminator and epoch >= adv_ep:
                        # random drop token from outputs_ebd
                        adve_loss = 0
                        trans = 0
                        drop = random.random()
                        if drop < 0.33:
                            drop_probs = modify_probe(probs, "delete")
                            drop_ebd = mapper(drop_probs)
                            drop_logits = extractor(drop_ebd) 
                            adve_loss += mae_criterion(drop_logits, message_base) 
                            trans = trans+1
                        add = random.random()
                        if add < 0.34:
                            add_probs = modify_probe(probs, "add")
                            add_ebd = mapper(add_probs)
                            add_logits = extractor(add_ebd) 
                            adve_loss += mae_criterion(add_logits, message_base) 
                            trans = trans+1

                        replace = random.random()
                        if replace < 0.33:
                            replace_probs = modify_probe(probs, "replace")
                            replace_ebd = mapper(replace_probs)
                            replace_logits = extractor(replace_ebd) 
                            adve_loss += mae_criterion(replace_logits, message_base) 
                            trans = trans+1
                        if adve_loss > 0:
                            adve_loss = adve_loss/trans
                        total_loss_back = 0.7*extractor_loss + 0.3*adve_loss+outputs.loss
                    else:
                        total_loss_back = extractor_loss +outputs.loss
                elif args.wm and epoch < step_ep:
                    total_loss_back = outputs.loss 
            
                total_loss += total_loss_back.detach().cpu().numpy()
            else:
                total_loss_back = extractor_loss
                total_loss += total_loss_back.detach().cpu().numpy() 
            
            total_loss_back.backward()

            optimizer.step()
            lr_scheduler.step()

            
            # get acc of decoded message
            bit_logits = wm_logits > 0.5
            match_bit = torch.sum(bit_logits == message_base, dim=1)/(args.message_max_length)
            match_bit = torch.sum(match_bit)/batch_size
            match_bits.append(match_bit.detach().cpu().item())
            
            if step > end_step:
                break
            
            if step == 0 and args.visualize:
                # get input sentence
                sentence = tokenizer.batch_decode(input_ids_original, skip_special_tokens=True)
                log_info(logging, "*****input sentence******:\n{}".format(sentence[0]))
                
                # batch decode output_logits
                token_ids = torch.argmax(output_logits, dim=-1)
                sentence = []
                for i in range(token_ids.shape[0]):
                    decoded_string = tokenizer.decode(token_ids[i], skip_special_tokens=True)
                    sentence.append(decoded_string)
                    #pdb.set_trace()
                log_info(logging, "*****output sentence******:\n{}".format(sentence[0]))
                log_info(logging, "*****message******:\n{}".format(message_base[0]))
            
        if args.verbose:
            for name, param in model.named_parameters():
                # get diff param between prev_model and model
                diff_param = param - prev_model.state_dict()[name]
                # check nonzero param
                if len(diff_param[diff_param != 0]) > 0:
                    print("diff_param[diff_param != 0]", diff_param[diff_param != 0])
        

        train_epoch_loss = total_loss / end_step
        train_message_loss = total_extractor_loss / end_step
        total_cos_loss = total_cos_loss / end_step

        cur_acc = np.mean(match_bits)
        list_total_loss.append(train_epoch_loss)
        list_cos_loss.append(total_cos_loss)
        list_total_extractor_loss.append(train_message_loss)
        all_accs.append(cur_acc)

        if cur_acc > best_acc:
            best_acc = np.mean(match_bits)
            # save model, extractor, mapper
            #save_mode, save_extractor, save_mapper = copy.deepcopy(model), copy.deepcopy(extractor), copy.deepcopy(mapper)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            torch.save(extractor.state_dict(), os.path.join(save_path, "extractor.pt"))
            torch.save(mapper.state_dict(), os.path.join(save_path, "mapper.pt"))

        # log cur_acc, train_message_loss, total_cos_loss, train_epoch_loss
        log_info(logging, f"epoch: {epoch} num_layer: {num_layers} cur_tmp: {temp} cur_mask: {maskper} cur_acc: {cur_acc} best_acc: {best_acc} train_message_loss: {train_message_loss} total_cos_loss: {total_cos_loss} train_epoch_loss: {train_epoch_loss}")
        
        plt.clf()
        plt.plot(list_total_extractor_loss, label="message loss")
        if train_semantic:
            plt.plot(list_cos_loss, label="cos loss")
            # show y axis in log scale
            plt.yscale('log')
        #plt.plot(list_total_loss, label="encoder decoder loss")
        plt.legend()
        if train_semantic:
            plt.savefig(save_path+"/loss_message_semantic.png")
        else:
            plt.savefig(save_path+"/loss_message.png")

    # log list_total_extractor_loss, list_cos_loss, all_accs, list_total_loss
    log_info(logging, f"list_total_extractor_loss: {list_total_extractor_loss}")
    log_info(logging, f"list_cos_loss: {list_cos_loss}")
    log_info(logging, f"list_acc: {all_accs}")
    log_info(logging, f"all_accs: {all_accs}")


if __name__ == "__main__":
    main()
