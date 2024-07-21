import pandas as pd
import torch
import copy
from transformers import T5Config
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import numpy
import numpy as np
from torch.utils.data import DataLoader
import pdb
import os

def log_info(logging, s):
    logging.info(s)


def get_dataset(tokenizer, args, logging, model_name_or_path, data_file, source_max_length, target_max_length, batch_size):
    # load dataset
    import random
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.datafile_path == "NicolaiSivesind/ChatGPT-Research-Abstracts": # max_length = 455
        dataset = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts")
        # filter dataset with word count > 100
        dataset_df = dataset["train"].to_pandas()
        index = dataset_df.index
        index = list(index)
        total_num = len(dataset_df)
        train_num = int(total_num * 0.8)
        
        # split train and validation dataset
        os.makedirs("./nlp_dataset/ChatGPT-Research-Abstracts", exist_ok=True)
        if os.path.exists("./nlp_dataset/ChatGPT-Research-Abstracts/index_train.txt"):
            with open("./nlp_dataset/ChatGPT-Research-Abstracts/index_train.txt", "r") as f:
                index_train = f.read().split(",")
                index_train = [int(i) for i in index_train]
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
        else:
            index_train = random.sample(index, train_num)
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
            # save index_train to file
            with open("./nlp_dataset/ChatGPT-Research-Abstracts/index_train.txt", "w") as f:
                f.write(",".join([str(i) for i in index_train]))

        text_column = "generated_abstract"
        label_column = "generated_abstract"

        if args.target_text_type == "human_machine":
            label_column = "real_abstract"
        elif args.target_text_type == "rephrase":
            import csv
            # read rephrase_text from csv file
            rephrase_text = []
            with open("./dataset_collect/rephrase_text.csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    rephrase_text = row
                    break
            ## add rephrase_text as a new column to dataset
            # get rephrase list list(index) from rephrase_text
            rephrase_text = [rephrase_text[i] for i in index]
            dataset_df = dataset["train"].to_pandas()
            dataset_df["rephrase_text"] = rephrase_text[:len(dataset_df)]
            dataset["train"] = Dataset.from_pandas(dataset_df)

            dataset_df = dataset["validation"].to_pandas()
            dataset_df["rephrase_text"] = rephrase_text[len(dataset["train"]):]
            dataset["validation"] = Dataset.from_pandas(dataset_df)

            label_column = "rephrase_text"
        elif args.target_text_type == "rephrase_multi":
            import csv
            # read rephrase_text from csv file
            rephrase_text1 = []
            with open("./dataset_collect/rephrase_text_full_0.csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    rephrase_text1 = row
                    break
                    
            rephrase_text2 = []
            with open("./dataset_collect/rephrase_text_full_1.csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    rephrase_text2 = row
                    break
            
            # rephrase_text is the concatenation of rephrase_text1 and rephrase_text2
            rephrase_text = rephrase_text1 + rephrase_text2
            
            column = ["rephrase1", "rephrase2", "rephrase3", "rephrase4", "rephrase5", "rephrase6", "rephrase7", "rephrase8", "rephrase9", "rephrase10"]
            dataset_df_train = dataset["train"].to_pandas()
            # dataset_df["rephrase_text"] = [rephrase_text[i:i+10] for i in range(0, 10*len(dataset_df), 10)]
            # rephrase_text = [rephrase_text[i:i+10] for i in range(0, len(rephrase_text), 10)]
            for i in range(len(column)):
                final_list = []
                for j in index_train:
                    final_list.append(rephrase_text[j*10+i])
                dataset_df_train[column[i]] = final_list
            dataset["train"] = Dataset.from_pandas(dataset_df_train)

            dataset_df_val = dataset["validation"].to_pandas()
            # dataset_df["rephrase_text"] = [rephrase_text[i:i+10] for i in range(len(dataset["train"])*10, (8000+2000)*10, 10)]
            #for i in range(len(column)):
            #    dataset_df[column[i]] = rephrase_text[j+i] for j in range(len(dataset["train"])*10, (8000+2000)*10, 10)
            
            for i in range(len(column)):
                final_list = []
                for j in index_val:
                    final_list.append(rephrase_text[j*10+i])
                dataset_df_val[column[i]] = final_list

            dataset["validation"] = Dataset.from_pandas(dataset_df_val)

            label_column = "rephrase_text"
            #pdb.set_trace()

    elif args.datafile_path == "wikitext": # max_length = 699
        dataset = load_dataset('wikitext', 'wikitext-2-v1')  
        dataset_df = dataset["train"].to_pandas()
        dataset_df = dataset_df[dataset_df["text"].str.len() > 50]
        dataset["train"] = Dataset.from_pandas(dataset_df)
        dataset_df = dataset["validation"].to_pandas()
        dataset_df = dataset_df[dataset_df["text"].str.len() > 50]
        dataset["validation"] = Dataset.from_pandas(dataset_df)

        text_column = "text"
        label_column = "text"

    elif args.datafile_path == "Hello-SimpleAI/HC3-gpt": # max_length = 639/ tokenn=650
        dataset = load_dataset("Hello-SimpleAI/HC3", "all")

        dataset_df = dataset["train"].to_pandas()
        dataset_df['mask'] = dataset_df.apply(lambda x: len(x["chatgpt_answers"]) > 0, axis=1)
        # remove dataset_df with mask = False
        dataset_df = dataset_df[dataset_df['mask'] == True]
        dataset_df = dataset_df.reset_index(drop=True)
        dataset_df["chatgpt_answers"] = dataset_df["chatgpt_answers"].apply(lambda x: x[0])
        print("dataset_df", dataset_df)
        index = dataset_df.index
        index = list(index)
        total_num = len(dataset_df)
        train_num = int(total_num * 0.8)
        dataset = dataset_df
        # convert dataset to DatasetDict
        dataset = DatasetDict({"train": Dataset.from_pandas(dataset_df)})
        
        # split train and validation dataset
        os.makedirs("./nlp_dataset/HC3-gpt", exist_ok=True)
        if os.path.exists("./nlp_dataset/HC3-gpt/index_train.txt"):
            with open("./nlp_dataset/HC3-gpt/index_train.txt", "r") as f:
                index_train = f.read().split(",")
                index_train = [int(i) for i in index_train]
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
        else:
            index_train = random.sample(index, train_num)
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
            # save index_train to file
            with open("./nlp_dataset/HC3-gpt/index_train.txt", "w") as f:
                f.write(",".join([str(i) for i in index_train]))

        text_column = "chatgpt_answers"
        label_column = "chatgpt_answers"
    elif args.datafile_path == "NicolaiSivesind/ChatGPT-Research-Abstracts-human": # max=584/ token=590
        dataset = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts")
        # filter dataset with word count > 100
        dataset_df = dataset["train"].to_pandas()
        index = dataset_df.index
        index = list(index)
        total_num = len(dataset_df)
        train_num = int(total_num * 0.8)
        
        # split train and validation dataset
        os.makedirs("./nlp_dataset/ChatGPT-Research-Abstracts-human", exist_ok=True)
        if os.path.exists("./nlp_dataset/ChatGPT-Research-Abstracts-human/index_train.txt"):
            with open("./nlp_dataset/ChatGPT-Research-Abstracts-human/index_train.txt", "r") as f:
                index_train = f.read().split(",")
                index_train = [int(i) for i in index_train]
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
        else:
            index_train = random.sample(index, train_num)
            index_val = list(set(index) - set(index_train))
            dataset["train_tmp"] = dataset["train"].select(index_train)
            dataset["validation"] = dataset["train"].select(index_val)
            dataset["train"] = dataset["train_tmp"]
            # save index_train to file
            with open("./nlp_dataset/ChatGPT-Research-Abstracts-human/index_train.txt", "w") as f:
                f.write(",".join([str(i) for i in index_train]))

        text_column = "real_abstract"
        label_column = "real_abstract"

    log_info(logging, f"Dataset length :{len(dataset['train'])}")
    

    def preprocess_function(sample, padding="max_length"):
        inputs = sample[text_column]
        model_inputs = tokenizer(inputs, max_length=source_max_length, padding=padding, truncation=True)
        if args.target_text_type == "original":
            labels = tokenizer(text_target=sample[label_column], max_length=target_max_length, padding=padding, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        elif args.target_text_type == "shift_token":
            ## shift tokens of sample[label_column]
            labels = tokenizer(text_target=sample[label_column], max_length=target_max_length, padding=padding, truncation=True)
            input_ids = labels["input_ids"]
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids[:,1:]
            input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.pad_token_id]]).repeat(input_ids.shape[0], 1)], dim=1)
            if padding == "max_length":
                input_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in input_ids]
            labels["input_ids"] = input_ids
        elif args.target_text_type == "rephrase":
            labels = tokenizer(text_target=sample[label_column], max_length=target_max_length, padding=padding, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        elif args.target_text_type == "human_machine":
            labels = tokenizer(text_target=sample[label_column], max_length=target_max_length, padding=padding, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        elif args.target_text_type == "rephrase_multi":
            column = ["rephrase1", "rephrase2", "rephrase3", "rephrase4", "rephrase5", "rephrase6", "rephrase7", "rephrase8", "rephrase9", "rephrase10"]
            total_label = torch.tensor([])
            for c in column:
                labels = tokenizer(text_target=sample[c], max_length=target_max_length, padding=padding, truncation=True)
                if padding == "max_length":
                    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
                # concate labels["input_ids"] to total_label
                total_label = torch.cat([total_label, torch.tensor(labels["input_ids"])], dim=1)
                # convert total_label to torch Long
                total_label = total_label.long()
        
        if args.target_text_type == "rephrase_multi":
            model_inputs["labels"]  = total_label # [10000, 256]
        else:
            model_inputs["labels"] = labels["input_ids"]
        
        attention_mask = model_inputs["attention_mask"]
        attention_mask = torch.tensor(attention_mask)
        message_all = torch.tensor(sample['message_all'])
        model_inputs["message_all"] = message_all
        model_inputs["message_base"] = sample['message_base']
        return model_inputs
    
    prev_mssage = numpy.random.randint(0, 2, size=(4))
    prev_mssage = torch.tensor(prev_mssage, dtype=torch.float)
    def str_convert(example):
        if args.target_text_type == "rephrase_multi":
            column = ["rephrase1", "rephrase2", "rephrase3", "rephrase4", "rephrase5", "rephrase6", "rephrase7", "rephrase8", "rephrase9", "rephrase10"]
            for c in column:
                example[c] = str(example[c])
        else:
            example[label_column] = str(example[label_column])
        if args.figurepint:
            message = numpy.random.randint(0, 2, size=(int(args.message_max_length-4)))
            message = torch.tensor(message, dtype=torch.float)
            # concate prev_message and message to message_all
            message_all = torch.cat([prev_mssage, message], dim=0)
            example['message_all'] = message_all.repeat(args.input_max_length, 1)
            example['message_base'] = message_all
        elif args.adaptive:
            if example['generated_word_count'] > 400:
                message = numpy.random.randint(0, 2, size=(16))
            elif example['generated_word_count'] <= 400 and example['generated_word_count'] >200:
                message = numpy.random.randint(0, 2, size=(12))
            message = torch.tensor(message, dtype=torch.float)

            example['message_all'] = message.repeat(args.input_max_length, 1)
            example['message_base'] = message
        else:
            message = numpy.random.randint(0, 2, size=(args.message_max_length))
            message = torch.tensor(message, dtype=torch.float)
            example['message_all'] = message.repeat(args.input_max_length, 1)
            example['message_base'] = message
        return example


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    dataset['train'] = dataset['train'].map(str_convert)
    print("current dataset length", len(dataset['train']))
    train_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset")    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=batch_size, pin_memory=True)

    dataset['validation'] = dataset['validation'].map(str_convert)
    val_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",)
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=batch_size, pin_memory=True)

    return train_dataloader, val_dataloader


def modify_probe(probs, mode):
    if mode == "delete":
        # randomly set one token in the second dimension to 0
        index = np.random.randint(0, probs.shape[1], probs.shape[0])
        mask = torch.ones_like(probs)
        mask[np.arange(probs.shape[0]), index] = 0
        probs = probs * mask
        return probs

    elif mode == "replace":
        # randomly set one token in the second dimension to 0
        index = np.random.randint(0, probs.shape[1], probs.shape[0])
        new_prob = torch.rand(probs.shape[0], probs.shape[2], device=probs.device)
        new_prob = new_prob / new_prob.sum(axis=1, keepdims=True)
        
        mask = torch.ones_like(probs)
        mask[np.arange(probs.shape[0]), index] = 0
        probs_copy = probs.clone()
        probs_copy[np.arange(probs.shape[0]), index] = new_prob 
        probs = probs * mask + probs_copy * (1 - mask)
        return probs

    elif mode == "add":
        index = np.random.randint(0, probs.shape[1], probs.shape[0])
        new_prob = torch.rand(probs.shape[0], probs.shape[2], device=probs.device)
        new_prob = new_prob / new_prob.sum(axis=1, keepdims=True)

        result = torch.zeros_like(probs)
        probs_copy = probs.clone()
        for i in range(probs.shape[0]):
            result[i] = torch.cat([probs_copy[i, :index[i]], 
                                new_prob[i].unsqueeze(0), 
                                probs_copy[i, index[i]:probs.shape[1]-1]])
        mask = torch.ones_like(probs)
        for i in range(probs.shape[0]):
            mask[i, index[i]:] = 0
        probs = probs * mask + result * (1 - mask)
        return probs