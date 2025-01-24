import argparse
import itertools
import json, ujson
import os, sys
import random
import time
from functools import partial
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import editdistance as ed
from tqdm import tqdm
import time
import torchaudio
from generation import GenerationAudioTokens, decode_save_concat
torch.set_num_threads(1)

def load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True,
        model_max_length=128000,
    )
    device_map = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer

def load_dataset(fn_jsonl):
    dataset = []
    with open(fn_jsonl, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def generate_batch_asr(dataset, batch_size=1, PROMPT_ASR="{}"):
    start = 0
    with tqdm(total=len(dataset), position=0) as pbar:
        while start < len(dataset):
            end = min(len(dataset), start + batch_size)
            pbar.update(end - start)
            batch_data = dataset[start:end]
            prompts = []
            for b in batch_data:
                prompt = PROMPT_ASR.format(json.dumps({'path': b['audio']}))
                prompts.append(prompt)
            start += batch_size
            yield prompts, batch_data

def format_batch_results(batch_results):
    result = ""
    for b in batch_results:
        result += ujson.dumps(b, ensure_ascii=False) + "\n"
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='./data/base_asr_example.jsonl')
    parser.add_argument('--result_jsonl', type=str, default='./results/asr_result.jsonl')
    parser.add_argument('--max_new_tokens', type=int, default=700)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--repetition_penalty', type=float, default=1.3)

    args = parser.parse_args()
    print(args)

    dataset = load_dataset(args.dataset)
    model, tokenizer = load_model_tokenizer(args)
    model.bind_processor(tokenizer, training=False, relative_path='/')

    audio_start_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
    audio_end_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
    PROMPT_ASR = '将语音转录为文本:' + audio_start_token + '{}' + audio_end_token

    os.makedirs(os.path.dirname(args.result_jsonl), exist_ok=True)
    fw = open(args.result_jsonl, 'w', encoding="utf8")    
    for prompts, batch_data in generate_batch_asr(dataset, PROMPT_ASR=PROMPT_ASR):
        try:
            ret = model.processor(prompts)
        except Exception as e:
            print(e)
            continue
        predicted_ids = model.generate(input_ids=ret.input_ids.cuda(),
                    attention_mask=ret.attention_mask.cuda(),
                    labels=None,
                    audios=ret.audios.cuda() if ret.audios is not None else None,
                    encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                    bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    num_return_sequences=1,
                    repetition_penalty=args.repetition_penalty,
                    )

        generated = tokenizer.batch_decode(predicted_ids[:,ret.input_ids.shape[1]:], skip_special_tokens=True)
        batch_results = []
        for i, data in enumerate(batch_data):
            data["generated"] = generated[i]
            batch_results.append(data)
            print(f'RES {data["uttid"]}: {generated[i]}')
        res = format_batch_results(batch_results)
        fw.write(res)

    fw.close()
    with open(f"{args.result_jsonl}.gen_config.json", 'w') as f:
        f.write(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    print(f"Finished. Save result to {args.result_jsonl}. Saved config to {args.result_jsonl}.gen_config.json")

