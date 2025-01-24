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

from constants import *
def load_vocoder(args):
    sys.path.append(os.path.join(args.vocoder))
    from cosy24k_vocoder import Cosy24kVocoder
    vocoder = Cosy24kVocoder.from_pretrained(os.path.join(args.vocoder, "hift.pt"))
    vocoder = vocoder.cuda()
    return vocoder

def split_string_with_punctuation_merged(s):
    # 正则表达式匹配任意标点符号
    pattern = r'([:,;!?、，。；：！？])'
    
    # 查找所有标点符号的位置
    punctuation_positions = [(m.start(0), m.group(0)) for m in re.finditer(pattern, s)]
    
    # 根据标点符号的位置分割字符串，并合并标点到前一个子字符串
    substrings = []
    last_index = 0
    for pos, punct in punctuation_positions:
        # 添加标点前的子字符串和标点本身
        substrings.append(s[last_index:pos] + punct)
        last_index = pos + len(punct)
    # 添加最后一个标点之后的子字符串（如果有的话）
    if last_index < len(s):
        substrings.append(s[last_index:])
    
    return substrings

def load_dataset(fn_jsonl):
    dataset = []
    with open(fn_jsonl, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def generate_batch_tts(dataset, batch_size=1, PROMPT_TTS='{}'):
    start = 0
    assert batch_size == 1, "batch_size should be 1"
    with tqdm(total=len(dataset), position=0) as pbar:
        while start < len(dataset):
            end = min(len(dataset), start + batch_size)
            pbar.update(end - start)
            batch_data = dataset[start:end]
            prompts = []
            cur_prompt = ""
            for b in batch_data:
                for text in split_string_with_punctuation_merged(b['trans']):
                    cur_prompt+=text
                    if len(cur_prompt) > 10:
                        prompts.append(PROMPT_TTS.format(cur_prompt))
                        cur_prompt = ""
                if cur_prompt:
                    prompts.append(PROMPT_TTS.format(cur_prompt))
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
    parser.add_argument('--vocoder', type=str, default='../third_party/cosy24k_vocoder')
    parser.add_argument('--dataset', type=str, default='./data/base_tts_example.jsonl')
    parser.add_argument('--result_jsonl', type=str, default='./results/tts_result.jsonl')
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
    PROMPT_TTS='{}'+tokenizer.convert_ids_to_tokens(model.config.audio_config.audiogen_start_token_id)
    vocoder = load_vocoder(args)
    wav_path = os.path.dirname(args.result_jsonl) + '/waves/'
    os.makedirs(wav_path, exist_ok=True)
    fw = open(args.result_jsonl, 'w', encoding="utf8")
   
    
    for prompts, batch_data in generate_batch_tts(dataset, PROMPT_TTS=PROMPT_TTS):
        audio_response = []
        gret = None
        for prompt in prompts:
            try:
                ret = model.processor([prompt])
            except Exception as e:
                print(e)
                continue
            print(f"Prompt: {prompt}")
            print(ret.input_ids)
            start = time.time()
            if gret is not None:
                gret.sequences[:,-1] = model.config.audio_config.audiogen_end_token_id
                gret.sequences = torch.concat([gret.sequences, ret.input_ids.cuda()], dim=1)
            gret = GenerationAudioTokens.generate(model, input_ids=ret.input_ids.cuda() if gret is None else gret.sequences.cuda(),
                        labels=None,
                        audios=ret.audios.cuda() if ret.audios is not None else None,
                        encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                        bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                        past_key_values= gret.past_key_values if gret is not None else None,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=args.do_sample, 
                        top_k=args.top_k, 
                        top_p=args.top_p, 
                        temperature=args.temperature,
                        num_return_sequences=1,
                        repetition_penalty=args.repetition_penalty,
                        return_dict_in_generate=True,
                        )
            audio_response.append(gret.audios_sequences)
        print(f"Gen Batch Time: {time.time() - start}")
        start = time.time()

        outputs = [os.path.abspath(wav_path)+'/' + data['uttid'] + '.wav' for data in batch_data]
        decode_save_concat(audio_response, vocoder, model, outputs[0], sampling_rate, wave_concat_overlap)
        print("Decode and Save Time: ", time.time() - start)

        batch_results = []
        for i, data in enumerate(batch_data):
            data["audio"] = outputs[i]
            batch_results.append(data)
            print("RES {}: {}".format(data["uttid"], outputs[i]))
        res = format_batch_results(batch_results)
        fw.write(res)
        fw.flush()

    fw.close()
    with open(f"{args.result_jsonl}.gen_config.json", 'w') as f:
        f.write(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    print(f"Finished. Save result to {args.result_jsonl}. Saved config to {args.result_jsonl}.gen_config.json")

