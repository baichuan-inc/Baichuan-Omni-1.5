import sys
import os
import numpy as np
import torch
import torchaudio
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation import decode_wave_vocoder, GenerationAudioTokens
import time
import re
import json, ujson
from constants import *

sys.path.append(os.path.join(COSY_VOCODER))
from cosy24k_vocoder import Cosy24kVocoder
vocoder = Cosy24kVocoder.from_pretrained(os.path.join(COSY_VOCODER, "hift.pt"))
vocoder = vocoder.cuda()

def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path="/")
    return model, tokenizer
model, tokenizer = init_model()

audio_start_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
audio_end_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
audiogen_start_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audiogen_start_token_id)
audiogen_end_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audiogen_end_token_id)
special_token_partten = re.compile('<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>') 

def wave_concat(wave_list, start, overlap=400):
    new_wave_list = []
    cur = start
    for wave in wave_list[start:]:
        if (
            cur - 1 >= 0
            and wave_list[cur - 1].shape[1] > overlap
            and wave.shape[1] > overlap
        ):
            new_wave_list.append(
                (
                    wave_list[cur - 1][:, -overlap:]
                    * torch.linspace(
                        1.0, 0.0, overlap, device=wave_list[cur - 1].device
                    )[None, :]
                    + wave[:, :overlap]
                    * torch.linspace(
                        0.0, 1.0, overlap, device=wave_list[cur - 1].device
                    )[None, :]
                )
            )
        new_wave_list.append(wave)
        cur += 1
    return torch.cat(new_wave_list, dim=1)

def save_local(wave, local_path):
    torchaudio.save(local_path, torch.cat(wave, dim=0).cpu(), sampling_rate)
    return audiogen_start_token + ujson.dumps({'path': local_path}, ensure_ascii=False) + audiogen_end_token

def generate_text_step(pret, plen, kv_cache_flag, audiogen_flag=True):
    if not kv_cache_flag:
        textret = model.generate(
                pret.input_ids.cuda(),
                attention_mask=pret.attention_mask.cuda(), 
                audios = pret.audios.cuda() if pret.audios is not None else None,
                encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
                bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
                tokenizer=tokenizer,
                max_new_tokens=50 if audiogen_flag else 1024,
                stop_strings=[audiogen_start_token, '<|endoftext|>'] if audiogen_flag else ['<|endoftext|>'],
                do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
            )
    else:
        # print("before text generation\n{}".format(tokenizer.decode(pret.sequences[0, :])))
        textret = model.generate(
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                tokenizer=tokenizer,
                past_key_values=(pret.past_key_values),
                stop_strings=[audiogen_start_token],
                max_new_tokens=50, do_sample=True, temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.05, return_dict_in_generate=True,
            )
    newtext = tokenizer.decode(textret.sequences[0, plen:])
    return textret, newtext

def generate_audio_step(pret):
    audioret = GenerationAudioTokens.generate(
                model,
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                past_key_values=(pret.past_key_values if pret.past_key_values is not None else None),
                max_new_tokens=500,
                do_sample=True, temperature=0.5, top_k=5, top_p=0.85, repetition_penalty=1.3, return_dict_in_generate=True,
    )
    wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), vocoder, model)
    return audioret, wave_segment

def generate_response(content, audiogen_flag=False):
    pret = model.processor([content])
    plen = pret.input_ids.shape[1]
    ret, text_segment = generate_text_step(pret, plen, False, audiogen_flag)
    wave_list = []
    full_text = re.sub(special_token_partten, '', text_segment)
    show_text = re.sub(special_token_partten, '', text_segment)
    if audiogen_flag:
        yield show_text, full_text, (sampling_rate, np.zeros(sampling_rate * 2, dtype=np.int16),)

        start = 0
        for i in range(100):
            m = ret.sequences[0, -1].item()
            if m == tokenizer.eos_token_id:
                if ret.sequences.shape[1] - plen > 1:
                    ret.sequences[0, -1] = (model.config.audio_config.audiogen_start_token_id)
                    ret, wave_segment = generate_audio_step(ret)
                    wave_list.extend(wave_segment)
                    full_text += save_local(wave_segment, os.path.join(g_cache_dir, f'assistant_turn{g_turn_i}_round{i}.wav'))
                    show_text += '<audio>'
                break

            ret.sequences[0, -1] = model.config.audio_config.audiogen_start_token_id
            ret, wave_segment = generate_audio_step(ret)
            wave_list.extend(wave_segment)
            full_text += save_local(wave_segment, os.path.join(g_cache_dir, f'assistant_turn{g_turn_i}_round{i}.wav'))
            show_text += '<audio>'

            if len(wave_list) > max(1, start):
                wave = wave_concat(wave_list, start, overlap=wave_concat_overlap)
                start = len(wave_list)
                yield show_text, full_text, (sampling_rate, (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16),)

            ret.sequences[0, -1] = model.config.audio_config.audiogen_end_token_id
            plen = ret.sequences.shape[1]
            ret, text_segment = generate_text_step(ret, plen, True, True)
            full_text += re.sub(special_token_partten, '', text_segment)
            show_text += re.sub(special_token_partten, '', text_segment) 
            print(f"ROUND {i+1}:", text_segment)

        if len(wave_list) > start:
            wave = wave_concat(wave_list, start, overlap=wave_concat_overlap)
            yield show_text, full_text, (sampling_rate, (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16),)
    
    yield show_text, full_text, None

def load_audio(audio_path):
    wave, sr = torchaudio.load(audio_path)
    wave_pkg = (sampling_rate, (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16))
    return wave_pkg


global g_history
global g_turn_i
g_history = []
g_turn_i = 0

os.makedirs(g_cache_dir, exist_ok=True)

def clear_history():
    global g_history
    global g_turn_i
    global g_cache_dir
    g_history = []
    g_turn_i = 0


def preprocess_messages(messages, audiogen_flag=True):
    text = ""
    print(messages)
    for i, msg in enumerate(messages):
        if audiogen_flag and msg["role"] == "assistant":
            text += role_prefix['audiogen']
        text += role_prefix[msg['role']]
        text += msg['content']
    if audiogen_flag:
        text += role_prefix['audiogen']
    text += role_prefix["assistant"]
    return text

def postprocess_messages(messages):
    new_messages = []
    for i, msg in enumerate(messages):
        new_messages.append(
            {
            'role': msg['role'],
            'content': re.sub(r'<audio_start_baichuan>.*?<audio_end_baichuan>|<audiogen_start_baichuan>.*?<audiogen_end_baichuan>', '<audio>', msg['content'])
        })
    return new_messages

def generate_one_turn(input_audio_path, system_prompt, audiogen_flag=True):
    global g_history
    global g_turn_i
    global g_cache_dir
    print("input_audio_path", input_audio_path)
    if len(g_history) == 0:
        g_history.append({
            "role": "system", 
            "content": system_prompt
        })

    fn_wav = os.path.join(g_cache_dir, f'user_turn{g_turn_i}.wav')
    os.system(f"cp {input_audio_path} {fn_wav}")
    g_history.append({
        "role": "user", 
        "content": audio_start_token + ujson.dumps({'path': fn_wav}, ensure_ascii=False) + audio_end_token
    })
    message = preprocess_messages(g_history, audiogen_flag)
    print("message", message)
    for show_text, full_text, wave_segment in generate_response(message, audiogen_flag):
        if wave_segment is not None and audiogen_flag:
            yield wave_segment, show_text, postprocess_messages(g_history)
        else:
            yield None, show_text, postprocess_messages(g_history)
    g_history.append({
        'role': 'assistant',
        'content': full_text,
    })

    print("History!!!")
    print(g_history)
    g_turn_i += 1


# 创建 Gradio 界面
with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], format="wav", type="filepath")
            submit = gr.Button("submit")
            clear = gr.Button("clear")
            system_prompt_input = gr.Textbox(label="System Prompt", value="请用【邻家女声】这个声音回答问题。")
            audio_flag = gr.Checkbox(label='response in audio', value=True)
            
        with gr.Column():
            chat_box = gr.Chatbot(type="messages")
            generated_text = gr.Textbox(label="Generated Text", lines=5, max_lines=200)
            generated_audio = gr.Audio(
                    label="Generated Audio",
                    streaming=True,
                    autoplay=True,
                    format="wav",
                    every=gr.Timer(0.01),
                )
    # 定义按钮的交互逻辑
    submit.click(generate_one_turn, 
        inputs=[audio_input, system_prompt_input, audio_flag], 
        outputs=[generated_audio, generated_text, chat_box])
    clear.click(clear_history)
    
# 启动应用
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
    server_port=145,
    debug=False,
    share_server_protocol="https",)
