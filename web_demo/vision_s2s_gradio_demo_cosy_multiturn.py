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
from PIL import Image
from decord import VideoReader, cpu
import shutil
import io
import cv2

os.makedirs(g_cache_dir, exist_ok=True)
os.makedirs(g_cache_dir+"/image", exist_ok=True)
os.makedirs(g_cache_dir+"/audio", exist_ok=True)
os.makedirs(g_cache_dir+"/video", exist_ok=True)

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
    model.bind_processor(tokenizer, training=False, relative_path=g_cache_dir)
    return model, tokenizer
model, tokenizer = init_model()

video_start_token = tokenizer.convert_ids_to_tokens(model.config.video_config.video_start_token_id)
video_end_token = tokenizer.convert_ids_to_tokens(model.config.video_config.video_end_token_id)
image_start_token = tokenizer.convert_ids_to_tokens(model.config.video_config.image_start_token_id)
image_end_token = tokenizer.convert_ids_to_tokens(model.config.video_config.image_end_token_id)
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
        textret = model.generate(input_ids=pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda() if pret.attention_mask is not None else None,
            labels=pret.labels.cuda() if pret.labels is not None else None,
            audios=pret.audios.cuda() if pret.audios is not None else None,
            images = [torch.tensor(img, dtype=torch.float32).cuda() for img in pret.images] if pret.images is not None else None,
            patch_nums = pret.patch_nums if pret.patch_nums is not None else None,
            images_grid = pret.images_grid if pret.images_grid is not None else None,
            videos = [torch.tensor(img, dtype=torch.float32).cuda() for img in pret.videos] if pret.videos is not None else None,
            videos_patch_nums=pret.videos_patch_nums if pret.videos_patch_nums is not None else None,
            videos_grid = pret.videos_grid if pret.videos_grid is not None else None,
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
                stop_strings=[audiogen_start_token,',','!','?','，','。','！','？','. '],
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
    if sr != sampling_rate:
        wave = torchaudio.functional.resample(wave, sr, sampling_rate)
    wave_pkg = (sampling_rate, (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16))
    return wave_pkg

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return None
    success, buffer = cv2.imencode('.mp4', frame)
    if not success:
        return None
    return buffer.tobytes()

def load_image(image_path):
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

def is_video(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

global g_history
global g_turn_i
g_history = []
g_turn_i = 0

def clear_history():
    global g_history
    global g_turn_i
    global g_cache_dir
    g_history = []
    g_turn_i = 0
    # os.system(f"rm -rf {g_cache_dir}")
    return None, None, None, None, None, None

def clear_upload_file():
    return None, None, None, None

def preprocess_messages(messages, audiogen_flag=True):
    text = ""
    for i, msg in enumerate(messages):
        if audiogen_flag and msg["role"] == "assistant":
            text += role_prefix['audiogen']
        text += role_prefix[msg['role']]
        text += msg['content']
    if audiogen_flag:
        text += role_prefix['audiogen']
    text += role_prefix["assistant"]
    return text

def parse_assistant_content(content):
    if '<audiogen_start_baichuan>' in content:
        wave = []
        text = ""

        result = []

        parts = re.split(r'<audiogen_start_baichuan>', content)
        prev_text = parts[0].strip()
        
        for part in parts[1:]:
            end_split = re.split(r'<audiogen_end_baichuan>', part, 1)
            
            if len(end_split) != 2:
                continue  
                
            json_str, remaining = end_split
            json_str = json_str.strip()

            cleaned_json = json_str.replace('\\/', '/')
            
            try:
                json_data = json.loads(cleaned_json)
            except json.JSONDecodeError:
                continue  
            if prev_text:
                result.append((prev_text, json_data))
                
            prev_text = remaining.strip()
    
        for t, w in result:
            text += t
            wav_pkg = load_audio(w["path"])
            wave.append(wav_pkg[1])
        wave = np.concatenate(wave, axis=0)
        return text, (wav_pkg[0], wave)
    else:
        return content, None  # Return None if no audio generated


def split_text(text, match_regex):
    matches = list(re.finditer(match_regex, text))
    # 初始化结果列表
    result = []
    match_flag_list = []
    # 上一个匹配的结束位置
    last_end = 0
    # 遍历所有匹配项
    for match in matches:
        # 添加匹配项之前的部分
        if text[last_end:match.start()]:
            result.append(text[last_end:match.start()])
            match_flag_list.append(False)
        # 添加匹配项
        result.append(match.group(0))
        match_flag_list.append(True)
        # 更新上一个匹配的结束位置
        last_end = match.end()
    # 添加最后一个匹配项之后的部分
    if text[last_end:]:
        result.append(text[last_end:])
        match_flag_list.append(False)
    return result, match_flag_list

def split_multimodal_chunk(text_list, mm_label_list, mtype='audio'):
    # 抽取text中的json格式音频/图像信息，读取并转化为特征，同时估计encoder token数，填入对应数量的pad token
    if (audio_start_token != None) and (mtype == 'audio'):
        match_regex = re.compile(audio_start_token + '.*?' + audio_end_token,re.S)
        drop_regex = re.compile(audio_start_token + "|" + audio_end_token,re.S)
    elif (image_start_token != None) and (mtype == 'image'):
        match_regex = re.compile(image_start_token + '.*?' + image_end_token,re.S)
        drop_regex = re.compile(image_start_token + "|" + image_end_token,re.S)
    elif (video_start_token != None) and (mtype == 'video'):
        match_regex = re.compile(video_start_token + '.*?' + video_end_token,re.S)
        drop_regex = re.compile(video_start_token + "|" + video_end_token,re.S)
    else:
        raise ValueError("mtype not supportted!")
    new_text_list = []
    new_mm_label_list = []
    for text,mm_label in zip(text_list,mm_label_list):
        for t,m in zip(*split_text(text, match_regex)):
            if m:
                new_text_list.append(re.sub(drop_regex, '', t))
                new_mm_label_list.append(mtype)
            else:
                new_text_list.append(t)
                new_mm_label_list.append(mm_label)
    return new_text_list, new_mm_label_list

def parse_user_content(content):
    new_messages = []
    
    all_text_list = [content]
    all_mm_label_list = ['text']
    # 处理多模态信息
    for mtype in ["image", "video", "audio"]:
        all_text_list, all_mm_label_list = split_multimodal_chunk(all_text_list, all_mm_label_list, mtype)
    
    for text, mm_label in zip(all_text_list, all_mm_label_list):
        if mm_label == 'audio':
            mm_info = re.sub(re.compile(audio_start_token + "|" + audio_end_token), '', text)
            audio_path = json.loads(mm_info)["path"]
            print(audio_path)
            # wav_pkg = load_audio(audio_path)
            audio_content = gr.Audio(audio_path)
            new_messages.append(
                {
                    "role": "user",
                    "content": audio_content,
                }
            )
        elif mm_label == 'image':
            mm_info = re.sub(re.compile(image_start_token + "|" + image_end_token), '', text)
            image_path = json.loads(mm_info)["local"]
            print(image_path)
            image_content = gr.Image(image_path)
            new_messages.append(
                {
                    "role": "user",
                    "content": image_content,
                }
            )
        elif mm_label == 'video':
            mm_info = re.sub(re.compile(video_start_token + "|" + video_end_token), '', text)
            video_path = json.loads(mm_info)["local"]
            print(video_path)
            video_content = gr.Video(video_path)
            new_messages.append(
                {
                    "role": "user",
                    "content": video_content,
                }
            )
        elif mm_label == 'text':
            new_messages.append(
                {
                    "role": "user",
                    "content": text,
                }
            )
        else:
            raise ValueError(f"mm_label not supportted! must in ['audio', 'image', 'video', 'text'] but get {mm_label}")
    
    return new_messages

def postprocess_messages(messages):
    
    new_messages = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            text, wave = parse_assistant_content(content)
            if wave is not None:
                new_messages.append(
                    {
                        "role": role,
                        "content": gr.Audio(wave),
                    }
                )
            new_messages.append(
                {
                    "role": role,
                    "content": text,
                }
            )
        elif role == "user":
            new_messages += parse_user_content(content)
        else: # system
            new_messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
    return new_messages

def generate_one_turn(input_audio_path, system_prompt, query, input_image_file, input_video_file, audiogen_flag=True):
    global g_history
    global g_turn_i
    global g_cache_dir

    if len(g_history) == 0:
        g_history.append({
            "role": "system", 
            "content": system_prompt
        })

    content = ""
    if input_image_file is not None:
        print("input_image_path", input_image_file)
        if isinstance(input_image_file, list):
            for image_file in input_image_file:
                image_filename = os.path.basename(image_file.name)
                fn_image = os.path.join(g_cache_dir, f'image/{image_filename}')
                shutil.copy(image_file.name, fn_image)
                content += image_start_token + ujson.dumps({'local': fn_image}, ensure_ascii=False) + image_end_token
        else:
            image_filename = os.path.basename(input_image_file.name)
            fn_image = os.path.join(g_cache_dir, f'image/{image_filename}')
            shutil.copy(input_image_file.name, fn_image)
            content += image_start_token + ujson.dumps({'local': fn_image}, ensure_ascii=False) + image_end_token
    
    if input_video_file is not None:
        print("input_video_path", input_video_file)
        if isinstance(input_video_file, list):
            for video_file in input_video_file:
                video_filename = os.path.basename(video_file.name)
                fn_video = os.path.join(g_cache_dir, f'video/{video_filename}')
                shutil.copy(video_file.name, fn_video)
                content += video_start_token + ujson.dumps({'local': fn_video}, ensure_ascii=False) + video_end_token
        else:
            video_filename = os.path.basename(input_video_file.name)
            fn_video = os.path.join(g_cache_dir, f'video/{video_filename}')
            shutil.copy(input_video_file.name, fn_video)
            content += video_start_token + ujson.dumps({'local': fn_video}, ensure_ascii=False) + video_end_token
    
    if input_audio_path is not None:
        print("input_audio_path", input_audio_path)
        fn_wav = os.path.join(g_cache_dir, f'audio/user_turn{g_turn_i}.wav')
        shutil.copy(input_audio_path, fn_wav)
        content += audio_start_token + ujson.dumps({'path': fn_wav}, ensure_ascii=False) + audio_end_token

    if query is not None:
        content += query

    g_history.append({
        "role": "user", 
        "content": content
    })

    message = preprocess_messages(g_history, audiogen_flag)
    print("message", message)
    for show_text, full_text, wave_segment in generate_response(message, audiogen_flag):
        if wave_segment is not None and audiogen_flag:
            post = postprocess_messages(g_history)
            yield wave_segment, show_text, postprocess_messages(g_history)
        else:
            post = postprocess_messages(g_history)
            yield None, show_text, postprocess_messages(g_history)
    g_history.append({
        'role': 'assistant',
        'content': full_text,
    })
    g_turn_i += 1
    
def convert_webm_to_mp4(input_file, output_file):
    try:
        cap = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error: {e}")
        raise

def add_image(file):
    return file

def add_video(file):
    return file
        
with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            query = gr.Textbox(lines=2, label='Text Input')
            audio_input = gr.Audio(sources=["microphone", "upload"], format="wav", type="filepath")

            # video_input = gr.Video(sources=[ "webcam"], height=400, width=700, container=True, interactive=True, show_download_button=True, label="📹 Video Recording (视频录制)")
            
            with gr.Row():
                add_image_file_btn = gr.UploadButton("📁 Upload (上传图片)", file_types=["image"])
                add_video_file_btn = gr.UploadButton("📁 Upload (上传视频)", file_types=["video"])
            with gr.Row():
                image_output = gr.Image(type='pil', label="图像")
                video_output = gr.Video(label="视频",show_download_button=True, format='mp4', autoplay=False)

            submit = gr.Button("submit")
            clear = gr.Button("clear")
            system_prompt_input = gr.Textbox(label="System Prompt", value="请用【邻家女声】这个声音回答问题。")
            audio_flag = gr.Checkbox(label='response in audio', value=True)
            
        with gr.Column():
            chat_box = gr.Chatbot(label="History", type="messages")
            generated_text = gr.Textbox(label="Generated Text", lines=5, max_lines=200)
            generated_audio = gr.Audio(
                    label="Generated Audio",
                    streaming=True,
                    autoplay=True,
                    format="wav",
                    every=gr.Timer(0.01),
                )


    # 定义按钮的交互逻辑
    # video_input.stop_recording(add_video, [video_input], [video_output], show_progress=True)
    add_image_file_btn.upload(add_image, [add_image_file_btn], [image_output], show_progress=True)
    add_video_file_btn.upload(add_video, [add_video_file_btn], [video_output], show_progress=True)

    submit.click(generate_one_turn, 
        inputs=[audio_input, system_prompt_input, query, add_image_file_btn, add_video_file_btn, audio_flag], 
        outputs=[generated_audio, generated_text, chat_box]
    ).then(
        clear_upload_file, [], [query, audio_input, add_image_file_btn, add_video_file_btn], queue=False
    )
    clear.click(clear_history, [], [query, audio_input, add_image_file_btn, add_video_file_btn, image_output, video_output])

    
# 启动应用
if __name__ == "__main__":
    demo.launch(
        ssl_verify=False, 
        share=True, 
        server_name="0.0.0.0",
        server_port=12345,
        debug=False,
        share_server_protocol="https",
        allowed_paths=[g_cache_dir]
    )
