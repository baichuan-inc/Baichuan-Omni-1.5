import gradio as gr
import numpy as np
import time, os, sys
# from funasr import AutoModel
import torch, torchaudio
from collections  import defaultdict
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import json
import re
import base64
from skimage.transform import resize
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from generation import decode_wave_vocoder, GenerationAudioTokens
from PIL import Image, ImageOps
from io import BytesIO
from gradio import image_utils, utils
import wave
import cv2
from constants import *

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
            # new_wave_list.append(torch.zeros(1, overlap))
        new_wave_list.append(wave)
        cur += 1
    return torch.cat(new_wave_list, dim=1)
 

def generate_response(input_string):

    print(input_string)
    ret = model.processor([input_string])
    textlen = ret.input_ids.shape[1]
    images = [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.images] if ret.images is not None else None
    videos = [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.videos] if ret.videos is not None else None
    textret = model.generate(input_ids=ret.input_ids.cuda(),
        attention_mask=ret.attention_mask.cuda() if ret.attention_mask is not None else None,
        labels=ret.labels.cuda() if ret.labels is not None else None,
        audios=ret.audios.cuda() if ret.audios is not None else None,
        images = [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.images] if ret.images is not None else None,
        patch_nums = patch_nums if ret.patch_nums is not None else None,
        images_grid = ret.images_grid if ret.images_grid is not None else None,
        videos = [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.videos] if ret.videos is not None else None,
        videos_patch_nums=ret.videos_patch_nums if ret.videos_patch_nums is not None else None,
        videos_grid = ret.videos_grid if ret.videos_grid is not None else None,
        encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
        bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
        max_new_tokens=256,
        tokenizer=tokenizer,
        stop_strings=["<audiogen_start_baichuan>", "，", "。"],
        do_sample=True,
        temperature=0.3,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.05,
        return_dict_in_generate=True,
    )

    message =  {
            "role": "assistant",
            "content": []
        }

    wave_list = []
    full_text = ""
    start = 0
    print("ROUND 0:", tokenizer.decode(textret.sequences[0, textlen:]))
    cur_text = tokenizer.decode(textret.sequences[0, textlen:])
    cur_msg = {
            "text": cur_text,
            "audio": None,
    }

    yield "", (24000, np.zeros(24000 * 2, dtype=np.int16))
    for i in range(100):
        m = textret.sequences[0, -1].item()
        if m == tokenizer.eos_token_id:
            if textret.sequences.shape[1] - textlen > 1:
                textret.sequences[0, -1] = (
                    model.config.audio_config.audiogen_start_token_id
                )
                ret = GenerationAudioTokens.generate(
                    model,
                    textret.sequences,
                    attention_mask=torch.ones_like(textret.sequences),
                    # position_ids = position_ids[:,-1:],
                    past_key_values=(
                        textret.past_key_values
                        if textret.past_key_values is not None
                        else None
                    ),
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.5,
                    top_k=5,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    return_dict_in_generate=True,
                )
                wave_list.extend(
                    decode_wave_vocoder(ret.audios_sequences.clone(), vocoder, model)
                )
            break

        textret.sequences[0, -1] = model.config.audio_config.audiogen_start_token_id
        gen_start_time = time.time()
        ret = GenerationAudioTokens.generate(
            model,
            textret.sequences,
            attention_mask=torch.ones_like(textret.sequences),
            # position_ids = position_ids[:,-1:],
            past_key_values=(
                textret.past_key_values if textret.past_key_values is not None else None
            ),
            max_new_tokens=500,
            do_sample=True,
            temperature=0.5,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
        )
        gen_end_time = time.time()
        print(f"generate time cost: {gen_end_time-gen_start_time:.2f}s")
        wave_start_time = time.time()

        wave = decode_wave_vocoder(ret.audios_sequences.clone(), vocoder, model)
        wave_end_time = time.time()
        print(f"decode_wave_vocoder time cost: {wave_end_time-wave_start_time:.2f}s")
        wave_list.extend(
            wave
        )
        cur_text = tokenizer.decode(textret.sequences[0, textlen:])
        full_text += cur_text

        fn_wav = os.path.join(g_cache_dir, f'audio/assistant_turn_round{i}.wav')
        torchaudio.save(fn_wav, torch.cat(wave, dim=0).cpu(), 24000)
        cur_msg["audio"] = fn_wav
        message["content"].append(cur_msg)

        if len(wave_list) > max(1, start):
            wave = wave_concat(wave_list, start, overlap=int(24000 * 0.025))
            start = len(wave_list)
            yield full_text, (
                24000,
                (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(
                    np.int16
                ),
            )

        ret.sequences[0, -1] = model.config.audio_config.audiogen_end_token_id
        # ret.sequences = torch.cat([ret.sequences, torch.tensor([m]).cuda().unsqueeze(0)],dim=1)
        textlen = ret.sequences.shape[1]
        textret = model.generate(
            ret.sequences,
            attention_mask=torch.ones_like(ret.sequences),
            # position_ids = (textmask.long().cumsum(dim=1))[:,-2:],
            tokenizer=tokenizer,
            stop_strings=["<audiogen_start_baichuan>"],
            past_key_values=(
                ret.past_key_values if ret.past_key_values is not None else None
            ),
            max_new_tokens=50,
            do_sample=True,
            temperature=0.3,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
        )
        print(f"ROUND {i+1}:", tokenizer.decode(textret.sequences[0, textlen:]))

        cur_msg = {
                "text": cur_text,
                "audio": None,
        }

    full_text += tokenizer.decode(textret.sequences[0, textlen:])
    if len(wave_list) > start:
        wave = wave_concat(wave_list, start, overlap=int(24000 * 0.01))
        yield full_text, (
            24000,
            (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(
                np.int16
            ),
        )

def load_audio(audio_path):
    wave, sr = torchaudio.load(audio_path)
    wave_pkg = (24000, (torch.clamp(wave.squeeze(), -0.99, 0.99).numpy() * 32768.0).astype(np.int16))
    return wave_pkg

        
def numpy_to_wav(audio_data, sample_rate=44100, file_name='output.wav'):
    # 确保音频数据是以 float32 格式存储
    # audio_data = np.asarray(audio_data, dtype=np.float32)

    # # 将数据归一化到 16-bit PCM 范围
    # audio_data = (audio_data * 32767).astype(np.int16)

    # 创建一个 WAV 文件对象
    with wave.open(file_name, 'wb') as wav_file:
        # 设置参数：声道数、样本宽度（字节数）、采样率、总帧数
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16-bit PCM 格式
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    
def audio_packing(system_prompt, audio_input, audio_pack_stream, image_pack_stream, max_frame):
    if audio_input == None:
        return None, None, None, None
    sr, data = audio_input
    input_audio_path = f'{g_cache_dir}/audio/{int(time.time())}.wav'
    numpy_to_wav(data, sr, input_audio_path)
    print("input_audio_path", input_audio_path)
    
    input_video_path = None
    if len(image_pack_stream) != 0:
        last_frame = None
        input_video_path = f'{g_cache_dir}/video/{int(time.time())}.mp4'
        min_frame = min(len(image_pack_stream), max_frame)
        img_str = '<video_start_baichuan>'
        videodims = (image_pack_stream[0].size)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        video = cv2.VideoWriter(input_video_path, fourcc, 1, videodims)
        print("min_frame:", min_frame)
        for i in range(-min_frame, 0, 1):
            img_filename = f'{g_cache_dir}/image/frame_{int(time.time()*1000)}.png'
            flipped_image = ImageOps.mirror(image_pack_stream[i])  # 水平翻转
            flipped_image.save(img_filename)
            last_frame = flipped_image.copy()
            video.write(cv2.cvtColor(np.array(last_frame), cv2.COLOR_RGB2BGR))
        video.release()
        print("input_video_path", input_video_path)
    

    message = "<B_SYS>"
    message += system_prompt
    message += "<C_Q>"
    if input_video_path is not None:
        message += "<video_start_baichuan>"
        message += json.dumps({"local": input_video_path})
        message += "<video_end_baichuan>"
    message += "<audio_start_baichuan>"
    message += json.dumps({"path": input_audio_path})
    message += "<audio_end_baichuan><audiotext_start_baichuan>"
    message += "<C_A>"

    if input_video_path is None:
        for text, wav_pkg in generate_response(input_string=message):
            yield wav_pkg, text, None, None
    else:
        for text, wav_pkg in generate_response(input_string=message):
            yield wav_pkg, text, last_frame, input_video_path


def video_packing(image_input, image_pack_stream):
    try:
        w, h = image_input.size
    except:
        return None
    image_input = image_input.resize((w//2, h//2), Image.Resampling.BILINEAR)
    if len(image_pack_stream) > 1000:
        image_pack_stream = image_pack_stream[1:]
    image_pack_stream.append(image_input)

    return image_pack_stream, len(image_pack_stream)

def clean_video(image_pack_stream):
    image_pack_stream = []
    return image_pack_stream

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(sources='webcam', streaming=True, type='pil')
        audio_input = gr.Audio(sources=["microphone"], streaming=False, format="wav")
    with gr.Row():
        slider = gr.Slider(4, 20, 4, step=1, label="帧数")
        video_que_size = gr.Text(label='视频队列大小')
        system_prompt_input = gr.Textbox(label="System Prompt", value="你是一个AI助手，请用【邻家女声】这个声音回答问题。")
    with gr.Row():
        text_output = gr.Textbox(label="Generated Text", lines=5, max_lines=200)
        show_vad_result = gr.Text(label='vad intervals')
        show_asr_result = gr.Text(label='asr result')
        show_llm_result = gr.Text(label='llm result')
        image_output = gr.Image(type='pil', label="最后帧")
        video_output = gr.Video(label="输出视频",show_download_button=True,format='mp4', autoplay=True)
        audio_output = gr.Audio(label="Generated Audio", streaming=True, autoplay=True, format="wav", every=gr.Timer(0.01))
    
    asr_model_cache = gr.State({})
    audio_pack_stream = gr.State()
    image_pack_stream = gr.State([])
    vad_state = gr.State({'idx':0, 'intervals':[], 'speaking': False, 'last_speaking': True, 'audio': None, 'cache':{}, 'image':None})
    asr_state = gr.State({'idx':0, 'delay_step': 0, 'result':[''], 'cache':{}, 'audio': None, 'last_speaking': True, 'image':None})
    query_state = gr.State([])
    ans_stage = gr.State("")
    ans_chunk = gr.State("")
    max_frame = gr.State(4)
    
    clear = gr.Button("Clear")

    slider.change(lambda x: x, slider, max_frame)
    # 下采样音频写入队列
    audio_input.change(audio_packing, 
                       [system_prompt_input, audio_input, audio_pack_stream, image_pack_stream, max_frame], 
                       [audio_output, text_output, image_output, video_output])
    audio_input.start_recording(clean_video, image_pack_stream, image_pack_stream)

    image_input.stream(video_packing, 
                       [image_input, image_pack_stream], 
                       [image_pack_stream, video_que_size])


if __name__ == "__main__":
    demo.launch(ssl_verify=False, share=False, server_name="0.0.0.0", server_port=1234, share_server_protocol="https", allowed_paths=[g_cache_dir])