import requests
import re, ujson, os, sys, fire, glob, random, time, json
import numpy as np
import io
import torch
from torch.utils.data import default_collate
import torchaudio
from typing import *
from dataclasses import dataclass, field
import transformers
from transformers.modeling_outputs import ModelOutput
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from functools import lru_cache
from io import BytesIO
from PIL import Image
import concurrent.futures as cf
from transformers.image_transforms import resize, center_crop, get_resize_output_image_size
from transformers.image_utils import PILImageResampling
from PIL import Image, ImageOps
from PIL import ImageFile
torch.set_num_threads(1)  # 限制torch的线程数 否则可能会卡住
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64
from decord import VideoReader, cpu
import cv2
import av
import imagesize
import tempfile
import math
from multiprocessing import Pool
from cairosvg import svg2png
import hashlib

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


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


def read_video(image_path, max_frame_number, decode_way):
    if decode_way=='1fps':
        try:
            # print(image_path)
            vr = VideoReader(image_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            frames = vr.get_batch(frame_idx).asnumpy()
            cnt = len(frames)
            frame_times = range(cnt)
        except Exception as e:
            print(image_path)
            print('error is', e)
            return None
    elif decode_way=='key':
        try: 
            with av.open(image_path) as container:                         
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'
                frames = []
                frame_times = []
                fps = int(stream.average_rate)
                cnt = 0
                for frame in container.decode(stream): # 关键帧存成image patch
                    image = np.array(frame.to_image())
                    frames.append(image)
                    frame_time = int(frame.time)
                    frame_times.append(frame_time)
                    cnt += 1
        except Exception as e:
            print('error is', e)
            return None
    if frames is None or len(frames)==0:
        return None
    if len(frames)>max_frame_number and max_frame_number>0:
        # 生成14个均匀间隔的索引
        indices = np.linspace(0, len(frames) - 1, max_frame_number, dtype=int)
        # 根据索引获取对应元素
        frames = frames[indices]
        frame_times = frame_times[indices]
    return frames, frame_times


class OmniImageProcessor:
    def __init__(self, config, **kwargs):
        self.config = config  # visual_config
        self.min_pixels = self.config.min_pixels if hasattr(self.config, 'min_pixels') else 56 * 56
        self.max_pixels = self.config.max_pixels if hasattr(self.config, 'max_pixels') else 28 * 28 * 1280
        self.patch_size = self.config.patch_size if hasattr(self.config, 'patch_size') else 14
        self.temporal_patch_size = self.config.temporal_patch_size if hasattr(self.config, 'temporal_patch_size') else 2
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.spatial_merge_size = self.config.spatial_merge_size if hasattr(self.config, 'spatial_merge_size') else 2

    def image_transform(self, strseq, return_mm_data = True):
        image = None
        if isinstance(strseq, str):
            if return_mm_data:
                image = Image.open(strseq).convert("RGB") 
        else:
            try:
                image = Image.open(BytesIO(strseq)).convert("RGB")
            except:
                image = Image.open(BytesIO(svg2png(bytestring=strseq))).convert("RGB") # interleaved有的是矢量图，需要转换
            
        image = np.array(image.convert("RGB")) # 这一步首先将图像转换为 RGB 格式，确保图像有三个通道（R、G、B）。然后使用 np.array() 将其转换为 NumPy 数组，方便后续处理。
        image_org_size = image.shape[:2] # 这里保存了图像的原始大小（高度和宽度），image.shape 返回图像的形状 (高度, 宽度, 通道数)，而 image.shape[:2] 提取了前两个值，即原始的高度和宽度。这个信息可以用于后续的对比或其他处理。
        
        # resize, crop, scale, normalize
        # 输出一个新的尺寸，这个尺寸通常是 (宽度, 高度) 格式，用于后续的图像调整操作，如缩放或裁剪。
        resized_height, resized_width = smart_resize(
            image_org_size[0], image_org_size[1],
            factor=self.patch_size * self.spatial_merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        output_size = (resized_height, resized_width)
        
        # 使用 resize 函数将图像调整到 output_size 大小。PILImageResampling.BICUBIC 指定使用双三次插值法来进行图像缩放，这种方法通常能够提供较好的图像质量。
        # image: 输入的图像数据，可以是 NumPy 数组或 PIL 图像对象；output_size: 目标大小，通常是一个二元组 (宽度, 高度)。这个尺寸可以是图像的绝对大小，也可以是相对于原始图像的比例；
        # resample: 可选的重采样方法，通常用于确定如何插值像素。例如，PILImageResampling.BICUBIC 表示使用双三次插值法，这是一种平滑的插值方法，常用于图像缩放。
        image = resize(image, output_size, PILImageResampling.BICUBIC)
        img = image.transpose(2, 0, 1)
        # 对图像进行归一化和标准化处理
        image = (img / 255.0 - np.array(self.config.image_mean)[:, np.newaxis, np.newaxis]) / np.array(self.config.image_std)[:,np.newaxis,np.newaxis]
        # 处理成patch
        patches = image[np.newaxis, :]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)


class OmniAudioProcessor:
    # 包含基本的音频特征抽取模块 + 输入数据解析模块
    def __init__(
        self,
        config,  # audio processor config
        **kwargs
    ):
        # make sure you have install 'conda install -c conda-forge 'ffmpeg<7'' for torchaudio
        assert(len(torchaudio.list_audio_backends()) > 0)
        self.config = config
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.n_fft // 2,
            num_mel_filters=self.config.num_mel_bins,
            min_frequency=0.0,
            max_frequency=self.config.sampling_rate / 2.0,
            sampling_rate=self.config.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.window = torch.hann_window(self.config.n_fft)
        
    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-6):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def zero_mean_unit_var_norm(x):
        return (x - x.mean()) / torch.sqrt(x.var() + 1e-8)

    def load_audio_waveform(self, uri, return_tensors=True, do_normalize=False):
        metadata = torchaudio.info(uri)  # sample_rate, num_frames, num_channels, bits_per_sample, encoding=PCM_S
        assert(metadata.num_channels <= 2), "acoustic file with {} channels.".format(metadata.num_channels)  # whisper only accept mono channel audio
        waveform_tensor, _ = torchaudio.load(uri, normalize=True)
        if self.config.sampling_rate != metadata.sample_rate:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, metadata.sample_rate, self.config.sampling_rate, lowpass_filter_width=128)

        # downmix to mono channel https://trac.ffmpeg.org/wiki/AudioChannelManipulation
        if metadata.num_channels > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

        # normalized to zero mean
        if do_normalize:
            waveform_tensor = self.zero_mean_unit_var_norm(waveform_tensor)

        if return_tensors:  # (channels, samples)
            return waveform_tensor
        else:
            return waveform_tensor.numpy()  

    def split_with_overlap(self, waveform):  # 如果长度超过最大长度限制 分割为带overlap的多段
        channels, wave_samples = waveform.shape
        max_audio_samples = self.config.max_audio_seconds * self.config.sampling_rate
        if wave_samples <= max_audio_samples or self.config.split_overlap < 0:
            return [waveform]  # 没有超出最大长度or截断逻辑 统一返回list
        
        split_waveform, start = [], 0
        while start < wave_samples:  # 统一按秒数对齐overlap
            if start > int(self.config.sampling_rate * self.config.split_overlap):
                start -= int(self.config.sampling_rate * self.config.split_overlap)  # 0表示没有overlap，>0 overlap对应秒数
            end = min(start + max_audio_samples, wave_samples)
            if end - start>= self.config.n_fft: # 保证至少有一帧数据
                split_waveform.append(waveform[:, start:end])  # 注意这里可能会切割出特别短的片段 需要在预处理判断并丢弃
            start = end
        return split_waveform

    @classmethod        
    def inference_output_length(cls, config, input_length):
        # for whisper + bridge
        kernel_size = config.kernel_size
        stride_size = config.stride_size
        avg_pooler = config.avg_pooler
        encoder_length = (input_length + 2 * (kernel_size // 2) - kernel_size) // 1 + 1  # conv layer1 with pad=1
        encoder_length = (encoder_length + 2 * (kernel_size // 2) - kernel_size) // stride_size + 1  # conv layer2 with pad=1
        if avg_pooler > 1:
            bridge_length = encoder_length // avg_pooler
        return encoder_length, bridge_length

    def extract_fbank_features(self, waveform):
        # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
        channels, wave_samples = waveform.shape
        assert(wave_samples >= self.config.n_fft)
        valid_frame_nums = min(self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length, wave_samples // self.config.hop_length + 1)
        if wave_samples < self.config.max_audio_seconds * self.config.sampling_rate:
            waveform = torch.nn.functional.pad(waveform, (0, self.config.max_audio_seconds * self.config.sampling_rate - wave_samples), "constant", 0)
        else:
            waveform = waveform[:, :self.config.max_audio_seconds * self.config.sampling_rate]

        # window = torch.hann_window(self.config.n_fft)
        stft = torch.stft(waveform, self.config.n_fft, self.config.hop_length, window=self.window, return_complex=True)  # fft, len(wave) // n_fft // 2 + 1
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        log_spec = log_spec[0].numpy()  # (channel, filters, samples) -> (filters, samples)
        log_spec[:, valid_frame_nums:] = 0.0  # pad0

        return log_spec, valid_frame_nums

    def data_augment(self, feature: np.array, input_length, training=True):
        # reference https://arxiv.org/pdf/1904.08779
        def mask_start_indices(input_length, mask_length, min_masks, mask_prob):
            num_masked_span = int(mask_prob * input_length / mask_length + random.random())
            num_masked_span = max(num_masked_span, min_masks)
            start_indices = list(range(input_length - mask_length))
            random.shuffle(start_indices)
            start_indices = start_indices[:num_masked_span]
            return start_indices

        if not training or (self.config.mask_time_prob <= 0 and self.config.mask_feature_prob <= 0):
            return feature
        if input_length < self.config.mask_time_length * self.config.mask_time_min_masks + 1:
            return feature
        if self.config.num_mel_bins < self.config.mask_feature_length * self.config.mask_feature_min_masks + 1: 
            return feature
        
        if self.config.mask_time_prob > 0:
            start_indices = mask_start_indices(input_length, self.config.mask_time_length, self.config.mask_time_min_masks, self.config.mask_time_prob) 
            for start_idx in start_indices:
                feature[:, start_idx: start_idx + self.config.mask_time_length] = 0.0
        if self.config.mask_feature_prob > 0:
            start_indices = mask_start_indices(self.config.num_mel_bins, self.config.mask_feature_length, self.config.mask_feature_min_masks, self.config.mask_feature_prob) 
            for start_idx in start_indices:
                feature[start_idx: start_idx + self.config.mask_feature_length, :] = 0.0

        return feature

@dataclass
class OmniProcessorOutput(ModelOutput):  
    input_ids: Optional["List|torch.Tensor"] = None
    labels: Optional["List|torch.Tensor"] = None
    attention_mask: Optional["List|torch.Tensor"] = None
    position_ids: Optional["List|torch.Tensor"] = None
    seqlens: Optional["List|torch.Tensor"] = None  # 需要配合Omni Modeling使用
    # audio fields
    audios: Optional["List|torch.Tensor"] = None
    encoder_length: Optional["List|torch.Tensor"] = None
    bridge_length: Optional["List|torch.Tensor"] = None
    # image fields
    images: Optional["List|torch.Tensor"] = None
    patch_nums: Optional["List|torch.Tensor"] = None
    images_size: Optional["List|torch.Tensor"] = None
    crop_size: Optional["List|torch.Tensor"] = None
    images_grid: Optional["List|torch.Tensor"] = None
    # video fields
    videos: Optional["List|torch.Tensor"] = None
    videos_patch_nums: Optional["List|torch.Tensor"] = None
    videos_size: Optional["List|torch.Tensor"] = None
    videos_crop_size: Optional["List|torch.Tensor"] = None
    videos_grid: Optional["List|torch.Tensor"] = None
    # processor fields
    raw_text: Optional[str] = None
    index: Optional[int] = None

    def concatenate(self, other):  # 仅限list使用
        def concat_one(a, b):
            if a is None and b is None:
                return None
            elif a is None and b is not None:
                return b 
            elif a is not None and b is None: 
                return a 
            else: 
                return a + b
        return OmniProcessorOutput(
            input_ids=concat_one(self.input_ids, other.input_ids),
            labels=concat_one(self.labels, other.labels),
            audios=concat_one(self.audios, other.audios),
            encoder_length=concat_one(self.encoder_length, other.encoder_length),
            bridge_length=concat_one(self.bridge_length, other.bridge_length), 
            images=concat_one(self.images, other.images),
            images_grid=concat_one(self.images_grid, other.images_grid),
            patch_nums=concat_one(self.patch_nums, other.patch_nums),

            videos=concat_one(self.videos, other.videos),
            videos_grid=concat_one(self.videos_grid, other.videos_grid),
            videos_patch_nums=concat_one(self.videos_patch_nums, other.videos_patch_nums),

            position_ids=concat_one(self.position_ids, other.position_ids),
            seqlens=concat_one(self.seqlens, other.seqlens),
            images_size=concat_one(self.images_size, other.images_size),
            videos_size=concat_one(self.videos_size, other.videos_size),
            index = self.index # concat保持index不变
        )

class OmniMMProcessor(object):
    def __init__(self,
                tokenizer: transformers.PreTrainedTokenizer,
                config,
                training,
                relative_path=None,
                parallel=None,
                **kwargs, 
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.audio_processor = OmniAudioProcessor(config.audio_config)
        self.visual_processor = None
        if hasattr(config, "visual_config"):
            self.visual_processor = OmniImageProcessor(config.visual_config)
        self.video_processor = None
        if hasattr(config, "video_config"):
            self.video_processor = OmniImageProcessor(config.video_config)
        self.training = training
        self.relative_path = relative_path
        self.parallel = parallel
        # audio tag
        self.audio_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_start_token_id)
        self.audio_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_end_token_id)
        self.audio_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_pad_token_id)
        self.audio_delim_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_delim_token_id)
        self.audiogen_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_start_token_id)
        self.audiogen_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_end_token_id)
        # image tag
        self.image_start_tag = None
        self.image_end_tag = None
        self.image_pad_tag = None
        self.video_start_tag = None
        self.video_end_tag = None
        # videoframe tag只是为了兼容图片帧作为输入的情况，没有token id，在抽取视频帧的时候，会将这个替换成image tag的start、end
        self.videoframe_start_tag = '<videoframe_start_omni>'
        self.videoframe_end_tag = '<videoframe_end_omni>'
        if hasattr(self.config, "visual_config"):
            # special token for start_tag
            self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_start_token_id)
            # special token for end_tag
            self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_end_token_id)
            # special token for pad_tag
            self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_pad_token_id)
            self.image_line_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_line_token_id)
            self.image_delimiter_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_delimiter_token_id) 
        if hasattr(self.config, "video_config"):
            self.video_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_start_token_id)
            self.video_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_end_token_id)
            self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_start_token_id)
            self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_end_token_id)
            self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_pad_token_id)
            self.video_place_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_place_token_id)
            
            self.frame_pattern = getattr(self.config.video_config, 'frame_pattern', '<frame>')


    # @lru_cache(maxsize=1024)
    def _get_audio(self, audio_info):
        try:
            audio_info = ujson.loads(audio_info) 
            if 'path' in audio_info.keys():
                audio_uri = None
                if os.path.exists(audio_info['path']):
                    audio_uri = audio_info['path']
                elif self.relative_path is not None:
                    audio_uri = os.path.join(self.relative_path, audio_info['path'].lstrip('/'))
                    if not os.path.exists(audio_uri):
                        audio_uri = None
                if audio_uri is not None:
                    waveform = self.audio_processor.load_audio_waveform(audio_uri, True)
                waveforms = self.audio_processor.split_with_overlap(waveform) 
                
                ret = OmniProcessorOutput()  # 默认初始化 audios字段为None
                for i, waveform in enumerate(waveforms): #(zip(waveforms,vocoder_waveforms)):
                    audio, input_length = self.audio_processor.extract_fbank_features(waveform)
                    audio = self.audio_processor.data_augment(audio, input_length, self.training)
                    encoder_length, bridge_length = self.audio_processor.inference_output_length(self.config.audio_config, input_length)
                    if bridge_length <= 0: 
                        continue
                    current_ret = OmniProcessorOutput(
                        audios=[audio[:,:input_length]], 
                        encoder_length=[encoder_length], 
                        bridge_length=[bridge_length],
                        )
                    if ret.audios is None:
                        ret = current_ret
                    else:
                        ret = ret.concatenate(current_ret)  # 拼接多个切片
                return ret
            else:
                raise ValueError("can not find path in audio_info") 
        except Exception as e:
            print("**** get audio error: {}, info: {} *****".format(str(e), str(audio_info)))
        return OmniProcessorOutput()

    # @lru_cache(maxsize=1024)
    def _get_image(self, image_info):
        try:
            try:
                image_info = ujson.loads(image_info)
            except:
                image_info = re.sub(r"(?<!\\)'", '"', image_info)
                image_info = ujson.loads(image_info)
            if 'base64' in image_info.keys():
                image_data = base64.b64decode(image_info['base64'])
                image_feat, org_size, image_list = self.visual_processor.image_transform(image_data)
            elif 'local' in image_info.keys():
                image_feat, org_size, image_list = self.visual_processor.image_transform(image_info['local'])
            elif 'path' in image_info.keys() and os.path.exists(image_info['path']):
                image_feat, org_size, image_list = self.visual_processor.image_transform(image_info['path'])
            elif 'url' in image_info.keys():
                image_bytes = self._get_vision_obj_byte('url', image_info['url'])
                image_feat, org_size, image_list = self.visual_processor.image_transform(image_bytes)
            else:
                raise ValueError("can not find any path in image_info")
            
            merge_length = self.visual_processor.merge_size**2
            patch_nums = np.array(image_list).prod() // merge_length
            
            if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                return OmniProcessorOutput(
                        images=[image_feat],
                        patch_nums=[patch_nums],
                        crop_size=[image_list],
                        images_size= [org_size],
                        images_grid=[image_list]
                        )
            else:
                print("**** image too small: {}, info: {} *****".format(str(org_size), str(image_info)))
                return OmniProcessorOutput()
           
        except Exception as e:
            print("**** get image error: {}, info: {} *****".format(str(e), str(image_info)))
        return OmniProcessorOutput()
    
    # @lru_cache(maxsize=1024)
    def _get_video_frame(self, video_frame_infos):
        try:
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, video_frame_infos)
            ret = OmniProcessorOutput()
            # 逐个解析
            for match in matches:
                video_frame_info = ujson.loads(match)
                # video_frame_info = ujson.loads(video_frame_info)
                if 'local' in video_frame_info.keys():
                    image_feat, org_size, image_list = self.video_processor.image_transform(video_frame_info['local'])
                elif 'path' in video_frame_info.keys() and os.path.exists(video_frame_info['path']):
                    image_feat, org_size, image_list = self.video_processor.image_transform(video_frame_info['path'])
                else:
                    raise ValueError("can not find any path in video_info")

                merge_length = self.video_processor.merge_size**2
                patch_nums = np.array(image_list).prod() // merge_length
                
                if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                    ret = ret.concatenate(
                            OmniProcessorOutput(
                                videos=[image_feat],
                                videos_patch_nums=[patch_nums],
                                videos_crop_size=[image_list],
                                videos_size= [org_size],
                                videos_grid=[image_list]
                            )
                        )
                else:
                    print("**** video too small: {}, info: {} *****".format(str(org_size), str(video_frame_info)))
            return ret
           
        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e), str(video_frame_info)))
        return OmniProcessorOutput()

    # 读取视频
    def _get_vision_obj_byte(self, source, path):
        vision_obj_byte = None
        if source == "local":
            if os.path.exists(path):
                vision_obj_byte = open(path, "rb").read()
            else:
                vision_obj_byte = None
        if source == "base64":
            vision_obj_byte = base64.b64decode(path)
        if source == "url":
            vision_obj_byte = requests.get(url=path).content
        return vision_obj_byte
    
    # 将视频切分为帧，保存至子目录中
    def _split_video_to_frames(self, video_info, max_frame_number=-1, decode_way="1fps"):
        if decode_way=='1fps':
            frame_suffix = f'_frames'
        elif decode_way=='key':
            frame_suffix = f'_keyframes'
        else:
            raise ValueError('unvalid decode way!!!')
        
        server = "local"
        if 'local' in video_info.keys():
            # 本地路径
            video_path = video_info['local']
            # 帧保存本地路径
            frame_path = video_path[:video_path.rfind('.')] + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('local', video_path)
        elif 'base64' in video_info.keys():
            md5 = hashlib.md5(video_info['base64'].encode('utf-8')).hexdigest()
            if self.relative_path is not None: 
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('base64', video_info['base64'])
        elif 'url' in video_info.keys():
            md5 = hashlib.md5(video_info['url'].encode('utf-8')).hexdigest()
            if self.relative_path is not None: 
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('url', video_info['url'])
        else:
            raise ValueError('unvalid video server !!!')
            return ""
        
        if mm_obj_byte is None: # 未读取到视频文件
            return ""
        if not os.path.exists(frame_path) or len(os.listdir(frame_path))==0:
            # 保存帧
            os.makedirs(frame_path, exist_ok=True)
            frames, frame_times = read_video(io.BytesIO(mm_obj_byte), max_frame_number=-1, decode_way=decode_way) #读取全部帧
            for frame_idx, frame in enumerate(frames):
                output_filename = os.path.join(frame_path, f"{frame_times[frame_idx]}.jpg")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_filename, frame)
        frame_paths = os.listdir(frame_path)
        
        # 选取帧
        frame_times = [int(filename.split('/')[-1].replace('.jpg', '')) for filename in frame_paths if filename.endswith('.jpg')] # 文件名对应秒数
        frame_times.sort() #从小到大排序
        frame_number = len(frame_times)
        if frame_number > max_frame_number:
            indices = np.linspace(0, frame_number - 1, max_frame_number, dtype=int)
        else:
            indices = np.linspace(0, frame_number - 1, frame_number, dtype=int)
        # 拼接模式
        replace_str = ""
        for frame_idx, idx in enumerate(indices):
            frame_time = frame_times[idx]  # frame_time表示帧对应的时间 单位为s 同时也是存储的文件名
            frame_dict = {"local": os.path.join(frame_path, f'{frame_time}.jpg')}
            frame_str = self.frame_pattern.format(frame_idx) if '{}' in self.frame_pattern else self.frame_pattern  # {}对应的是第几张图片
            frame_str = frame_str.replace('<TIMEIDX>', str(frame_time))  # TIMEIDX对应的是第几秒
            frame_str = frame_str.replace('<TIMESTAMP>', time.strftime("%H:%M:%S", time.gmtime(frame_time)))  # TIMESTAMP对应的是时间戳
            frame_str = frame_str.replace('<frame>', f'{self.image_start_tag}{json.dumps(frame_dict)}{self.image_end_tag}')
            replace_str += frame_str
        
        return replace_str
    
    def sample_frame(self,frames_str,max_frame = 32):
        def uniform_sample(lst, num_samples):
            if num_samples > len(lst):
                return lst
            interval = len(lst) / num_samples
            samples = [lst[int(i * interval)] for i in range(num_samples)]
            return samples
        p = rf'({self.image_start_tag}.*?{self.image_end_tag})'
        frames_str_split = re.split(p,frames_str)
        frame_idxs = [idx for idx in range(len(frames_str_split)) if self.image_start_tag in frames_str_split[idx]]
        sample_frame_idxs = set(uniform_sample(frame_idxs, max_frame))
        return ''.join([item for idx,item in enumerate(frames_str_split) if idx in sample_frame_idxs or self.image_start_tag not in frames_str_split[idx]])

    def _get_video_frame_str(self, video_info):
        try:
            if self.videoframe_start_tag in video_info:#如果是以视频帧的形式表示一个视频，则替换成image tag
                frames_str = video_info
                frames_str = frames_str.replace(self.videoframe_start_tag,self.image_start_tag).replace(self.videoframe_end_tag,self.image_end_tag)
                return self.sample_frame(frames_str, max_frame = self.config.video_config.max_frame_num)
            video_info = ujson.loads(video_info)
            # 获取包含多帧图像路径的字符串，最大帧数量max_frame_number
            frames_str = self._split_video_to_frames(video_info, max_frame_number=self.config.video_config.max_frame_num, decode_way=self.config.video_config.decode_way)
            return frames_str
        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e), str(video_info)))
        return ""
    
    def _replace_image(self, image_text):
        image_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', image_text)
        ret = self._get_image(image_info)  # 重复取结果 cached result
        if ret.patch_nums is None:
            return ''
        return ret, self.image_start_tag + self.image_pad_tag * ret.patch_nums[0] + self.image_end_tag
    
    def _replace_video_frame(self, video_frame_text):
        video_frame_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', video_frame_text)
        ret = self._get_video_frame(video_frame_info)  # 重复取结果 cached result
        if ret.videos_patch_nums is None:
            return ''
        video_frame_str = [self.image_start_tag + self.video_place_tag * ret.videos_patch_nums[i] + self.image_end_tag for i in range(len(ret.videos_patch_nums))]
        return ret, ''.join(video_frame_str)
        
    
    def split_multimodal_chunk(self, text_list, mm_label_list, trainable_list, mtype='audio'):
        # 抽取text中的json格式音频/图像信息，读取并转化为特征，同时估计encoder token数，填入对应数量的pad token
        if (self.audio_start_tag != None) and (mtype == 'audio'):
            match_regex = re.compile(self.audio_start_tag + '.*?' + self.audio_end_tag,re.S)
            drop_regex = re.compile(self.audio_start_tag + "|" + self.audio_end_tag,re.S)
        elif (self.image_start_tag != None) and (mtype == 'image'):
            match_regex = re.compile(self.image_start_tag + '.*?' + self.image_end_tag,re.S)
            drop_regex = re.compile(self.image_start_tag + "|" + self.image_end_tag,re.S)
        elif (self.audiogen_start_tag != None) and (mtype == 'audiogen'):
            match_regex = re.compile(self.audiogen_start_tag + '.*?' + self.audiogen_end_tag,re.S)
            drop_regex = re.compile(self.audiogen_start_tag + "|" + self.audiogen_end_tag,re.S)
        elif (self.video_start_tag != None) and (mtype == 'video'):
            match_regex = re.compile(self.video_start_tag + '.*?' + self.video_end_tag,re.S)
            drop_regex = re.compile(self.video_start_tag + "|" + self.video_end_tag,re.S)
        else:
            raise ValueError("mtype not supportted!")
        new_text_list = []
        new_mm_label_list = []
        new_trainable_flag_list = []
        for text,mm_label,trainable in zip(text_list,mm_label_list,trainable_list):
            for t,m in zip(*split_text(text, match_regex)):
                new_trainable_flag_list.append(trainable)
                if m:
                    new_text_list.append(re.sub(drop_regex, '', t))
                    new_mm_label_list.append(mtype)
                else:
                    new_text_list.append(t)
                    new_mm_label_list.append(mm_label)
        return new_text_list, new_mm_label_list, new_trainable_flag_list
    
    def process_multimodal_chunk(self, text, mm_label, trainable):
        ret = OmniProcessorOutput()
        if mm_label == 'audio':
            ret = self._get_audio(text)
            if ret.bridge_length is not None:    
                ret.input_ids = self.tokenizer.encode(self.audio_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audio_end_tag,add_special_tokens=False)
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'audiogen':
            ret = self._get_audio(text)
            if ret.bridge_length is not None:    
                ret.input_ids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'image':
            ret, input_str = self._replace_image(text)
            if input_str:
                ret.input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
            else:
                raise ValueError("Get image data Failed at Process image chunk")
        elif mm_label == 'video':
            frame_str = self.video_start_tag+self._get_video_frame_str(text)+self.video_end_tag
            ret, input_str = self._replace_video_frame(frame_str)
            if input_str:
                ret.input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
            else:
                raise ValueError("Get video data Failed at Process video chunk")               
        elif mm_label == 'text':
            ret.input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ret.input_ids) > self.tokenizer.model_max_length-1:  # 过滤长文本
                raise ValueError(f"Text too long, please check text length！ 【{text[:5]+'...'*6+text[-5:]}】")
        else:
            raise ValueError(f"mm_label not supportted! must in ['audio', 'audiogen', 'image', 'video', 'text'] but get {mm_label}")
        return ret
    
    def process_one(self, text, index=0, raw_only=False):
        ret = OmniProcessorOutput(index=index)
        all_text_list = []
        all_mm_label_list = []
        all_trainable_flag_list = []
        text_list, match_flag = split_text(text, re.compile("<trainable_start>.*?<trainable_end>",re.S))
        if len(text_list) == 1:
            text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text_list[0])
            all_text_list.append(text)
            all_mm_label_list.append('text')
            all_trainable_flag_list.append(True)
        else:
            for text, match in zip(text_list, match_flag):
                text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text)
                if text.strip() == '':
                    continue  # 把多余的空格干掉
                all_text_list.append(text)
                all_mm_label_list.append('text')
                all_trainable_flag_list.append(match)
        # 处理多模态信息
        for mtype in self.config.multimodal:  # 循环获取音频 图像结果 
            all_text_list, all_mm_label_list, all_trainable_flag_list = self.split_multimodal_chunk(all_text_list, all_mm_label_list, all_trainable_flag_list, mtype)
        if len(all_text_list) == 0:
            print(f"Process {text} chunk error: No valid Text data!!!!!")
            return OmniProcessorOutput(index=index)
        
        for text, mm_label, trainable in zip(all_text_list, all_mm_label_list, all_trainable_flag_list):
            try:
                mret = self.process_multimodal_chunk(text, mm_label, trainable)
                ret = ret.concatenate(mret)
            except ValueError as e:
                tt = text[:24].replace('\n','<LF>')
                print(f"Process {tt if mm_label == 'text' else text} {mm_label} chunk error: {str(e)}")
                return OmniProcessorOutput(index=index)

        if raw_only:
            ret.raw_text = self.tokenizer.decode(ret.input_ids, skip_special_tokens=False)
            return ret
        return ret

    @torch.no_grad()
    def __call__(self, example, parallel=128):
        if isinstance(example, Dict):
            pass 
        elif isinstance(example, str):
            return self.process_one(example)
        elif isinstance(example, List):  # batch推理 异步多线程处理
            with cf.ThreadPoolExecutor(min(parallel, len(example))) as executor:
                future_list = [executor.submit(self.process_one, di, idx) for idx, di in enumerate(example)]
                batch_data = [key.result() for key in cf.as_completed(future_list)]
            valid_num = sum([1 if x.input_ids is not None else 0 for x in batch_data])
            assert(valid_num == len(batch_data))  # 推理数据严格要求数量对齐
            batch_data = sorted(batch_data, key=lambda x: x.index)  # 保证顺序不变
            
            ret = OmniProcessorOutput()
            for i in range(len(batch_data)):
                ret = ret.concatenate(batch_data[i])
            self.tokenizer.padding_side = "left"
            max_len = min(max([len(x.input_ids) for x in batch_data]),self.tokenizer.model_max_length)
            padding_result = self.tokenizer.pad({"input_ids": [r.input_ids for r in batch_data]}, return_tensors='pt')
            ret.input_ids = padding_result["input_ids"]
            ret.attention_mask = padding_result["attention_mask"]  # batch推理不pack 不需要seqlens
            
            if ret.audios is not None:
                max_audios_len = max([x.shape[-1] for x in ret.audios])
                ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios])
            
                ret.encoder_length = default_collate(ret.encoder_length)
                ret.bridge_length = default_collate(ret.bridge_length)
            
            if ret.images is not None:
                ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.images]
                ret.patch_nums = default_collate(ret.patch_nums)
                
            if ret.videos is not None:
                ret.videos = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.videos]
                ret.videos_patch_nums = default_collate(ret.videos_patch_nums)

            return ret

        else:
            raise ValueError("example format supported yet")
