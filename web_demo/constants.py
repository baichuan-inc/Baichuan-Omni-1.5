MODEL_PATH = "../baichuan-omni/model"
COSY_VOCODER = "../third_party/cosy24k_vocoder"
g_cache_dir = "../cache"
sampling_rate = 24000
wave_concat_overlap = int(sampling_rate * 0.01)
role_prefix = {
    'system': '<B_SYS>',
    'user': '<C_Q>',
    'assistant': '<C_A>',
    'audiogen': '<audiotext_start_baichuan>'
}
max_frames = 8