# Copyright 2023 Baichuan Inc. All Rights Reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import WhisperConfig
from transformers import CLIPVisionConfig

logger = logging.get_logger(__name__)


class OmniConfig(PretrainedConfig):
    model_type = "omni"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=125696,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        sparse_attention_heads=None,
        sparse_attention_layers=[],
        head_dim=None,
        attention_qkv_pack=True,
        attention_qkv_bias=False,
        use_norm_head=True,
        hidden_act="silu",
        max_position_embeddings=4096,
        position_embedding_type="rope",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        audio_config=None,
        visual_config=None,
        video_config=None,
        vocoder_config=None,
        flow_matching_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or self.num_attention_heads
        self.sparse_attention_heads = sparse_attention_heads
        self.sparse_attention_layers = sparse_attention_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.attention_qkv_pack = attention_qkv_pack
        self.attention_qkv_bias = attention_qkv_bias
        self.use_norm_head = use_norm_head
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        assert self.position_embedding_type.lower() in ("rope", "alibi")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if audio_config is not None:
            self.audio_config = WhisperConfig(**audio_config)
            if self.audio_config.vq_config is not None:
                self.audio_config.vq_config = PretrainedConfig(**self.audio_config.vq_config)
        if vocoder_config is not None:
            self.vocoder_config = WhisperConfig(**vocoder_config)
        if flow_matching_config is not None:
            self.flow_matching_config = PretrainedConfig(**flow_matching_config)
            self.flow_matching_config.cfm_params = PretrainedConfig(**self.flow_matching_config.cfm_params)
        if visual_config is not None:
            self.visual_config = CLIPVisionConfig(**visual_config)
        if video_config is not None:
            self.video_config = CLIPVisionConfig(**video_config)


    def to_diff_dict(self):
        data = super().to_diff_dict()
        data["model_type"] = self.model_type
        return data

    def get_rotary_base(self):
        if hasattr(self, "rotary_emb_base"):
            return self.rotary_emb_base
        else:
            return self.rope_theta

if __name__ == '__main__':  
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("./", trust_remote_code=True)
    print(config)