from transformers import PretrainedConfig


class DashengAudioGenConfig(PretrainedConfig):
    model_type = "dasheng_audiogen"

    def __init__(
        self,
        text_encoder_name: str = "google/flan-t5-large",
        tokenizer_name: str = "mispeech/dashengtokenizer",
        use_zero_instruction: bool = False,
        instruction_seq_len: int = 1,
        task_instruction_dim: int = 1024,
        sample_rate: int = 16000,
        downsampling_ratio: int = 640,
        latent_dim: int = 1280,
        content_dim: int = 1024,
        frame_resolution: float = 0.005,
        duration_offset: float = 1.0,
        tokenizer_max_length: int = 512,
        dit_img_size: int = 1000,
        dit_patch_size: int = 1,
        dit_in_chans: int = 1280,
        dit_out_chans: int = 1280,
        dit_input_type: str = "1d",
        dit_embed_dim: int = 1536,
        dit_depth: int = 32,
        dit_num_heads: int = 24,
        dit_mlp_ratio: float = 4.0,
        dit_qk_norm: str = "layernorm",
        dit_norm_layer: str = "layernorm",
        dit_act_layer: str = "geglu",
        dit_context_norm: bool = True,
        dit_time_fusion: str = "ada",
        dit_ada_sola_rank: int = 32,
        dit_ada_sola_alpha: int = 32,
        dit_ta_context_dim: int = 1024,
        dit_ta_context_fusion: str = "add",
        dit_ta_context_norm: bool = True,
        dit_context_dim: int = 1024,
        dit_context_fusion: str = "cross",
        dit_context_pe_method: str = "none",
        dit_pe_method: str = "none",
        dit_rope_mode: str = "shared",
        adapter_num_heads: int = 16,
        adapter_dropout: float = 0.2,
        adapter_duration_grad_scale: float = 0.1,
        duration_predictor_filter_channels: int = 512,
        duration_predictor_n_layers: int = 5,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_p_dropout: float = 0.5,
        special_tokens: list = None,
        train_special_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder_name = text_encoder_name
        self.tokenizer_name = tokenizer_name
        self.use_zero_instruction = use_zero_instruction
        self.instruction_seq_len = instruction_seq_len
        self.task_instruction_dim = task_instruction_dim
        self.sample_rate = sample_rate
        self.downsampling_ratio = downsampling_ratio
        self.latent_dim = latent_dim
        self.content_dim = content_dim
        self.frame_resolution = frame_resolution
        self.duration_offset = duration_offset
        self.tokenizer_max_length = tokenizer_max_length
        self.dit_img_size = dit_img_size
        self.dit_patch_size = dit_patch_size
        self.dit_in_chans = dit_in_chans
        self.dit_out_chans = dit_out_chans
        self.dit_input_type = dit_input_type
        self.dit_embed_dim = dit_embed_dim
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_qk_norm = dit_qk_norm
        self.dit_norm_layer = dit_norm_layer
        self.dit_act_layer = dit_act_layer
        self.dit_context_norm = dit_context_norm
        self.dit_time_fusion = dit_time_fusion
        self.dit_ada_sola_rank = dit_ada_sola_rank
        self.dit_ada_sola_alpha = dit_ada_sola_alpha
        self.dit_ta_context_dim = dit_ta_context_dim
        self.dit_ta_context_fusion = dit_ta_context_fusion
        self.dit_ta_context_norm = dit_ta_context_norm
        self.dit_context_dim = dit_context_dim
        self.dit_context_fusion = dit_context_fusion
        self.dit_context_pe_method = dit_context_pe_method
        self.dit_pe_method = dit_pe_method
        self.dit_rope_mode = dit_rope_mode
        self.adapter_num_heads = adapter_num_heads
        self.adapter_dropout = adapter_dropout
        self.adapter_duration_grad_scale = adapter_duration_grad_scale
        self.duration_predictor_filter_channels = duration_predictor_filter_channels
        self.duration_predictor_n_layers = duration_predictor_n_layers
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_p_dropout = duration_predictor_p_dropout
        self.special_tokens = special_tokens or []
        self.train_special_tokens = train_special_tokens
