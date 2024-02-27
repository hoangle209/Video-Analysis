from dataclasses import dataclass

@dataclass
class MOTRv2Config:
    resume: str
    meta_arch:  str
    dataset_file:  str
    with_box_refine: bool
    sample_mode: str
    sample_interval: int
    sampler_lengths: int
    fp_ratio: int
    query_interaction_layer: str
    query_denoise: float
    score_threshold: float
    update_score_threshold:float
    miss_tolerance: int
    device: str
    random_drop: int
    hidden_dim: int
    backbone: str
    dim_feedforward: int
    position_embedding: str
    position_embedding_scale: float
    lr_backbone_names: str
    lr_backbone: float
    save_period: int
    two_stage: bool
    accurate_ratio: bool
    frozen_weights: object
    num_anchors: int
    enable_fpn: bool
    dilation:  bool
    num_feature_levels:  int
    enc_layers:  int
    dec_layers:  int
    dropout:  float
    nheads :  int
    num_queries:  int
    dec_n_points:  int
    enc_n_points:  int
    decoder_cross_self:  bool
    sigmoid_attn: bool
    crop: bool
    cj: bool
    extra_track_attn: bool
    max_size: int
    val_width: int
    filter_ignore: bool
    append_crowd: bool
    masks: bool
    aux_loss: bool
    set_cost_class: int
    set_cost_bbox: int
    set_cost_giou: int
    mask_loss_coef: int
    dice_loss_coef: int
    cls_loss_coef: int
    bbox_loss_coef: int
    giou_loss_coef: int
    focal_alpha: float
    remove_difficult: bool
    eval: bool
    vis: bool
    num_workers: int
    cache_mode: bool
    fp_ratio: float
    merger_dropout:float
    update_query_pos: bool
    sampler_steps: object
    sampler_lengths: list
    exp_name: str
    memory_bank_score_thresh: float
    memory_bank_len: int
    memory_bank_type: object
    memory_bank_with_self_attn: bool
    use_checkpoint: bool
    shard_id: int
    num_shards: int
    init_method: object
    opts: object
    output_dir : str
    
   