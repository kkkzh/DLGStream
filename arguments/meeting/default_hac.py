ModelParams = dict(
    dataset_type='meeting',
    feat_dim= 50,
    load_memory=True
)

OptimizationParams = dict(
    dataloader=True,
    batch_size=1,
    coarse_iterations=4000,
    refine_iterations=15000,
    densify_from_iter=500,
    densify_until_iter=8000,
    static_densify_until_iter=6000,
    position_lr_init=0.0,
    position_lr_final=0.0,
    mlp_deform_lr_init=0.005,  # anchor # 0.005
    mlp_deform_lr_final=0.0005,         # 0.0005
    mlp_cov_lr_init=0.004,  # feat # 0.004
    mlp_cov_lr_final=0.0004,       # 0.004
    mlp_opacity_lr_init=0.005,  # offset # 0.002
    mlp_opacity_lr_final=0.00005,        # 0.00002
    mlp_color_lr_init=0.005,  # scaling # 0.008
    mlp_color_lr_final=0.0005,          # 0.008
    opacity_threshold_coarse=0.005,
    static_densification_interval=300,  # 300 for discussion 200 for trimming
    densify_grad_threshold_dynamic=0.0002,
    temporal_reg=1.0,  # 1.0
    step_flag1=1000,
    step_flag2=3000,
)