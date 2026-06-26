ModelParams = dict(
    dataset_type='widerange4d',
    feat_dim= 50,
    dy_threshold=50,
    num_times=240
)

OptimizationParams = dict(
    dataloader=True,
    batch_size=1,
    coarse_iterations=15000,
    refine_iterations=30000,
    densify_from_iter=500,
    densify_until_iter=10000,  # 15000
    static_densify_until_iter=8000,  # 10000
    position_lr_init=0.0,
    position_lr_final=0.0,
    temporal_feature_lr_init=0.0005,
    temporal_feature_lr_final=0.000005,
    mlp_deform_lr_init=0.005,  # anchor # 0.005
    mlp_deform_lr_final=0.0005,         # 0.0005
    mlp_cov_lr_init=0.004,  # feat # 0.004
    mlp_cov_lr_final=0.0004,       # 0.004
    mlp_opacity_lr_init=0.005,  # offset # 0.002
    mlp_opacity_lr_final=0.00005,        # 0.00002
    mlp_color_lr_init=0.005,  # scaling # 0.008
    mlp_color_lr_final=0.0005,          # 0.008
    opacity_threshold_coarse=0.005,
    static_densification_interval=300,
    dynamic_densification_interval=300,
    densify_grad_threshold_dynamic=0.0002,
    temporal_reg=1.0,  # 1.0
    step_flag1=7000,
    step_flag2=8000,
    coarse_gaussian_num=-1
)