ModelParams = dict(
    dataset_type='meeting',
    sh_degree=1
)

OptimizationParams = dict(
    dataloader=True,
    batch_size=2,
    coarse_iterations=1200,
    refine_iterations=7000,
    densify_until_iter=25000,
    opacity_threshold_coarse=0.05,
    stage=[1000, 3000, 4000, 5000],
    static_densification_interval=200,
    densify_grad_threshold_dynamic=0.0001,
    rotation_dynamic_offset_lr=0.05,
    rotation_offset_lr=0.001,
    neighbor_reg=1.0,
    temporal_reg=1.0,  # 1.0
    num_gaussian=80000,
    num_gaussian2=80000
)