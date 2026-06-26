ModelParams = dict(
    dataset_type='n3dv',
    sh_degree=1
)

OptimizationParams = dict(
    dataloader=True,
    batch_size=2,
    coarse_iterations=1200,
    refine_iterations=9000,
    densify_until_iter=25000,
    opacity_threshold_coarse=0.05,
    static_densification_interval=200,
    densify_grad_threshold_dynamic=0.0001,
    mlp_deform_lr_init=0.005,
    mlp_deform_lr_final=0.00005,
    neighbor_reg=1.0,
    temporal_reg=1.0,  # 1.0
    stage=[2000, 5000, 6000, 9000],
    num_gaussian=200000,
    num_gaussian2=80000
)