

# 集成模型的设置
ensemble_cfg = {
    'cfg_list': [
        'cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml',
        'cfgs/nuscenes_models/transfusion_lidar.yaml',
        'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml',
        'cfgs/nuscenes_models/cbgs_voxel0075_largekernel3d_centerpoint.yaml',
    ],
    'ckpt_list': [
        'ckpts/nuscenes_ckpts/voxelnext_nuscenes_kernel1.pth',
        'ckpts/nuscenes_ckpts/cbgs_transfusion_lidar.pth',
        'ckpts/nuscenes_ckpts/cbgs_voxel0075_centerpoint_nds_6648.pth',
        'ckpts/nuscenes_ckpts/largekernel3D_centerpoint_openpcdet.pth'
    ],
    'MAX_MERGE_NUMS': 5,    # normal is less than 5 while ensemble the model nums is 3 or 4
    'BOX_SIZE': 7,
    'BOX_WEIGHT': [1,1,1,1,1,1,1],
}


if __name__ == '__main__':
    my_cfg = ensemble_cfg
    print(my_cfg['cfg_list'], my_cfg['ckpt_list'])