from nuscenes.nuscenes import NuScenes

root_path = '../../data/nuscenes/v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
print(nusc)


def nusc_prcocess_scene(nusc):
    """
    遍历每个scene来统计sampel和annotation的数量
    """
    print('******* NuScenes Data Numbers *******')
    print('nusc.scene: {}, nusc.sample: {}, nusc.sample_data: {}, nusc.sample_annotation: {}'
          .format(len(nusc.scene), len(nusc.sample), len(nusc.sample_data), len(nusc.sample_annotation)))

    scene_nums = 0
    sample_nums = 0
    sample_annotation_nums = 0

    for scene in nusc.scene:
        scene_nums += 1

        # 遍历当前scene下的所有sample
        next_sample_token = scene['first_sample_token']
        while next_sample_token:
            # get current sample
            sample = nusc.get('sample', next_sample_token)

            sample_annotation_nums += len(sample['anns'])
            sample_nums += 1

            # get next sample token
            next_sample_token = sample['next']

    print('******* Count Data Compare *******')
    print('scene_nums: {}, sample_nums: {}, sample_annotation_nums: {}'.
          format(scene_nums, sample_nums, sample_annotation_nums))


def nusc_process_sample(nusc):
    """
        遍历每个sample来统计sampel和annotation的数量
    """
    print('******* NuScenes Data Numbers *******')
    print('nusc.scene: {}, nusc.sample: {}, nusc.sample_data: {}, nusc.sample_annotation: {}'
          .format(len(nusc.scene), len(nusc.sample), len(nusc.sample_data), len(nusc.sample_annotation)))

    sample_nums = 0
    sample_annotation_nums = 0

    scene_token_set = set()
    # lidar_token_set = set()
    for sample in nusc.sample:
        # count sample nums
        sample_nums += 1

        # count scene nums
        scene_token = sample['scene_token']
        scene_token_set.add(scene_token)

        # count current sample annotations nums
        sample_annotation_nums += len(sample['anns'])

    scene_nums = len(scene_token_set)

    print('******* Count Data Compare *******')
    print('scene_nums: {}, sample_nums: {}, sample_annotation_nums: {}'.
          format(scene_nums, sample_nums, sample_annotation_nums))


def nusc_process_sampledata(nusc):
    """
        遍历每个sample_data来统计sampel和annotation的数量
    """
    print('******* NuScenes Data Numbers *******')
    print('nusc.scene: {}, nusc.sample: {}, nusc.sample_data: {}, nusc.sample_annotation: {}'
          .format(len(nusc.scene), len(nusc.sample), len(nusc.sample_data), len(nusc.sample_annotation)))

    sample_data_nums = 0
    sample_annotation_nums = 0

    scene_token_set = set()
    sample_token_set = set()
    for sample_data in nusc.sample_data:
        sample_data_nums += 1

        sample_token = sample_data['sample_token']
        sample = nusc.get('sample', sample_token)
        sample_token_set.add(sample_token)          # repeat

        # sample_annotation_nums += len(sample['anns'])

        scene_token = sample['scene_token']
        scene = nusc.get('scene', scene_token)
        scene_token_set.add(scene_token)            # repeat

    for sample_token in sample_token_set:
        sample = nusc.get('sample', sample_token)
        sample_annotation_nums += len(sample['anns'])

    sample_nums = len(sample_token_set)
    scene_nums = len(scene_token_set)

    print('******* Count Data Compare *******')
    print('scene_nums: {}, sample_nums: {}, sample_data_nums: {}, sample_annotation_nums: {}'.
          format(scene_nums, sample_nums, sample_data_nums, sample_annotation_nums))


def nusc_count_sampledata(nusc):
    """
        遍历sampledata统计每个类别的数量
    """
    print('******* NuScenes Data Numbers *******')
    print('nusc.scene: {}, nusc.sample: {}, nusc.sample_data: {}, nusc.sample_annotation: {}'
          .format(len(nusc.scene), len(nusc.sample), len(nusc.sample_data), len(nusc.sample_annotation)))

    from collections import defaultdict
    countClass = defaultdict(int)
    countKeyframe = defaultdict(int)
    countChannel = defaultdict(int)
    countKeyframChannel = defaultdict(int)
    for sample_data in nusc.sample_data:
        sensor_cls = sample_data['sensor_modality']
        countClass[sensor_cls] += 1
        channel = sample_data['channel']
        countChannel[channel] += 1
        if sample_data['is_key_frame']:
            countKeyframe[sensor_cls] += 1
            countKeyframChannel[channel] += 1

    print('countDict: {}, \n, countKeyframe: {}'.format(countClass, countKeyframe))
    print('countChannel: {}'.format(countChannel))
    print('countKeyframChannel: {}'.format(countKeyframChannel))


def nusec_count_samplenums(nusc):
    lidar_data_nums, camera_data_nums, radar_data_nums = 0, 0, 0
    for scene in nusc.scene:
        sample = nusc.get('sample', scene['first_sample_token'])

        # get sensor data has next
        lidar_token = sample['data']['LIDAR_TOP']
        camera_token = sample['data']['CAM_FRONT']
        radar_token = sample['data']['RADAR_FRONT']

        # count sensor data
        while lidar_token:
            lidar = nusc.get('sample_data', lidar_token)
            lidar_data_nums += 1
            lidar_token = lidar['next']

        while camera_token:
            camera = nusc.get('sample_data', camera_token)
            camera_data_nums += 1
            camera_token = camera['next']

        while radar_token:
            radar = nusc.get('sample_data', radar_token)
            radar_data_nums += 1
            radar_token = radar['next']

    total_nums = lidar_data_nums + camera_data_nums + radar_data_nums
    print('lidar_data_nums: {}, camera_data_nums: {}, radar_data_nums: {}'.
          format(lidar_data_nums, camera_data_nums, radar_data_nums))
    print('Total numbers: {}'.format(total_nums))


def nusc_vis_sampledata(index=1):
    scene = nusc.scene[0]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']
    camera_token = sample['data']['CAM_FRONT']
    radar_token = sample['data']['RADAR_FRONT']

    nusc.render_sample(sample_token)
    nusc.render_sample_data(lidar_token)
    nusc.render_sample_data(camera_token)
    nusc.render_sample_data(radar_token)


if __name__ == '__main__':
    # nusc_prcocess_scene(nusc)
    # nusc_process_sample(nusc)
    # nusc_process_sampledata(nusc)
    # nusc_count_sampledata(nusc)
    # nusec_count_samplenums(nusc)

    nusc_vis_sampledata()







