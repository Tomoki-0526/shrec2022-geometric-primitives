import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def parse_object(idx, pcd_root, gt_root, output_root):
    print("Parsing: pointCloud{}".format(idx))
    pcd_path = os.path.join(pcd_root, 'pointCloud{}.txt'.format(idx))
    gt_path = os.path.join(gt_root, 'GTpointCloud{}.txt'.format(idx))
    save_path = os.path.join(output_root, 'pointCloud{}'.format(idx))
    os.makedirs(save_path, exist_ok=True)

    object_coords = []
    object_colors = []
    object_semantic_gt = []
    object_instance_gt = []

    pcd = np.loadtxt(pcd_path, delimiter=',')
    gt = np.loadtxt(gt_path)
    coords = pcd
    colors = np.zeros_like(coords)
    class_name = gt[0]
    semantic_gt = np.repeat(class_name, coords.shape[0])
    semantic_gt = semantic_gt.reshape(-1, 1)
    instance_gt = np.repeat(0, coords.shape[0])
    instance_gt = instance_gt.reshape(-1, 1)

    object_coords.append(coords)
    object_colors.append(colors)
    object_semantic_gt.append(semantic_gt)
    object_instance_gt.append(instance_gt)

    object_coords = np.ascontiguousarray(np.vstack(object_coords))
    object_colors = np.ascontiguousarray(np.vstack(object_colors))
    object_semantic_gt = np.ascontiguousarray(np.vstack(object_semantic_gt))
    object_instance_gt = np.ascontiguousarray(np.vstack(object_instance_gt))

    np.save(os.path.join(save_path, 'coords.npy'), object_coords.astype(np.float32))
    np.save(os.path.join(save_path, 'colors.npy'), object_colors.astype(np.uint8))
    np.save(os.path.join(save_path, 'segment.npy'), object_semantic_gt.astype(np.int16))
    np.save(os.path.join(save_path, 'instance.npy'), object_instance_gt.astype(np.int16))


def main_process():
    pcd_root = '/home/szj/SHREC2022/dataset/test/pointCloud'
    gt_root = '/home/szj/SHREC2022/dataset/test/GTpointCloud'
    output_root = '/home/szj/SHREC2022/ptv3_dataset/test'

    pcd_files = sorted(os.listdir(pcd_root))
    idx_list = [int(f.split('.')[0].split('pointCloud')[1]) for f in pcd_files]

    pool = ProcessPoolExecutor(max_workers=8)
    _ = list(
        pool.map(
            parse_object,
            idx_list,
            repeat(pcd_root),
            repeat(gt_root),
            repeat(output_root),
        )
    )


if __name__ == '__main__':
    main_process()
