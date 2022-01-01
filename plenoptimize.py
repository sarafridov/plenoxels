import os
import json
from argparse import ArgumentParser
from re import split
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import jax
np.random.seed(0)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

# Import jax only after setting the visible gpu
import jax
import jax.numpy as jnp
import plenoxel
from jax.ops import index, index_update, index_add
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
if __name__ != "__main__":
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.001'


flags = ArgumentParser()


flags.add_argument(
    "--data_dir", '-d',
    type=str,
    default='./nerf/data/nerf_synthetic/',
    help="Dataset directory e.g. nerf_synthetic/"
)
flags.add_argument(
    "--expname",
    type=str,
    default="experiment",
    help="Experiment name."
)
flags.add_argument(
    "--scene",
    type=str,
    default='lego',
    help="Name of the synthetic scene."
)
flags.add_argument(
    "--log_dir",
    type=str,
    default='jax_logs/',
    help="Directory to save outputs."
)
flags.add_argument(
    "--resolution",
    type=int,
    default=256,
    help="Grid size."
)
flags.add_argument(
    "--ini_rgb",
    type=float,
    default=0.0,
    help="Initial harmonics value in grid."
)
flags.add_argument(
    "--ini_sigma",
    type=float,
    default=0.1,
    help="Initial sigma value in grid."
)
flags.add_argument(
    "--radius",
    type=float,
    default=1.3,
    help="Grid radius. 1.3 works well on most scenes, but ship requires 1.5"
)
flags.add_argument(
    "--harmonic_degree",
    type=int,
    default=2,
    help="Degree of spherical harmonics. Supports 0, 1, 2, 3, 4."
)
flags.add_argument(
    '--num_epochs',
    type=int,
    default=1,
    help='Epochs to train for.'
)
flags.add_argument(
    '--render_interval',
    type=int,
    default=40,
    help='Render images during test/val step every x images.'
)
flags.add_argument(
    '--val_interval',
    type=int,
    default=2,
    help='Run test/val step every x epochs.'
)
flags.add_argument(
    '--lr_rgb',
    type=float,
    default=None,
    help='SGD step size for rgb. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--lr_sigma',
    type=float,
    default=None,
    help='SGD step size for sigma. Default chooses automatically based on resolution.'
    )
flags.add_argument(
    '--physical_batch_size',
    type=int,
    default=4000,
    help='Number of rays per batch, to avoid OOM.'
    )
flags.add_argument(
    '--logical_batch_size',
    type=int,
    default=4000,
    help='Number of rays per optimization batch. Must be a multiple of physical_batch_size.'
    )
flags.add_argument(
    '--jitter',
    type=float,
    default=0.0,
    help='Take samples that are jittered within each voxel, where values are computed with trilinear interpolation. Parameter controls the std dev of the jitter, as a fraction of voxel_len.'
)
flags.add_argument(
    '--uniform',
    type=float,
    default=0.5,
    help='Initialize sample locations to be uniformly spaced at this interval (as a fraction of voxel_len), rather than at voxel intersections (default if uniform=0).'
)
flags.add_argument(
    '--occupancy_penalty',
    type=float,
    default=0.0,
    help='Penalty in the loss term for occupancy; encourages a sparse grid.'
)
flags.add_argument(
    '--reload_epoch',
    type=int,
    default=None,
    help='Epoch at which to resume training from a saved model.'
)
flags.add_argument(
    '--save_interval',
    type=int,
    default=1,
    help='Save the grid checkpoints after every x epochs.'
)
flags.add_argument(
    '--prune_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when pruning should be done.'
)
flags.add_argument(
    '--prune_method',
    type=str,
    default='weight',
    help='Weight or sigma: prune based on contribution to training rays, or opacity.'
)
flags.add_argument(
    '--prune_threshold',
    type=float,
    default=0.001,
    help='Threshold for pruning voxels (either by weight or by sigma).'
)
flags.add_argument(
    '--split_epochs',
    type=int,
    nargs='+',
    default=[],
    help='List of epoch numbers when splitting should be done.'
)
flags.add_argument(
    '--interpolation',
    type=str,
    default='trilinear',
    help='Type of interpolation to use. Options are constant, trilinear, or tricubic.'
)
flags.add_argument(
    '--nv',
    action='store_true',
    help='Use the Neural Volumes rendering formula instead of the Max (NeRF) rendering formula.'
)

FLAGS = flags.parse_args()
data_dir = FLAGS.data_dir + FLAGS.scene
radius = FLAGS.radius


def get_data(root, stage):
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
        fpath = os.path.join(data_path, os.path.basename(frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt


if __name__ == "__main__":
    focal, train_c2w, train_gt = get_data(data_dir, "train")
    test_focal, test_c2w, test_gt = get_data(data_dir, "test")
    assert focal == test_focal
    H, W = train_gt[0].shape[:2]
    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)


log_dir = FLAGS.log_dir + FLAGS.expname
os.makedirs(log_dir, exist_ok=True)


automatic_lr = False
if FLAGS.lr_rgb is None or FLAGS.lr_sigma is None:
    automatic_lr = True
    FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
    FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)


if FLAGS.reload_epoch is not None:
    reload_dir = os.path.join(log_dir, f'epoch_{FLAGS.reload_epoch}')
    print(f'Reloading the grid from {reload_dir}')
    data_dict = plenoxel.load_grid(dirname=reload_dir, sh_dim = (FLAGS.harmonic_degree + 1)**2)
else:
    print(f'Initializing the grid')
    data_dict = plenoxel.initialize_grid(resolution=FLAGS.resolution, ini_rgb=FLAGS.ini_rgb, ini_sigma=FLAGS.ini_sigma, harmonic_degree=FLAGS.harmonic_degree)


# low-pass filter the ground truth image so the effective resolution matches twice that of the grid
def lowpass(gt, resolution):
    if gt.ndim > 3:
        print(f'lowpass called on image with more than 3 dimensions; did you mean to use multi_lowpass?')
    H = gt.shape[0]
    W = gt.shape[1]
    im = Image.fromarray((np.squeeze(np.asarray(gt))*255).astype(np.uint8))
    im = im.resize(size=(resolution*2, resolution*2))
    im = im.resize(size=(H, W))
    return np.asarray(im) / 255.0


# low-pass filter a stack of images where the first dimension indexes over the images
def multi_lowpass(gt, resolution):
    if gt.ndim <= 3:
        print(f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[-3]
    W = gt.shape[-2]
    clean_gt = np.copy(gt)
    for i in range(len(gt)):
        im = Image.fromarray(np.squeeze(gt[i,...] * 255).astype(np.uint8))
        im = im.resize(size=(resolution*2, resolution*2))
        im = im.resize(size=(H, W))
        im = np.asarray(im) / 255.0
        clean_gt[i,...] = im
    return clean_gt


def get_loss(data_dict, c2w, gt, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rays = plenoxel.get_rays(H, W, focal, c2w)
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - lowpass(gt, resolution))**2)
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1]))
    return loss


def get_loss_rays(data_dict, rays, gt, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - gt)**2)
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1]))
    return loss


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5, indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def render_pose_rays(data_dict, c2w, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, batch_size, interpolation, nv):
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    rays_o = np.reshape(rays_o, [-1,3])
    rays_d = np.reshape(rays_d, [-1,3])
    rgbs = []
    disps = []
    for i in range(int(np.ceil(H*W/batch_size))):
        start = i*batch_size
        stop = min(H*W, (i+1)*batch_size)
        if jitter > 0:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation, nv))
        else:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, None, radius, harmonic_degree, jitter, uniform, interpolation, nv))
        rgbs.append(rgbi)
        disps.append(dispi)
    rgb = jnp.reshape(jnp.concatenate(rgbs, axis=0), (H, W, 3))
    disp = jnp.reshape(jnp.concatenate(disps, axis=0), (H, W))
    return rgb, disp, None, None


def run_test_step(i, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, key, name_appendage=''):
    print('Evaluating')
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    tpsnr = 0.0
    for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):
        rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation, FLAGS.nv)
        mse = jnp.mean((rgb - gt)**2)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr

        if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
            disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
            vis = jnp.concatenate((gt, rgb, disp3), axis=1)
            vis = np.asarray((vis * 255)).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
        del rgb, disp
    tpsnr /= n_test_imgs
    return tpsnr


def update_grid(old_grid, lr, grid_grad):
    return index_add(old_grid, index[...], -1 * lr * grid_grad)


def update_grids(old_grid, lrs, grid_grad):
    for i in range(len(old_grid)):
        old_grid[i] = index_add(old_grid[i], index[...], -1 * lrs[i] * grid_grad[i])
    return old_grid


if FLAGS.physical_batch_size is not None:
    print(f'precomputing all the training rays')
    # Precompute all the training rays and shuffle them
    rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)


print(f'generating random keys')
split_keys_partial = jax.vmap(jax.random.split, in_axes=0, out_axes=0)
split_keys = jax.vmap(split_keys_partial, in_axes=1, out_axes=1)
if FLAGS.physical_batch_size is None:
    keys = jax.vmap(jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0), in_axes=1, out_axes=1)(jnp.reshape(jnp.arange(800*800), (800,800)))
else: 
    keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(FLAGS.physical_batch_size))
render_keys = jax.vmap(jax.random.PRNGKey, in_axes=0, out_axes=0)(jnp.arange(800*800))
if FLAGS.jitter == 0:
    render_keys = None
    keys = None


def main():
    global rays_rgb, keys, render_keys, data_dict, FLAGS, radius, train_c2w, train_gt, test_c2w, test_gt, automatic_lr
    start_epoch = 0
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    if FLAGS.reload_epoch is not None:
        start_epoch = FLAGS.reload_epoch + 1
    if np.isin(FLAGS.reload_epoch, FLAGS.prune_epochs):
        data_dict = plenoxel.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
    if np.isin(FLAGS.reload_epoch, FLAGS.split_epochs):
        data_dict = plenoxel.split_grid(data_dict)
        FLAGS.resolution = FLAGS.resolution * 2
        if automatic_lr:
            FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
            FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)
    for i in range(start_epoch, FLAGS.num_epochs):
        # Shuffle data before each epoch
        if FLAGS.physical_batch_size is None:
            temp = list(zip(train_c2w, train_gt))
            np.random.shuffle(temp)
            train_c2w, train_gt = zip(*temp)
        else:
            assert FLAGS.logical_batch_size % FLAGS.physical_batch_size == 0
            # Shuffle rays over all training images
            rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

        print('epoch', i)
        indices, data = data_dict
        if FLAGS.physical_batch_size is None:
            occupancy_penalty = FLAGS.occupancy_penalty / len(train_c2w)
            for j, (c2w, gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=len(train_c2w)):
                if FLAGS.jitter > 0:
                    splitkeys = split_keys(keys)
                    keys = splitkeys[...,0,:]
                    subkeys = splitkeys[...,1,:]
                else:
                    subkeys = None
                mse, data_grad = jax.value_and_grad(lambda grid: get_loss((indices, grid), c2w, gt, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 
        else:
            occupancy_penalty = FLAGS.occupancy_penalty / (len(rays_rgb) // FLAGS.logical_batch_size)
            for k in tqdm(range(len(rays_rgb) // FLAGS.logical_batch_size)):
                logical_grad = None
                for j in range(FLAGS.logical_batch_size // FLAGS.physical_batch_size):
                    if FLAGS.jitter > 0:
                        splitkeys = split_keys_partial(keys)
                        keys = splitkeys[...,0,:]
                        subkeys = splitkeys[...,1,:]
                    else:
                        subkeys = None
                    effective_j = k*(FLAGS.logical_batch_size // FLAGS.physical_batch_size) + j
                    batch = rays_rgb[effective_j*FLAGS.physical_batch_size:(effective_j+1)*FLAGS.physical_batch_size] # [B, 2+1, 3*?]
                    batch_rays, target_s = (batch[:,0,:], batch[:,1,:]), batch[:,2,:]
                    mse, data_grad = jax.value_and_grad(lambda grid: get_loss_rays((indices, grid), batch_rays, target_s, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, subkeys, sh_dim, occupancy_penalty, FLAGS.interpolation, FLAGS.nv))(data) 
                    if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                        if logical_grad is None:
                            logical_grad = data_grad
                        else:
                            logical_grad = [a + b for a, b in zip(logical_grad, data_grad)]
                        del data_grad
                    del mse, batch, batch_rays, target_s, subkeys, effective_j
                lrs = [FLAGS.lr_rgb / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]*sh_dim + [FLAGS.lr_sigma / (FLAGS.logical_batch_size // FLAGS.physical_batch_size)]
                if FLAGS.logical_batch_size > FLAGS.physical_batch_size:
                    data = update_grids(data, lrs, logical_grad)
                    del logical_grad
                else:
                    data = update_grids(data, lrs, data_grad)
                    del data_grad, logical_grad
        data_dict = (indices, data)
        del indices, data
        if np.isin(i, FLAGS.prune_epochs):
            data_dict = plenoxel.prune_grid(data_dict, method=FLAGS.prune_method, threshold=FLAGS.prune_threshold, train_c2w=train_c2w, H=H, W=W, focal=focal, batch_size=FLAGS.physical_batch_size, resolution=FLAGS.resolution, key=render_keys, radius=FLAGS.radius, harmonic_degree=FLAGS.harmonic_degree, jitter=FLAGS.jitter, uniform=FLAGS.uniform, interpolation=FLAGS.interpolation)
        if np.isin(i, FLAGS.split_epochs):
            data_dict = plenoxel.split_grid(data_dict)
            FLAGS.lr_rgb = FLAGS.lr_rgb * 3
            FLAGS.lr_sigma = FLAGS.lr_sigma * 3
            FLAGS.resolution = FLAGS.resolution * 2
            if automatic_lr:
                FLAGS.lr_rgb = 150 * (FLAGS.resolution ** 1.75)
                FLAGS.lr_sigma = 51.5 * (FLAGS.resolution ** 2.37)
            if FLAGS.physical_batch_size is not None:
                # Recompute all the training rays at the new resolution and shuffle them
                rays = np.stack([get_rays_np(H, W, focal, p) for p in train_c2w[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
                rays_rgb = np.concatenate([rays, multi_lowpass(train_gt[:,None], FLAGS.resolution).astype(np.float32)], 1) # [N, ro+rd+rgb, H, W,   3]
                rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
                rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
                rays_rgb = rays_rgb.take(np.random.permutation(rays_rgb.shape[0]), axis=0)

        if i % FLAGS.save_interval == FLAGS.save_interval - 1 or i == FLAGS.num_epochs - 1:
            print(f'Saving checkpoint at epoch {i}')
            plenoxel.save_grid(data_dict, os.path.join(log_dir, f'epoch_{i}'))

        if i % FLAGS.val_interval == FLAGS.val_interval - 1 or i == FLAGS.num_epochs - 1:
            validation_psnr = run_test_step(i + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, render_keys)
            print(f'at epoch {i}, test psnr is {validation_psnr}')
        
    
if __name__ == "__main__":
    main()