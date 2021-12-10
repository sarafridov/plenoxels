import jax
import jax.numpy as jnp
from jax import lax
from jax.ops import index, index_update
import numpy as np
import matplotlib.pyplot as plt
import os
import sh
from tqdm import tqdm


# Based on https://github.com/google-research/google-research/blob/d0a9b1dad5c760a9cfab2a7e5e487be00886803c/jaxnerf/nerf/model_utils.py#L166
def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd=True):
  """Volumetric Rendering Function.
  Args:
    rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    sigma: jnp.ndarray(float32), density, [batch_size, num_samples].
    z_vals: jnp.ndarray(float32), [batch_size, num_samples].
    dirs: jnp.ndarray(float32), [batch_size, 3].
    white_bkgd: bool.
  Returns:
    comp_rgb: jnp.ndarray(float32), [batch_size, 3].
    disp: jnp.ndarray(float32), [batch_size].
    acc: jnp.ndarray(float32), [batch_size].
    weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
  eps = 1e-10
  dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]
  dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)  # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
  # Note that we're quietly turning sigma from [..., 0] to [...].
  alpha = 1.0 - jnp.exp(-jax.nn.relu(sigma) * dists)  # What fraction of light gets stuck in each voxel
  accum_prod = jnp.concatenate([
      jnp.ones_like(alpha[Ellipsis, :1], alpha.dtype),
      jnp.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, axis=-1)
  ],
                               axis=-1)  # How much light is left as we enter each voxel
  weights = alpha * accum_prod  # The absolute amount of light that gets stuck in each voxel
  comp_rgb = (weights[Ellipsis, None] * jax.nn.sigmoid(rgb)).sum(axis=-2)  # Accumulated color over the samples, ignoring background
  depth = (weights * z_vals[Ellipsis, :-1]).sum(axis=-1) # Weighted average of depths by contribution to final color
  acc = weights.sum(axis=-1)  # Total amount of light absorbed along the ray
  # Equivalent to (but slightly more efficient and stable than):
  #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
  inv_eps = 1 / eps
  disp = acc / depth
  disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps) # disparity = inverse depth
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[Ellipsis, None])  # Including the white background in the final color
  return comp_rgb, disp, acc, weights


# The volumetric rendering formula from Neural Volumes: https://arxiv.org/abs/1906.07751
def nv_rendering(rgb, sigma, z_vals, dirs, white_bkgd=True):
  eps = 1e-10
  dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]
  dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)  # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
  # Note that we're quietly turning sigma from [..., 0] to [...].
  alpha = 1.0 - jnp.exp(-jax.nn.relu(sigma) * dists)  # What fraction of light gets stuck in each voxel
  tau = jnp.clip(jnp.cumsum(alpha, axis=-1), a_min=0, a_max=1)
  weights = tau[Ellipsis, 1:] - tau[Ellipsis, :-1]
  # jnp.concatenate(tau[Ellipsis, 1:], jnp.ones_like(tau[Ellipsis, :1])) # The absolute amount of light that gets stuck in each voxel
  comp_rgb = (weights[Ellipsis, None] * jax.nn.sigmoid(rgb[:,:-1,:])).sum(axis=-2)  # Accumulated color over the samples, ignoring background
  depth = (weights * z_vals[Ellipsis, :-2]).sum(axis=-1) # Weighted average of depths by contribution to final color
  acc = weights.sum(axis=-1)  # Total amount of light absorbed along the ray
  # Equivalent to (but slightly more efficient and stable than):
  #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
  inv_eps = 1 / eps
  disp = acc / depth
  disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps) # disparity = inverse depth
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[Ellipsis, None])  # Including the white background in the final color
  return comp_rgb, disp, acc, weights


eps = 1e-5

def near_zero(vector):
  return jnp.abs(vector) < eps


def safe_floor(vector):
  return jnp.floor(vector + eps)


def safe_ceil(vector):
  return jnp.ceil(vector - eps)


@jax.partial(jax.jit, static_argnums=(2,3,4,5,8))
def intersection_distances(inputs, data_dict, resolution, radius, jitter, uniform, key, sh_dim, interpolation, matrix, powers):
  start, stop, offset, interval = inputs["start"], inputs["stop"], inputs["offset"], inputs["interval"]
  if uniform == 0:
    # For a single ray, compute all the possible voxel intersections up to the upper bound number, starting when the ray hits the cube
    upper_bound = int(1 + resolution) # per dimension upper bound on the number of voxel intersections
    intersections0 = jnp.linspace(start=start[0] + offset[0], stop=start[0] + offset[0] + interval[0] * upper_bound, num=upper_bound, endpoint=False)
    intersections1 = jnp.linspace(start=start[1] + offset[1], stop=start[1] + offset[1] + interval[1] * upper_bound, num=upper_bound, endpoint=False)
    intersections2 = jnp.linspace(start=start[2] + offset[2], stop=start[2] + offset[2] + interval[2] * upper_bound, num=upper_bound, endpoint=False)
    intersections = jnp.concatenate([intersections0, intersections1, intersections2], axis=None)
    intersections = jnp.sort(intersections) # TODO: replace this with just a merge of the three intersection arrays
  else:
    voxel_len = radius * 2.0 / resolution
    realstart = jnp.min(start)
    count = int(resolution*3 / uniform)
    intersections = jnp.linspace(start=realstart + uniform*voxel_len, stop=realstart + uniform*voxel_len*(count+1), num=count, endpoint=False)
  intersections = jnp.where(intersections <= stop, intersections, stop)
  # Get the values at these intersection points
  ray_o, ray_d = inputs["ray_o"], inputs["ray_d"]
  voxel_sh, voxel_sigma, intersections = values_oneray(intersections, data_dict, ray_o, ray_d, resolution, key, sh_dim, radius, jitter, 1e-5, interpolation, matrix, powers)
  return voxel_sh, voxel_sigma, intersections


get_intersections_partial = jax.vmap(fun=intersection_distances, in_axes=({"start": 0, "stop": 0, "offset": 0, "interval": 0, "ray_o": 0, "ray_d": 0}, None, None, None, None, None, 0, None, None, None, None), out_axes=0)
get_intersections = jax.vmap(fun=get_intersections_partial, in_axes=({"start": 1, "stop": 1, "offset": 1, "interval": 1, "ray_o": 1, "ray_d": 1}, None, None, None, None, None, 1, None, None, None, None), out_axes=1)


@jax.partial(jax.jit, static_argnums=(3,4))
def voxel_ids_oneray(intersections, ray_o, ray_d, voxel_len, resolution, eps=1e-5):
  # For a single ray, compute the ids of all the voxels it passes through
  # Compute the midpoint of the ray segment inside each voxel
  midpoints = (intersections[Ellipsis, 1:] + intersections[Ellipsis, :-1]) / 2.0
  midpoints = ray_o[jnp.newaxis, :] + midpoints[:, jnp.newaxis] * ray_d[jnp.newaxis, :]
  ids = jnp.array(jnp.floor(midpoints / voxel_len + eps) + resolution / 2, dtype=int)
  return ids


voxel_ids_partial = jax.jit(jax.vmap(fun=voxel_ids_oneray, in_axes=(0, 0, 0, None, None), out_axes=0))
voxel_ids = jax.jit(jax.vmap(fun=voxel_ids_partial, in_axes=(1, 1, 1, None, None), out_axes=1))


def scalarize(i, j, k, resolution):
  return i*resolution*resolution + j*resolution + k


def vectorize(index, resolution):
  i = index // (resolution**2)
  j = (index - i*resolution*resolution) // resolution
  k = index - i*resolution*resolution - j*resolution
  return jnp.array([i, j, k])


# Remove voxels that are empty, where empty is determined by weight (contribution to training pixels) or sigma (opacity)
def prune_grid(grid, method, threshold, train_c2w, H, W, focal, batch_size, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation):
  # method can be 'weight' or 'sigma'
  # sigma: prune by opacity
  # weight: prune by contribution to the training rays
  indices, data = grid
  if method == 'sigma':
    keep_idx = jnp.argwhere(data[-1] >= threshold)  # [N_keep, 1]
  elif method == 'weight':
    print(f'rendering all the training views to accumulate weight')
    max_contribution = np.zeros((resolution, resolution, resolution))
    for c2w in tqdm(train_c2w):
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = np.reshape(rays_o, [-1,3])
        rays_d = np.reshape(rays_d, [-1,3])
        for i in range(int(np.ceil(H*W/batch_size))):
            start = i*batch_size
            stop = min(H*W, (i+1)*batch_size)
            if jitter > 0:
                _, _, _, weightsi, voxel_idsi = jax.lax.stop_gradient(render_rays(grid, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation))
            else:
                _, _, _, weightsi, voxel_idsi = jax.lax.stop_gradient(render_rays(grid, (rays_o[start:stop], rays_d[start:stop]), resolution, key, radius, harmonic_degree, jitter, uniform, interpolation))
            weightsi = np.asarray(weightsi)
            voxel_idsi = np.asarray(voxel_idsi[:,:-1,:])
            max_contribution[voxel_idsi[...,0], voxel_idsi[...,1], voxel_idsi[...,2]] = np.maximum(max_contribution[voxel_idsi[...,0], voxel_idsi[...,1], voxel_idsi[...,2]], weightsi)
    keep_idx = jnp.argwhere(max_contribution >= threshold)  # [N_keep, 3]
    keep_idx = indices[keep_idx[:,0], keep_idx[:,1], keep_idx[:,2]]  # [N_keep, 1]
    del max_contribution, weightsi, voxel_idsi
  keep_idx = jnp.squeeze(keep_idx)  # Indexes into the data
  # Also keep any neighbors of any voxels that are kept
  keep_idx = jax.vmap(lambda idx: data_index_to_scalar(idx, grid))(keep_idx)  # Map index into data to scalar spatial index
  keep_idx = jax.vmap(lambda idx: get_neighbors(idx, resolution))(keep_idx).flatten()  # Get neighbors of these spatial indices
  jnpindices = jnp.array(indices)
  keep_idx = jax.vmap(lambda idx: scalar_to_data_index(idx, jnpindices))(keep_idx)  # Map scalar spatial index to index into data
  # Filter the data
  keep_idx = jnp.unique(keep_idx)  # dedup and sort
  data = [d[keep_idx] for d in data] 
  sort_idx = jnp.argsort(indices[indices>=0])
  idx = jnp.argwhere(indices>=0)[sort_idx][keep_idx]  # [N_keep, 3]
  indices = jnp.ones((resolution, resolution, resolution), dtype=int) * -1
  indices = indices.at[idx[:,0], idx[:,1], idx[:,2]].set(jnp.arange(len(keep_idx), dtype=int))
  print(f'after pruning, the number of nonempty indices is {len(jnp.argwhere(indices >= 0))}')
  del idx, keep_idx, jnpindices
  return (indices, data)


# Map a position in the data array to the corresponding scalar spatial index
def data_index_to_scalar(idx, grid):
  indices, data = grid
  active_voxels = jnp.argwhere(indices>=0)  # [N_active_voxels, 3]
  assert len(data[-1]) == len(active_voxels)
  resolution = len(indices)
  return scalarize(active_voxels[idx,0], active_voxels[idx,1], active_voxels[idx,2], resolution)


# Map a scalar index idx to the corresponding position in the data array, or -1 if pruned
def scalar_to_data_index(idx, indices):
  resolution = len(indices)
  vector_idx = vectorize(idx, resolution)
  print(f'indices has type {type(indices)} and idx has type {type(idx)}')
  return indices[vector_idx[0], vector_idx[1], vector_idx[2]]


# Map an index in a grid to a set of 8 child indices in the split grid
def expand_index(idx, new_resolution):
  i000 = scalarize(idx[0]*2, idx[1]*2, idx[2]*2, new_resolution)
  i001 = scalarize(idx[0]*2, idx[1]*2, idx[2]*2 + 1, new_resolution)
  i010 = scalarize(idx[0]*2, idx[1]*2 + 1, idx[2]*2, new_resolution)
  i011 = scalarize(idx[0]*2, idx[1]*2 + 1, idx[2]*2 + 1, new_resolution)
  i100 = scalarize(idx[0]*2 + 1, idx[1]*2, idx[2]*2, new_resolution)
  i101 = scalarize(idx[0]*2 + 1, idx[1]*2, idx[2]*2 + 1, new_resolution)
  i110 = scalarize(idx[0]*2 + 1, idx[1]*2 + 1, idx[2]*2, new_resolution)
  i111 = scalarize(idx[0]*2 + 1, idx[1]*2 + 1, idx[2]*2 + 1, new_resolution)
  return jnp.array([i000, i001, i010, i011, i100, i101, i110, i111])


def map_neighbors(offset):
  # offset is a ternary 3-vector; return its index into the offsets array (of length 27, in expand_data)
  return (offset[0] + 1)*9 + (offset[1] + 1)*3 + offset[2] + 1


# Split each nonempty voxel in each dimension, using trilinear interpolation to initialize child voxels
def split_weights(childx, childy, childz):
  # childx, childy, childz are each -1 or 1 denoting the position of the child voxel within the parent (-1 instead of 0 for convenience)
  # all 27 neighbors of the parent are considered in the weights, but only 8 of the weights are nonzero for each child
  weights = jnp.zeros(27)
  # center of parent voxel is distance 1/4 from center of each child (nearest neighbor in all dimensions).
  weights = weights.at[13].set(0.75 * 0.75 * 0.75)
  # neighbors that are one away have 2 zeros and one nonzero. There should be 3 of these.
  weights = weights.at[map_neighbors([childx, 0, 0])].set(0.75 * 0.75 * 0.25)
  weights = weights.at[map_neighbors([0, childy, 0])].set(0.75 * 0.75 * 0.25)
  weights = weights.at[map_neighbors([0, 0, childz])].set(0.75 * 0.75 * 0.25)
  # neighbors that are 2 away have 1 zero and two nonzeros. There should be 3 of these.
  weights = weights.at[map_neighbors([childx, childy, 0])].set(0.75 * 0.25 * 0.25)
  weights = weights.at[map_neighbors([childx, 0, childz])].set(0.75 * 0.25 * 0.25)
  weights = weights.at[map_neighbors([0, childy, childz])].set(0.75 * 0.25 * 0.25)
  # one neighbor is 3 away and has all 3 nonzeros.
  weights = weights.at[map_neighbors([childx, childy, childz])].set(0.25 * 0.25 * 0.25)
  return weights


def expand_data(idx, grid):
  # idx is a vector index of the voxel to be split
  offsets = jnp.array([[-1,-1,-1], [-1,-1,0], [-1,-1,1], [-1,0,-1], [-1,0,0], [-1,0,1], [-1,1,-1], [-1,1,0], [-1,1,1],
                       [0,-1,-1], [0,-1,0], [0,-1,1], [0,0,-1], [0,0,0], [0,0,1], [0,1,-1], [0,1,0], [0,1,1],
                       [1,-1,-1], [1,-1,0], [1,-1,1], [1,0,-1], [1,0,0], [1,0,1], [1,1,-1], [1,1,0], [1,1,1]])  # [27, 3]
  neighbor_idx = idx[jnp.newaxis,:] + offsets  # [27, 3]
  neighbor_data = grid_lookup(neighbor_idx[:,0], neighbor_idx[:,1], neighbor_idx[:,2], grid)
  child_idx = jnp.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]])  # [8, 3]
  weights = jax.vmap(split_weights)(child_idx[:,0], child_idx[:,1], child_idx[:,2])  # [8, 27] first index is over the 8 child voxels, second index is over the neighbors for the parent, only 8 of which are relevant to each child
  expanded_data = [jnp.sum(weights[..., jnp.newaxis] * d, axis=1) for d in neighbor_data[:-1]]
  expanded_data.append(jnp.sum(weights * neighbor_data[-1], axis=1))
  del weights, offsets, neighbor_idx, neighbor_data, child_idx
  return expanded_data


# Map an index (scalarized) to itself and its 6 neighbors
def get_neighbors(idx, resolution):
  volid = vectorize(idx, resolution)
  front = scalarize(jnp.minimum(resolution-1, volid[0] + 1), volid[1], volid[2], resolution)
  back = scalarize(jnp.maximum(0, volid[0] - 1), volid[1], volid[2], resolution)
  top = scalarize(volid[0], jnp.minimum(resolution-1, volid[1] + 1), volid[2], resolution)
  bottom = scalarize(volid[0], jnp.maximum(0, volid[1] - 1), volid[2], resolution)
  right = scalarize(volid[0], volid[1], jnp.minimum(resolution-1, volid[2] + 1), resolution)
  left = scalarize(volid[0], volid[1], jnp.maximum(0, volid[2] - 1), resolution)
  return jnp.array([idx, front, back, top, bottom, right, left])


# Subdivide each voxel into 8 voxels, using trilinear interpolation and respecting sparsity
def split_grid(grid):
  indices, data = grid
  # Expand the indices, respecting sparsity
  new_resolution = len(indices) * 2
  big_indices = jnp.ones((new_resolution, new_resolution, new_resolution), dtype=int) * -1
  keep_idx = jnp.argwhere(indices >= 0)  # [N_keep, 3]
  # Expand the data, with trilinear interpolation
  big_data_partial = jax.vmap(expand_data, in_axes=(0, None))(keep_idx, grid)
  big_data = [d.reshape(len(data[-1])*8, 3) for d in big_data_partial[:-1]]
  big_data.append(big_data_partial[-1].reshape(len(data[-1])*8))
  del data
  big_keep_idx = jnp.ravel(jax.vmap(lambda index: expand_index(index, new_resolution), in_axes=0)(keep_idx))
  idx = vectorize(big_keep_idx, new_resolution)  # [3, N_keep*8]
  big_indices = big_indices.at[idx[0,:], idx[1,:], idx[2,:]].set(jnp.arange(len(big_keep_idx), dtype=int))
  del idx, big_keep_idx, keep_idx, indices
  print(f'after splitting, the number of nonempty indices is {len(jnp.argwhere(big_indices >= 0))}')
  return (big_indices, big_data)


def initialize_grid(resolution, ini_rgb=0.0, ini_sigma=0.1, harmonic_degree=0):
  sh_dim = (harmonic_degree + 1)**2
  data = []  # data is a list of length sh_dim + 1
  for _ in range(sh_dim):
    data.append(jnp.ones((resolution**3, 3), dtype=np.float32) * ini_rgb)
  data.append(jnp.ones((resolution**3), dtype=np.float32) * ini_sigma)
  indices = jnp.arange(resolution**3, dtype=int).reshape((resolution, resolution, resolution))
  return (indices, data)


def save_grid(grid, dirname):
  indices, data = grid
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  np.save(os.path.join(dirname, f'sigma_grid.npy'), data[-1])
  for i in range(len(data)-1):
    np.save(os.path.join(dirname, f'sh_grid{i}.npy'), data[i])
  np.save(os.path.join(dirname, f'indices.npy'), indices)


def load_grid(dirname, sh_dim):
  data = []
  for i in range(sh_dim):
    data.append(np.load(os.path.join(dirname, f'sh_grid{i}.npy')))
  data.append(np.load(os.path.join(dirname, f'sigma_grid.npy')))
  indices = np.load(os.path.join(dirname, f'indices.npy'))
  return (indices, data)


@jax.jit
def trilinear_interpolation_weight(xyzs):
  # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
  xs = xyzs[:,0]
  ys = xyzs[:,1]
  zs = xyzs[:,2]
  weight000 = (1-xs) * (1-ys) * (1-zs)  # [n_pts]
  weight001 = (1-xs) * (1-ys) * zs  # [n_pts]
  weight010 = (1-xs) * ys * (1-zs)  # [n_pts]
  weight011 = (1-xs) * ys * zs  # [n_pts]
  weight100 = xs * (1-ys) * (1-zs)  # [n_pts]
  weight101 = xs * (1-ys) * zs  # [n_pts]
  weight110 = xs * ys * (1-zs)  # [n_pts]
  weight111 = xs * ys * zs  # [n_pts]
  weights =  jnp.stack([weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111], axis=-1) # [n_pts, 8]
  return weights


def apply_power(power, xyzs):
  return xyzs[:,0]**power[0] * xyzs[:,1]**power[1] * xyzs[:,2]**power[2]


@jax.jit
def tricubic_interpolation(xyzs, corner_pts, grid, matrix, powers):
  # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
  # corner_pts should have shape [n_pts, 3] and denote the grid coordinates of the 000 interpolation point
  # matrix should be [64, 64] output of tricubic_interpolation_matrix
  # powers should be [64, 3] and contain all combinations of the powers 0 through 3 in three dimensions
  neighbor_data = jax.vmap(lambda pts: tricubic_neighbors(pts, grid))(corner_pts)  # list where each entry has shape [n_pts, 64, ...] 
  coeffs = [jnp.clip(jax.vmap(lambda d: jnp.matmul(matrix, d))(d), a_min=-1e7, a_max=1e7) for d in neighbor_data]  # list where each entry has shape [n_pts, 64, ...]
  things_to_multiply_by_coeffs = jnp.clip(jax.vmap(lambda power: apply_power(power, xyzs), out_axes=-1)(powers), a_min=-1e7, a_max=1e7)  # [n_pts, 64]
  result = [jnp.sum(coeff * things_to_multiply_by_coeffs[..., jnp.newaxis], axis=1) for coeff in coeffs[:-1]]  # list where each entry has shape [n_pts, ...]
  result.append(jnp.sum(coeffs[-1] * things_to_multiply_by_coeffs, axis=1))
  return result


@jax.jit
# Get the data at the 64 neighboring voxels needed for tricubic interpolation
def tricubic_neighbors(idx, grid):
  # idx is a vector index of the voxel to be interpolated
  offsets = []
  for i in range(4):
    for j in range(4):
      for k in range(4):
        offsets.append([i-1,j-1,k-1])
  offsets = jnp.array(offsets)  # [64, 3]
  neighbor_idx = idx[jnp.newaxis, :] + offsets  # [64, 3]
  resolution = len(grid[0])
  neighbor_idx = jnp.clip(neighbor_idx, a_min=0, a_max=resolution-1)
  neighbor_data = jax.vmap(lambda neighbor: grid_lookup(neighbor[0], neighbor[1], neighbor[2], grid))(neighbor_idx)
  return neighbor_data


# Generate the 64 by 64 weight matrix that maps grid values to polynomial coefficients
@jax.jit
def tricubic_interpolation_matrix():
  # Set up the indices
  powers = []
  for i in range(4):
    for j in range(4):
      for k in range(4):
        powers.append([i,j,k])
  powers = np.asarray(powers)  # [64, 3]  all combinations of the powers 0 through 3 in three dimensions
  coords = powers - 1  # [64, 3]  relative coordinates of neighboring voxels
  # Set up the weight matrix
  matrix = np.zeros((64,64))
  for i in range(64):
    for j in range(64):
      x = coords[i, 0]
      y = coords[i, 1]
      z = coords[i, 2]
      matrix[i,j] = x**powers[j, 0] * y**powers[j, 1] * z**powers[j, 2]
  # Invert the weight matrix
  inverted_matrix = np.linalg.inv(matrix)
  return jnp.array(inverted_matrix), powers


@jax.jit
def grid_lookup(x, y, z, grid):
  indices, data = grid
  ret = [jnp.where(indices[x,y,z,jnp.newaxis]>=0, d[indices[x,y,z]], jnp.zeros(3)) for d in data[:-1]]
  ret.append(jnp.where(indices[x,y,z]>=0, data[-1][indices[x,y,z]], 0))
  return ret


@jax.partial(jax.jit, static_argnums=(4,7,8,10))
def values_oneray(intersections, grid, ray_o, ray_d, resolution, key, sh_dim, radius, jitter, eps, interpolation, matrix, powers):
  voxel_len = radius * 2.0 / resolution
  if not jitter:
    pts = ray_o[jnp.newaxis, :] + intersections[:, jnp.newaxis] * ray_d[jnp.newaxis, :]  # [n_intersections, 3]
    pts = pts[:, jnp.newaxis, :]  # [n_intersections, 1, 3]
    offsets = jnp.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]) * voxel_len / 2.0  # [8, 3]
    neighbors = jnp.clip(pts + offsets[jnp.newaxis, :, :], a_min=-radius, a_max=radius)  # [n_intersections, 8, 3]
    neighbor_centers = jnp.clip((jnp.floor(neighbors / voxel_len + eps) + 0.5) * voxel_len, a_min=-(radius - voxel_len/2), a_max=radius - voxel_len/2)  # [n_intersections, 8, 3]
    neighbor_ids = jnp.array(jnp.floor(neighbor_centers / voxel_len + eps) + resolution / 2, dtype=int)  # [n_intersections, 8, 3]
    neighbor_ids = jnp.clip(neighbor_ids, a_min=0, a_max=resolution-1)
    xyzs = (pts[:,0,:] - neighbor_centers[:,0,:]) / voxel_len
    if interpolation == 'tricubic':
      pt_data = tricubic_interpolation(xyzs, neighbor_ids[:,0,:], grid, matrix, powers)
      pt_sigma = pt_data[-1][:-1]
      pt_sh = [d[:-1,:] for d in pt_data[:-1]]
    elif interpolation == 'trilinear':
      weights = trilinear_interpolation_weight(xyzs)  # [n_intersections, 8]
      neighbor_data = grid_lookup(neighbor_ids[...,0], neighbor_ids[...,1], neighbor_ids[...,2], grid)
      neighbor_sh = neighbor_data[:-1]
      neighbor_sigma = neighbor_data[-1]
      pt_sigma = jnp.sum(weights * neighbor_sigma, axis=1)[:-1]
      pt_sh = [jnp.sum(weights[..., jnp.newaxis] * nsh, axis=1)[:-1,:] for nsh in neighbor_sh]
    elif interpolation == 'constant':
      voxel_ids = neighbor_ids[:,0,:]
      voxel_data = jax.vmap(lambda voxel_id: grid_lookup(voxel_id[0], voxel_id[1], voxel_id[2], grid))(voxel_ids)
      pt_sigma = voxel_data[-1][:-1]
      pt_sh = [d[:-1,:] for d in voxel_data[:-1]]
    else:
      print(f'Unrecognized interpolation method {interpolation}.')
      assert False
    return pt_sh, pt_sigma, intersections
  else: # Only does trilinear with jitter
    jitters = jax.random.normal(key=key, shape=(intersections.shape[0],)) * voxel_len * jitter
    jittered_intersections = jnp.clip(intersections + jitters, a_min=intersections[0], a_max=intersections[-1])
    jittered_pts = ray_o[jnp.newaxis, :] + jittered_intersections[:, jnp.newaxis] * ray_d[jnp.newaxis, :]  # [n_intersections, 3]
    jittered_pts = jittered_pts[:, jnp.newaxis, :]  # [n_intersections, 1, 3]
    offsets = jnp.array([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]) * voxel_len / 2.0  # [8, 3]
    neighbors = jnp.clip(jittered_pts + offsets[jnp.newaxis, :, :], a_min=-radius, a_max=radius)  # [n_intersections, 8, 3]
    neighbor_centers = jnp.clip((jnp.floor(neighbors / voxel_len + eps) + 0.5) * voxel_len, a_min=-(radius - voxel_len/2), a_max=radius - voxel_len/2)  # [n_intersections, 8, 3]
    neighbor_ids = jnp.array(jnp.floor(neighbor_centers / voxel_len + eps) + resolution / 2, dtype=int)  # [n_intersections, 8, 3]
    neighbor_ids = jnp.clip(neighbor_ids, a_min=0, a_max=resolution-1)
    xyzs = (jittered_pts[:,0,:] - neighbor_centers[:,0,:]) / voxel_len
    weights = trilinear_interpolation_weight(xyzs)  # [n_intersections, 8]
    neighbor_data = grid_lookup(neighbor_ids[...,0], neighbor_ids[...,1], neighbor_ids[...,2], grid)
    neighbor_sh = neighbor_data[:-1]
    neighbor_sigma = neighbor_data[-1]
    pt_sigma = jnp.sum(weights * neighbor_sigma, axis=1)[:-1]
    pt_sh = [jnp.sum(weights[..., jnp.newaxis] * nsh, axis=1)[:-1,:] for nsh in neighbor_sh]
    idx = jnp.argsort(jittered_intersections)  # Should be nearly sorted already
    return [sh[idx][:-1] for sh in pt_sh], pt_sigma[idx][:-1], jittered_intersections[idx]


@jax.partial(jax.jit, static_argnums=(2,4,5,6,7,8,9))
def render_rays(grid, rays, resolution, keys, radius=1.3, harmonic_degree=0, jitter=0, uniform=0, interpolation='trilinear', nv=False):
  sh_dim = (harmonic_degree + 1)**2
  voxel_len = radius * 2.0 / resolution
  assert (resolution // 2) * 2 == resolution # Renderer assumes resolution is a multiple of 2
  rays_o, rays_d = rays
  # Compute when the rays enter and leave the grid
  offsets_pos = jax.lax.stop_gradient((radius - rays_o) / rays_d)
  offsets_neg = jax.lax.stop_gradient((-radius - rays_o) / rays_d)
  offsets_in = jax.lax.stop_gradient(jnp.minimum(offsets_pos, offsets_neg))
  offsets_out = jax.lax.stop_gradient(jnp.maximum(offsets_pos, offsets_neg))
  start = jax.lax.stop_gradient(jnp.max(offsets_in, axis=-1, keepdims=True))
  stop = jax.lax.stop_gradient(jnp.min(offsets_out, axis=-1, keepdims=True))
  first_intersection = jax.lax.stop_gradient(rays_o + start * rays_d)
  # Compute locations of ray-voxel intersections along each dimension
  interval = jax.lax.stop_gradient(voxel_len / jnp.abs(rays_d))
  offset_bigger = jax.lax.stop_gradient((safe_ceil(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
  offset_smaller = jax.lax.stop_gradient((safe_floor(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
  offset = jax.lax.stop_gradient(jnp.maximum(offset_bigger, offset_smaller))
  # Compute the samples along each ray
  matrix = None
  powers = None
  if interpolation == 'tricubic':
    matrix, powers = tricubic_interpolation_matrix()
  if len(rays_o.shape) > 2:
    voxel_sh, voxel_sigma, intersections = get_intersections({"start": start, "stop": stop, "offset": offset, "interval": interval, "ray_o": rays_o, "ray_d": rays_d}, grid, resolution, radius, jitter, uniform, keys, sh_dim, interpolation, matrix, powers)
  else:
    voxel_sh, voxel_sigma, intersections = get_intersections_partial({"start": start, "stop": stop, "offset": offset, "interval": interval, "ray_o": rays_o, "ray_d": rays_d}, grid, resolution, radius, jitter, uniform, keys, sh_dim, interpolation, matrix, powers)
  # Apply spherical harmonics
  voxel_rgb = sh.eval_sh(harmonic_degree, voxel_sh, rays_d)
  # Call volumetric_rendering
  if nv:
    rgb, disp, acc, weights = nv_rendering(voxel_rgb, voxel_sigma, intersections, rays_d)
  else:
    rgb, disp, acc, weights = volumetric_rendering(voxel_rgb, voxel_sigma, intersections, rays_d)
  pts = rays_o[:, jnp.newaxis, :] + intersections[:, :, jnp.newaxis] * rays_d[:, jnp.newaxis, :]  # [n_rays, n_intersections, 3]
  ids = jnp.clip(jnp.array(jnp.floor(pts / voxel_len + eps) + resolution / 2, dtype=int), a_min=0, a_max=resolution-1)
  return rgb, disp, acc, weights, ids


def get_rays(H, W, focal, c2w):
    i, j = jnp.meshgrid(jnp.linspace(0, W-1, W) + 0.5, jnp.linspace(0, H-1, H) + 0.5) 
    dirs = jnp.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jnp.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jnp.sum(dirs[..., jnp.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = jnp.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d
