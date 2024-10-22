#%%
from ventric_mesh import mesh_utils as mu

import numpy as np
import matplotlib.pyplot as plt

def create_lv(image_size, num_slices, ring_thickness, start_reduction, max_shift):
    lv = np.zeros((num_slices, image_size, image_size), dtype=bool)
    initial_outer_diameter = image_size - start_reduction

    for slice_idx in range(num_slices):
        # Linearly decrease the diameter of the rings
        outer_diameter = np.linspace(initial_outer_diameter, ring_thickness, num_slices)[slice_idx]
        inner_diameter = max(0, outer_diameter - ring_thickness)

        # Random shift for the center
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)

        # Create each slice with the shifted center
        lv[slice_idx] = create_ring(inner_diameter, outer_diameter, image_size, shift_x, shift_y)

    return lv

def create_ring(inner_diameter, outer_diameter, image_size, shift_x, shift_y):
    x, y = np.ogrid[:image_size, :image_size]
    center_x, center_y = image_size // 2, image_size // 2

    # Adjust center with the shift
    center_x += shift_x
    center_y += shift_y

    dist_from_center = (x - center_x)**2 + (y - center_y)**2
    inner_radius = inner_diameter / 2
    outer_radius = outer_diameter / 2

    mask = (dist_from_center >= inner_radius**2) & (dist_from_center <= outer_radius**2)
    return mask

#%% Example parameters
def test_meshing():
    image_size = 100
    num_slices = 5
    ring_thickness = 20
    start_reduction = 20
    max_shift = 5  # Maximum shift in pixels for the ring center

    mask = create_lv(image_size, num_slices, ring_thickness, start_reduction, max_shift)
    #
    resolution=0.1
    slice_thickness=1
    seed_num_base_epi=20
    seed_num_base_endo=15
    num_z_sections_epi=15
    num_z_sections_endo=10
    num_mid_layers_base=1

    points_cloud_epi,points_cloud_endo, apex_k_epi, apex_k_endo=mu.NodeGenerator(mask,resolution,slice_thickness,seed_num_base_epi,seed_num_base_endo,num_z_sections_epi,num_z_sections_endo)
    mesh, mesh_epi, mesh_endo, mesh_base=mu.VentricMesh_delaunay(points_cloud_epi,points_cloud_endo,num_mid_layers_base, apex_k_epi, apex_k_endo,save_flag=False,filename_suffix='test',result_folder='')
    aspect_ratios=mu.check_mesh_quality(mesh)
    num_large_aspect_ratio = sum(1 for ratio in aspect_ratios if ratio > 5)
    # check if more than 5% of the elements have very high aspect ratios 
    assert num_large_aspect_ratio<len(aspect_ratios)*0.05