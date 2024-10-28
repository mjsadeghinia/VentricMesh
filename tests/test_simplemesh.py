# %%
from ventric_mesh import mesh_utils as mu
import ventric_mesh.utils as utils


import numpy as np
import matplotlib.pyplot as plt


def create_lv(image_size, num_slices, ring_thickness, start_reduction, max_shift):
    lv = np.zeros((num_slices, image_size, image_size), dtype=bool)
    initial_outer_diameter = image_size - start_reduction

    for slice_idx in range(num_slices):
        # Linearly decrease the diameter of the rings
        outer_diameter = np.linspace(
            initial_outer_diameter, ring_thickness, num_slices
        )[slice_idx]
        inner_diameter = max(0, outer_diameter - ring_thickness)

        # Random shift for the center
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)

        # Create each slice with the shifted center
        lv[slice_idx] = create_ring(
            inner_diameter, outer_diameter, image_size, shift_x, shift_y
        )

    return lv


def create_ring(inner_diameter, outer_diameter, image_size, shift_x, shift_y):
    x, y = np.ogrid[:image_size, :image_size]
    center_x, center_y = image_size // 2, image_size // 2

    # Adjust center with the shift
    center_x += shift_x
    center_y += shift_y

    dist_from_center = (x - center_x) ** 2 + (y - center_y) ** 2
    inner_radius = inner_diameter / 2
    outer_radius = outer_diameter / 2

    mask = (dist_from_center >= inner_radius**2) & (dist_from_center <= outer_radius**2)
    return mask

def plot_voxels(voxel_array, resolution, slice_thickness, alpha = 1):
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    # fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')

    voxel_array = np.transpose(voxel_array, (1, 2, 0))

    nx, ny, nz = voxel_array.shape

    x = np.arange(0, nx * resolution, resolution)
    y = np.arange(0, ny * resolution, resolution)
    # z = np.arange(0, nz * slice_thickness, slice_thickness)
    z = np.linspace(0, nz * slice_thickness, nz, endpoint=False)
    # expanded_voxel_array = np.repeat(voxel_array, slice_thickness, axis=2)
    # Set the Z-axis tick positions and labels
    z_ticks = np.arange(0, nz, 1)  # Original tick positions
    z_tick_labels = [f"{val * slice_thickness:.1f}" for val in z_ticks]  # Scale and format labels
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    color = [1, 1 , 1, alpha]
    ax.voxels(voxel_array, facecolors=color, edgecolor='k')

    ax.set_xlabel('X Axis (mm)')
    ax.set_ylabel('Y Axis (mm)')
    ax.set_zlabel('Z Axis')
    return ax

# %% Example parameters
def test_meshing():
    image_size = 150
    num_slices = 10
    ring_thickness = 15
    start_reduction = 100
    max_shift = 0  # Maximum shift in pixels for the ring center
    resolution = 1
    slice_thickness = 5

    mask = create_lv(image_size, num_slices, ring_thickness, start_reduction, max_shift)

    # ax=plot_voxels(mask, resolution, slice_thickness)
    # ax.view_init(elev=-150, azim=-45)
    # ax.set_box_aspect(aspect=(1, 1, 1))
    # ax.set_xlim([30, 120])
    # ax.set_ylim([30, 120])
    # ax.set_axis_off()
    # fname = '3D_view.png'
    # plt.savefig(fname)
    
    seed_num_base_epi = 100
    seed_num_base_endo = 100
    num_z_sections_epi = 50
    num_z_sections_endo = 50
    num_mid_layers_base = 1

    mask_epi, mask_endo = mu.get_endo_epi(mask)
    coords_epi = mu.get_coords_from_mask(mask_epi, resolution, slice_thickness)
    coords_endo = mu.get_coords_from_mask(mask_endo, resolution, slice_thickness)
    (
        points_cloud_epi,
        points_cloud_endo,
        k_apex_epi,
        k_apex_endo,
        normals_list_epi,
        normals_list_endo,
    ) = mu.NodeGenerator(
        mask,
        resolution,
        slice_thickness,
        seed_num_base_epi,
        seed_num_base_endo,
        num_z_sections_epi,
        num_z_sections_endo,
        smooth_shax_epi=.02,
        smooth_shax_endo=.02,
        n_points_lax=64,
        smooth_lax_epi=.1,
        smooth_lax_endo=.1,
    )
    mesh_epi_filename, mesh_endo_filename, mesh_base_filename = mu.VentricMesh_poisson(
        points_cloud_epi,
        points_cloud_endo,
        num_mid_layers_base,
        SurfaceMeshSizeEpi=4,
        SurfaceMeshSizeEndo=4,
        normals_list_epi=normals_list_epi,
        normals_list_endo=normals_list_endo,
        save_flag=True,
        filename_suffix="",
        result_folder="",
    )
    output_mesh_filename = "Mesh_3D.msh"
    mu.generate_3d_mesh_from_seperate_stl(
        mesh_epi_filename,
        mesh_endo_filename,
        mesh_base_filename,
        output_mesh_filename,
        MeshSizeMin=3,
        MeshSizeMax=5,
    )
    errors_epi = utils.calculate_error_between_coords_and_mesh(
        coords_epi, mesh_epi_filename
    )
    errors_endo = utils.calculate_error_between_coords_and_mesh(
        coords_endo, mesh_endo_filename
    )
    
    all_errors = np.hstack((errors_endo, errors_epi)) 
    # fig = utils.plot_coords_and_mesh(coords_epi, coords_endo, mesh_epi_filename, mesh_endo_filename)
    # fname = "Mesh_vs_Coords.html"
    # fig.write_html(fname)
    num_large_error = (all_errors>5*resolution).sum()
    # check if more than 5% of the elements have very high aspect ratios
    assert num_large_error<0.01*len(all_errors)

# %%
