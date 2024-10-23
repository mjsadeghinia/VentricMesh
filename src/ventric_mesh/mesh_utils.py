# %%
import numpy as np
import open3d as o3d
import os

from scipy.spatial import Delaunay
from scipy.interpolate import splev
from scipy.interpolate import splprep
from scipy.ndimage import binary_dilation
from stl import mesh
import warnings
import gmsh
from tqdm import tqdm
from structlog import get_logger

from ventric_mesh.utils import *

logger = get_logger()


# %%
# ----------------------------------------------------------------
# -------------    Preproscessing of the Masks    ----------------
# ----------------------------------------------------------------
# Extracting of edges of epi and endo
# Here we create the edges of each stack if the edges are connected so we are on the last stacks where the there is only epicardium otherwise we have both epi and endo. In this case the samples with more element (lenght) is the outer diameter and thus epicardium
def get_endo_epi(mask):
    if len(mask.shape) > 3:
        logger.error("The mask should be a list (corresponding to longitudinal slices) containting binary images (corresponding to short axis images)")
    K, I, J = mask.shape
    kernel = np.ones((3, 3), np.uint8)
    mask_epi = np.zeros((K, I, J))
    mask_endo = np.zeros((K, I, J))

    for k in range(K):
        mask_k = mask[k, :, :]
        img = np.uint8(mask_k * 255)
        img_dilated = binary_dilation(img, structure=kernel).astype(img.dtype)
        img_edges = img_dilated - img
        img_edges[img_edges == 2] = 0
        flag, visited, visited_reversed = is_connected(img_edges)
        if flag:
            img_epi = img_edges
            img_endo = np.zeros((I, J))
        else:
            img_epi = np.zeros((I, J), dtype=np.uint8)
            img_endo = np.zeros((I, J), dtype=np.uint8)
            if len(visited) > len(visited_reversed):
                for x, y in visited:
                    img_epi[x, y] = 1
                for x, y in visited_reversed:
                    img_endo[x, y] = 1
            else:
                for x, y in visited_reversed:
                    img_epi[x, y] = 1
                for x, y in visited:
                    img_endo[x, y] = 1
        mask_epi[k, :, :] = img_epi
        mask_endo[k, :, :] = img_endo
    return mask_epi, mask_endo


# ----------------------------------------------------------------
# ------------ Creating BSplines and Smooth Contours -------------
# ----------------------------------------------------------------
def get_coords_from_mask(mask, resolution, slice_thickness):
    K, I, _ = mask.shape
    coords = []
    for k in range(K):
        img = mask[k, :, :]
        coords_k = coords_from_img(img, resolution)
        if len(coords_k) > 0:
            # Sort the coordinates
            coords_sorted = sorting_coords(coords_k, resolution)
            # Assign z-values based on slice index and slice thickness
            z = -(k) * slice_thickness
            z_list = np.ones(coords_sorted.shape[0]) * z
            # Combine x, y, z coordinates
            coords_k_with_z = np.column_stack((coords_sorted, z_list))
            coords.append(coords_k_with_z)
    return coords

def get_shax_from_coords(coords, smooth_level):
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="scipy.interpolate"
    )
    tck = []
    for coord_k in tqdm(coords, desc="Creating SHAX Curves", ncols=100):
        # Coordinates are already sorted and include z-values
        # Add the first point to the end to make it periodic
        coords_sorted = np.vstack((coord_k, coord_k[0, :]))
        # Calculate area for smoothing
        area = calculate_area_points(coords_sorted[:, :2])
        # Spline fitting
        tck_k, u = splprep(
            [coords_sorted[:, 0], coords_sorted[:, 1], coords_sorted[:, 2]],
            s=smooth_level * area,
            per=True,
            k=3,
        )
        tck.append(tck_k)
    return tck


# % Creating LAX BSplines from SHAX
def get_sample_points_from_shax(tck_shax, n_points):
    sample_points = []
    K = len(tck_shax)
    shax_points = get_points_from_tck(tck_shax, -1)
    # We find the center based on the SHAX of the last slice
    LV_center = np.mean(shax_points[:2], axis=1)
    for k in tqdm(range(K), desc="Creating LAX Sample points", ncols=100):
        points = get_n_points_from_shax(n_points, tck_shax[k], LV_center)
        sample_points.append(points)
    return sample_points


def get_apex_threshold(points_epi, points_endo):
    K_endo = len(points_endo) - 1
    a_epi = calculate_area_points(points_epi[K_endo])
    threshold = a_epi * 0.05
    return threshold


def get_apex_coords(points, K, threshold, slice_thickness):
    apex_xy = np.mean(points, axis=0)
    area = calculate_area_points(points)
    if area < threshold:
        apex_z = -((K - 1) * slice_thickness + area / 2 / threshold * slice_thickness)
    else:
        apex_z = -((K - 1) * slice_thickness + 1 / 2 * slice_thickness)
    return np.hstack((apex_xy, apex_z))


def create_lax_points(sample_points, apex_threshold, slice_thickness):
    """
    The samples points need to be sorted to create LAX points. For n_points in samples we have n_curves=n_points/2.
    This means that for each time step we have n_curves each has 2*K+1 which K is the number of SHAX slices and the 1 corresponds to apex.
    The apex
    """
    n_points = len(sample_points[0])
    n_curves = int(n_points / 2)
    LAX_points = []
    K = len(sample_points)
    # We find the center of the last slice SHAX
    Last_SHAX_points = sample_points[-1][:, :2]
    apex = get_apex_coords(
        Last_SHAX_points, K, apex_threshold, slice_thickness
    )
    # We find the points for each curves of LAX
    for m in tqdm(range(n_curves), desc="Creating LAX Curves", ncols=100):
        points_1 = []
        points_2 = []
        for k in range(K):
            points_1.append(sample_points[k][m])
            points_2.append(sample_points[k][m + n_curves])
        points_1 = np.array(points_1)
        points_2 = np.array(points_2[::-1])
        points = np.vstack((points_1, apex, points_2))
        LAX_points.append(points)
    return LAX_points, apex


def get_weights_for_lax(K, weight_factor):
    W_vector = np.ones(K * 2 + 1)
    W_vector[0] = weight_factor
    W_vector[K] = weight_factor
    W_vector[-1] = weight_factor
    return W_vector


def get_lax_from_laxpoints(LAX_points, smooth_level, lax_spline_order=3):
    n_curves = len(LAX_points)
    tck = []
    for n in range(n_curves):
        K = int((len(LAX_points[n]) - 1) / 2)
        # We use weights to ensure that all LAX pass through base and apex
        W_vector = get_weights_for_lax(K, 1000)
        # spline fitting
        tck_n, u_epi = splprep(
            [
                LAX_points[n][:, 0],
                LAX_points[n][:, 1],
                LAX_points[n][:, 2],
            ],
            w=W_vector,
            s=smooth_level,
            per=False,
            k=lax_spline_order,
        )
        tck.append(tck_n)
    return tck


def get_shax_points_from_lax(tck_lax, z_section):
    n_LAX = len(tck_lax)
    shax_points = np.zeros((n_LAX * 2, 3))
    for n in range(n_LAX):
        tck_lax_n = tck_lax[n]
        points = splev(np.linspace(0, 1, 10000), tck_lax_n)
        points = np.array(points)
        apex_ind = np.argmin(points[2, :])
        idx = (np.abs(points[2, :apex_ind] - z_section)).argmin()
        shax_points[n, :] = points[:, idx]
        idx = (np.abs(points[2, apex_ind:] - z_section)).argmin()
        shax_points[n + n_LAX, :] = points[:, idx + apex_ind]
    return shax_points


def get_shax_area_from_lax(tck_lax, apex, num_sections):
    n_LAX = len(tck_lax)
    shax_points = np.zeros((n_LAX * 2, 3))
    z_list = np.linspace(0, apex[2], num_sections)
    area_shax = np.zeros(num_sections)
    # radii_shax=np.zeros(num_sections)
    for k in range(len(z_list)):
        shax_points = get_shax_points_from_lax(tck_lax, z_list[k])
        tck_k, u_epi = splprep(
            [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
            s=0,
            per=True,
            k=3,
        )
        area_shax[k] = calculate_area_b_spline(tck_k)
    return area_shax


def create_z_sections_for_shax(tck_lax, apex, num_sections):
    # The area is cacluated for num_sections-1 as the base will be added at the end
    area = get_shax_area_from_lax(tck_lax, apex, num_sections - 1)
    area_norm = area / sum(area)
    z_sections = np.cumsum(area_norm * apex[2])
    z_sections = np.hstack([0, z_sections])
    return z_sections


def get_shax_from_lax(tck_lax, apex, num_sections, z_sections_flag=0):
    T_total = len(tck_lax[0])
    tck_shax = []
    if z_sections_flag == 1:
        z_sections = create_z_sections_for_shax(tck_lax, apex, num_sections)
    elif z_sections_flag == 0:
        z_sections = np.linspace(0, apex[2], num_sections)
    for z in z_sections:
        shax_points = get_shax_points_from_lax(tck_lax, z)
        tck_shax_k, u = splprep(
            [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
            s=0,
            per=True,
            k=3,
        )
        tck_shax.append(tck_shax_k)
    return tck_shax


# ----------------------------------------------------------------
# --------- Creating Points Cloud based on SHAX from LAX ---------
# ----------------------------------------------------------------


def equally_spaced_points_on_spline(tck_k, N):
    # Evaluate the spline over a fine grid to get cumulative arc length
    n_points = 1000
    t = np.linspace(0, 1, n_points)
    x, y, z = splev(t, tck_k)
    points = np.vstack((x, y, z)).T

    # Compute the cumulative arc length at each point
    diff_points = np.diff(points, axis=0)
    arc_lengths = np.sqrt((diff_points ** 2).sum(axis=1))
    cumulative_lengths = np.zeros(n_points)
    cumulative_lengths[1:] = np.cumsum(arc_lengths)
    total_length = cumulative_lengths[-1]
    segment_length = total_length / N

    # Find the t values that correspond to the segments with equal lengths
    equally_spaced_t_values = np.zeros(N)
    for i in range(N):
        target_length = i * segment_length
        idx = np.where(cumulative_lengths >= target_length)[0][0]
        equally_spaced_t_values[i] = t[idx]

    x, y, z = splev(equally_spaced_t_values, tck_k)
    points_equal_spaced = np.vstack((x, y, z)).T

    # Compute the centroid of the points
    centroid = np.mean(points_equal_spaced, axis=0)

    # Compute normal vectors as (point - centroid)
    normals = points_equal_spaced - centroid
    # Normalize the normals
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]

    return points_equal_spaced, normals


def check_apex_change(apex, center):
    dist_apex_center_shax = np.sqrt(
        (apex[0] - center[0]) ** 2 + (apex[1] - center[1]) ** 2
    )
    if dist_apex_center_shax > 1:
        warnings.warn(
            f"The apex positision is chaneged {dist_apex_center_shax} mm, which is above the threshold of 1 mm.",
            UserWarning,
        )

    return


def create_apex_lax_points(old_shax_points):
    K_apex = len(old_shax_points)
    n_curves = int(old_shax_points[0].shape[0] / 2)
    center = np.mean(old_shax_points, axis=1)[-1]

    # We find the points for each curves of LAX
    apex_lax_points = []
    for m in range(n_curves):
        points_1 = []
        points_2 = []
        for k in range(K_apex):
            points_1.append(old_shax_points[k][m])
            points_2.append(old_shax_points[k][m + n_curves])
        points_1 = np.array(points_1)
        points_2 = np.array(points_2[::-1])
        apex_lax_points.append(np.vstack((points_1, center, points_2)))

    return apex_lax_points


def get_apex_lax(apex_lax_points):
    n_curves = len(apex_lax_points)
    K_apex_lax = int((len(apex_lax_points[0]) - 1) / 2)
    t_nurbs = []
    c_nurbs = []
    k_nurbs = []
    for n in range(n_curves):
        # We use weights to ensure that all LAX pass through base and apex
        W_vector = get_weights_for_lax(K_apex_lax, 1000)
        # spline fitting
        tck_tk, u_epi = splprep(
            [
                apex_lax_points[n][:, 0],
                apex_lax_points[n][:, 1],
                apex_lax_points[n][:, 2],
            ],
            w=W_vector,
            s=0,
            per=False,
            k=3,
        )
        # spline evaluations
        t_nurbs.append(tck_tk[0])  # Knot vector
        c_nurbs.append(tck_tk[1])  # Coefficients
        k_nurbs.append(tck_tk[2])  # Degree
    tck = (t_nurbs, c_nurbs, k_nurbs)
    return tck


def get_apex_shax_points_from_lax(apex_tck_lax, z):
    n_curves = len(apex_tck_lax[0])
    shax_points = np.zeros((n_curves * 2, 3))
    for n in range(n_curves):
        apex_tck_lax_k = (apex_tck_lax[0][n], apex_tck_lax[1][n], apex_tck_lax[2][n])
        points = splev(np.linspace(0, 1, 1000), apex_tck_lax_k)
        points = np.array(points)
        apex_ind = np.argmin(points[2, :])
        idx = (np.abs(points[2, :apex_ind] - z)).argmin()
        shax_points[n, :] = points[:, idx]
        idx = (np.abs(points[2, apex_ind:] - z)).argmin()
        shax_points[n + n_curves, :] = points[:, idx + apex_ind]
    return shax_points


def create_apex_shax(apex_shax_points, z_sections):
    tck_apex_shax = []
    for i, z in enumerate(z_sections):
        shax_points = apex_shax_points[i]
        shax_points[:,2] = z
        tck_epi_k, u_epi = splprep(
            [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
            s=0,
            per=True,
            k=3,
        )
        tck_apex_shax.append(tck_epi_k)
    return tck_apex_shax


def pop_too_close_shax(old_shax_points):
    # Here we check if the shax points for the current layer is too close to the previous layer
    if len(old_shax_points) < 2:
        return old_shax_points
    diff = old_shax_points[-1] - old_shax_points[-2]
    dist = np.sum(diff**2, axis=1)
    dist_mean = np.mean(dist)
    if dist_mean < 1e-5:
        old_shax_points.pop()
    return old_shax_points


def get_num_apex_slices(point_cloud, k, apex, seed_num_threshold):
    num_points_last_slice = point_cloud[k - 1].shape[0]
    num_apex_slices_from_points = int(num_points_last_slice / seed_num_threshold) + 1
    slice_thickness = point_cloud[k - 2][0, 2] - point_cloud[k - 1][0, 2]
    apex_thickness = point_cloud[k - 1][0, 2] - apex[2]
    num_apex_slices_from_thickness = int(apex_thickness / slice_thickness)
    num_apex_slices = max(num_apex_slices_from_points, num_apex_slices_from_thickness)
    return num_apex_slices


def check_apex_shift(apex, center):
    apex_shift = np.sqrt((center[0] - apex[0]) ** 2 + (center[1] - apex[1]) ** 2)
    if (apex_shift) > 0.5:
        logger.warning(f"Apex was shifted {np.round(apex_shift,4)} mm")


def find_midpoints(last_slice_points, center, m):
    mid_points = []
    # steps = np.linspace(0, 1, m + 2)[:-1]  # m equally spaced steps between 0 and 1, excluding 0 and 1
    steps = third_order_interpolate(0, 1, m + 2)[:-1]
    for step in steps:
        interpolated_points = last_slice_points * (1 - step) + center * step
        mid_points.append(interpolated_points)
    return mid_points

def create_apex_normals(points, center):
    # Compute normal vectors as (point - centroid)
    normals = points - center
    # Normalize the normals
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    return normals

def create_apex_point_cloud(point_cloud, k, tck_shax, apex, seed_num_threshold):
    K = len(tck_shax)
    num_points_last_slice = point_cloud[k - 1].shape[0]
    last_slice_points = point_cloud[k - 1]
    center = np.mean(last_slice_points, axis=0)
    center[2] = apex[2]  # Ensure the z-coordinate matches the apex
    check_apex_shift(apex, center)
    
    # Determine the number of intermediate slices between the last slice and the apex
    num_slices = 2
    apex_seed_num = np.floor(np.linspace(num_points_last_slice, 4, num_slices))
    iter = 0
    while apex_seed_num[-2] - apex_seed_num[-1] > 4 and iter < 5:
        num_slices += 1
        iter += 1
        apex_seed_num = np.floor(np.linspace(num_points_last_slice, 4, num_slices))
    if iter == 5:
        logger.warning('Number of apex short-axis slices may not be sufficient; it is advised to decrease the seed_num_threshold')
    
    m = len(apex_seed_num) - 1
    old_shax_points = find_midpoints(last_slice_points, center, m)
    
    z_last_section = old_shax_points[0][0, 2]
    z_apex = center[2]
    z_sections = third_order_interpolate(z_last_section, z_apex, num_slices)[1:]
    tck_apex_shax = create_apex_shax(old_shax_points[1:], z_sections)
    
    # Initialize lists to collect apex points and normals
    points_apex = []
    normals_apex = []
    
    for n, apex_seed_num_k in enumerate(apex_seed_num[1:]):
        tck_apex_shax_k = tck_apex_shax[n]
        points, _ = equally_spaced_points_on_spline(tck_apex_shax_k, int(apex_seed_num_k))
        points[:, 2] = np.mean(points[:, 2])
        normals = create_apex_normals(points, center)
        points_apex.append(points)
        normals_apex.append(normals)
    
    # Add the apex center point
    points_apex.append(center.reshape(1, -1))
    # For the apex point, set the normal to point upwards along the z-axis
    normals_apex.append(np.array([[0, 0, 1]]))
    
    # Concatenate all points and normals
    points_apex = np.concatenate(points_apex, axis=0)
    normals_apex = np.concatenate(normals_apex, axis=0)
    
    return points_apex, normals_apex


    # fig = go.Figure()
    # for n in range(len(apex_tck_lax[0])):
    #     new_points_epi = splev(np.linspace(0, 1, 100), (apex_tck_lax[0][n],apex_tck_lax[1][n],apex_tck_lax[2][n]))
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x=new_points_epi[0],
    #             y=new_points_epi[1],
    #             z=new_points_epi[2],
    #             showlegend=False,
    #             mode="lines",
    #             name=f"SHAX Epi k={k}",
    #             line=dict(color="red"),
    #         )
    #     )
    # for n in range(len(tck_apex_shax[0])):
    #     new_points_epi = splev(np.linspace(0, 1, 100), (tck_apex_shax[0][n],tck_apex_shax[1][n],tck_apex_shax[2][n]))
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x=new_points_epi[0],
    #             y=new_points_epi[1],
    #             z=new_points_epi[2],
    #             showlegend=False,
    #             mode="lines",
    #             name=f"SHAX Epi k={k}",
    #             line=dict(color="red"),
    #         )
    #     )
    # fnmae = 'test.html'
    # fig.write_html(fnmae)
    
def third_order_interpolate(start_point, end_point, num_point):
    # Define x values corresponding to the z values
    x_values = np.linspace(0, num_point + 1, num_point)

    # Define the third-order polynomial coefficients
    # Let's assume a third-order polynomial of the form z = ax^3 + bx^2 + cx + d
    # We need to determine coefficients a, b, c, d such that:
    # z(0) = z_last_section
    # z(num_apex_slices + 1) = z_apex

    # Create a system of linear equations to solve for a, b, c, d
    A = np.array([
        [0**3, 0**2, 0, 1],
        [(num_point + 1)**3, (num_point + 1)**2, (num_point + 1), 1],
        [(num_point / 2)**3, (num_point / 2)**2, (num_point / 2), 1],
        [(num_point / 3)**3, (num_point / 3)**2, (num_point / 3), 1]
    ])
    b = np.array([start_point, end_point, (start_point + end_point) / 2, (2 * start_point + end_point) / 3])

    # Solve for coefficients a, b, c, d
    coefficients = np.linalg.solve(A, b)

    # Create the polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Evaluate the polynomial at the x values, excluding the end points
    z_sections = polynomial(x_values)
    return z_sections

def create_point_cloud(tck_shax, apex, seed_num_base=30, seed_num_threshold=8):
    point_cloud = []
    normals_list = []
    k_apex = 0
    K = len(tck_shax)
    area_shax = np.zeros(K)
    
    for k in tqdm(range(K), desc="Creating point cloud for mesh generation", ncols=100):
        tck_k = tck_shax[k]
        area_shax[k] = calculate_area_b_spline(tck_k)
        seed_num_k = int(np.cbrt(area_shax[k] / area_shax[0]) * seed_num_base)
        if seed_num_k < seed_num_threshold:
            k_apex = k if k_apex == 0 else k_apex
            points_apex, normals_apex = create_apex_point_cloud(
                point_cloud, k, tck_shax, apex, seed_num_threshold
            )
            point_cloud.append(points_apex)
            normals_list.append(normals_apex)
            break
        else:
            points, normals = equally_spaced_points_on_spline(tck_k, seed_num_k)
            # Ensuring that base is always at z=0
            if k == 0:
                points[:, 2] = 0
            else:
                points[:, 2] = np.mean(points[:, 2])
            point_cloud.append(points)
            normals_list.append(normals)
    
    # Concatenate all points and normals
    point_cloud = np.concatenate(point_cloud, axis=0)
    normals = np.concatenate(normals_list, axis=0)
    
    return point_cloud, k_apex, normals




# ----------------------------------------------------------------
# --------- Creating Mesh from Points Cloud (epi and endo)--------
# ----------------------------------------------------------------
def align_points(points):
    target_center = points[-1][:2]
    for i in range(len(points) - 1):
        array = points[i]
        current_center = np.mean(array[:, :2], axis=0)
        shift = target_center - current_center
        points[i][:, :2] += shift
    return points


def calculate_area(points):
    points = np.vstack((points, points[0]))
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    # Using the Shoelace formula to calculate the area
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def create_flattened_points(points_cloud_aligned, areas, shift_value=1):
    flattened_points = []
    total_shift = 0
    # Flatten the last element (single point) and add it to the list
    last_point = points_cloud_aligned[-1][:2]
    flattened_points.append(last_point)
    for i in range(
        len(points_cloud_aligned) - 2, -1, -1
    ):  # Starting from the third-to-last element
        current_points = points_cloud_aligned[i]
        current_area = areas[i]
        center = np.mean(current_points[:, :2], axis=0)
        shifted_points = (
            current_points[:, :2]
            + total_shift
            * (current_points[:, :2] - center)
            / np.linalg.norm(current_points[:, :2] - center, axis=1)[:, np.newaxis]
        )
        shifted_points_flipped = np.flip(shifted_points, axis=0)
        flattened_points.extend(shifted_points_flipped.tolist())
        total_shift += shift_value
        total_shift += np.sqrt(current_area)
    flattened_points.reverse()
    return np.array(flattened_points)


def create_mesh(points, simplices, stl_mesh=None):
    if stl_mesh is None:
        stl_mesh = mesh.Mesh(np.zeros(simplices.shape[0], dtype=mesh.Mesh.dtype))
    for i, simplex in enumerate(simplices):
        for j in range(3):
            stl_mesh.vectors[i][j] = points[simplex[j], :]
    return stl_mesh


def delauny_tri(point_cloud, shift_value):
    points_cloud_aligned = align_points(point_cloud)
    areas = []
    for points in points_cloud_aligned[:-1]:
        area = calculate_area(points)
        areas.append(area)
    flattened_points = create_flattened_points(points_cloud_aligned, areas, shift_value)
    tri = Delaunay(flattened_points)
    return tri


def expand_slice(points, scale):
    center = np.mean(points, axis=0)
    points_scaled = center + (points - center) * scale
    return points_scaled


def filter_simplices(simplices, threshold):
    # Find rows where all values are greater than the threshold
    rows_to_remove = np.all(simplices >= threshold, axis=1)
    # Filter out these rows
    filtered_simplices = simplices[~rows_to_remove]
    return filtered_simplices


def clean_faces(faces, slice1_size):
    # here we remove the faces which is created within on slince
    condition = np.all(faces < slice1_size, axis=1)
    faces_cleaned = faces[~condition]
    condition = np.all(faces_cleaned >= slice1_size, axis=1)
    faces_cleaned = faces_cleaned[~condition]
    return faces_cleaned


def create_slice_mesh(slice1, slice2, scale):
    flat_slice1 = slice1[:, :2]
    # Handle the case where slice2 is a single point
    if len(slice2.shape) > 1:
        flat_slice2 = slice2[:, :2]
    else:
        flat_slice2 = slice2[:2]
    # Expand the slice with larger area
    if calculate_area_points(flat_slice1)>=calculate_area_points(flat_slice2):
        adjusted_slice1 = expand_slice(flat_slice1, scale)
        combined_slice = np.vstack([adjusted_slice1, flat_slice2])
        # Perform Delaunay triangulation
        threshold = adjusted_slice1.shape[0]
    else:
        adjusted_slice2 = expand_slice(flat_slice2, scale)
        combined_slice = np.vstack([flat_slice1, adjusted_slice2])
        threshold = flat_slice1.shape[0]
    # Perform Delaunay triangulation
    tri = Delaunay(combined_slice)
    faces = filter_simplices(tri.simplices, threshold)
    faces = clean_faces(faces, flat_slice1.shape[0])
    # plot_delaunay_2d(tri.c, combined_slice)
    # plot_delaunay_2d(faces, combined_slice)
    return faces


def create_mesh_slice_by_slice(point_cloud, scale, k_apex):
    vertices = []
    faces = []
    points_cloud_aligned = align_points(point_cloud)
    num_shax = len(points_cloud_aligned) - 1
    for k in range(num_shax):
        slice1 = np.array(points_cloud_aligned[k])
        slice2 = np.array(points_cloud_aligned[k + 1])
        if k>k_apex:
            scale = 2
        slice_faces = create_slice_mesh(slice1, slice2, scale)
        faces_offset = sum(map(len, vertices))
        faces.append(slice_faces + faces_offset)
        vertices.append(point_cloud[k])
    faces = np.vstack(faces)
    return np.vstack(point_cloud), faces


# ----------------------------------------------------------------
# ------------- Creating Mesh from Points Cloud (base)------------
# ----------------------------------------------------------------


def interpolate_splines(tck_inner, tck_outer, num_mid_layers):
    # Number of evaluation points
    num_points = 100
    # Evaluate splines at a fine set of parameter values to Compute centroid (common center point)
    u_fine = np.linspace(0, 1, 1000)
    points_inner = np.array(splev(u_fine, tck_inner))
    points_outer = np.array(splev(u_fine, tck_outer))
    centroid_x = np.mean(np.concatenate([points_inner[0], points_outer[0]]))
    centroid_y = np.mean(np.concatenate([points_inner[1], points_outer[1]]))
    LV_center = [centroid_x, centroid_y,0.0]
    equally_spaced_points_inner = get_n_points_from_shax(num_points, tck_inner, LV_center)
    equally_spaced_points_outer = get_n_points_from_shax(num_points, tck_outer, LV_center)
    # Create intermediate control points from epi to endo
    mid_layers_points = []
    
    for i in reversed(range(1, num_mid_layers + 1)):
        fraction = i / (num_mid_layers + 1)
        mid_points = []
        for point_inner, point_outer in zip(equally_spaced_points_inner,equally_spaced_points_outer):
            mid_point = (1 - fraction) * (point_inner) + fraction * (point_outer)
            mid_points.append(mid_point)
        mid_points = np.array(mid_points)
        mid_layers_points.append(mid_points)
    # Generate new spline representations for the mid-layers
    tck_layers = []
    tck_layers.append(tck_outer)
    for points in mid_layers_points:
        tck, _ = splprep([points[:,0], points[:,1], points[:,2]], s=0, per=True, k=3)
        tck_layers.append(tck)
    tck_layers.append(tck_inner)
    return tck_layers


def create_base_point_cloud(base_endo, base_epi, num_mid_layers=1):
    base_points_cloud = []
    # Creating 2D splines for endo and epi (using only x and y coordinates)
    tck_endo_3d, _ = splprep(
        [base_endo[:, 0], base_endo[:, 1], base_endo[:, 2]], s=0, per=True, k=3
    )
    tck_epi_3d, _ = splprep(
        [base_epi[:, 0], base_epi[:, 1], base_epi[:, 2]], s=0, per=True, k=3
    )
    tck_layers = interpolate_splines(tck_endo_3d, tck_epi_3d, num_mid_layers)
    num_points_endo = base_endo.shape[0]
    num_points_epi = base_epi.shape[0]
    num_points_layers = np.linspace(
        num_points_epi, num_points_endo, num_mid_layers + 2, dtype=int
    )
    for n in range(len(num_points_layers)):
        points = equally_spaced_points_on_spline(
            tck_layers[n], num_points_layers[n]
        )
        points[:, 2] = 0
        base_points_cloud.append(points)
    return base_points_cloud


def create_base_point_cloud_poisson(base_endo, base_epi, num_mid_layers=1):
    base_points_cloud = []
    # we flatten the basal points to z=0 while for z coordinates we use linear interpolations
    z_values = np.linspace(base_epi[0, 2],base_endo[0, 2],num_mid_layers + 2)
    tck_endo_3d, _ = splprep(
    [base_endo[:, 0], base_endo[:, 1], np.zeros_like(base_endo[:, 2])], s=0, per=True, k=3
    )
    tck_epi_3d, _ = splprep(
        [base_epi[:, 0], base_epi[:, 1], np.zeros_like(base_epi[:, 2])], s=0, per=True, k=3
    )
    tck_layers = interpolate_splines(tck_endo_3d, tck_epi_3d, num_mid_layers)
    num_points_endo = base_endo.shape[0]
    num_points_epi = base_epi.shape[0]
    num_points_layers = np.linspace(
        num_points_epi, num_points_endo, num_mid_layers + 2, dtype=int
    )
    for n in range(len(num_points_layers)):
        points = equally_spaced_points_on_spline(
            tck_layers[n], num_points_layers[n]
        )
        points[:, 2] = z_values[n]
        base_points_cloud.append(points)
    return base_points_cloud

def create_base_mesh(base_points):
    vertices = np.vstack(base_points)
    tri = Delaunay(vertices[:, :2])
    threshold = vertices.shape[0] - base_points[-1].shape[0]
    faces = filter_simplices(tri.simplices, threshold)
    return vertices, faces


# ----------------------------------------------------------------
# ------------ Merging Meshes from epi, endo and base ------------
# ----------------------------------------------------------------


def get_num_base_vertices(vertices):
    mask = vertices[:, 2] == 0
    # count = np.count_nonzero(mask)
    count_sum = np.sum(mask)
    return count_sum


def merge_meshes(
    vertices_epi, faces_epi, vertices_base, faces_base, vertices_endo, faces_endo
):
    # the base is mesh from epi to endo
    num_base_endo = get_num_base_vertices(vertices_endo)
    num_base_epi = get_num_base_vertices(vertices_epi)
    num_epi = vertices_epi.shape[0]
    num_endo = vertices_endo.shape[0]
    # num_base=vertices_base.shape[0]
    new_faces_base = np.copy(faces_base)

    new_vertices_base = vertices_base[
        num_base_epi : vertices_base.shape[0] - num_base_endo
    ]
    merged_vertices = np.vstack((vertices_epi, vertices_endo, new_vertices_base))

    # Apply the first condition
    offset = (num_epi) + (num_endo) - (num_base_epi)
    mask1 = (faces_base >= num_base_epi) & (
        faces_base < vertices_base.shape[0] - num_base_endo
    )
    new_faces_base[mask1] += offset

    # Apply the second condition
    mask2 = faces_base >= vertices_base.shape[0] - num_base_endo
    offset = (num_epi) - (vertices_base.shape[0] - num_base_endo)
    new_faces_base[mask2] += offset
    new_faces_endo = faces_endo + vertices_epi.shape[0]

    merged_faces = np.vstack((faces_epi, new_faces_endo, new_faces_base))
    mesh_merged = create_mesh(merged_vertices, merged_faces)
    return mesh_merged
# %%
# ----------------------------------------------------------------
# ----- Poisson surface reconstruction and mesh generation  ------
# ----------------------------------------------------------------
def make_open3d_point_cloud(points: np.array, normals: np.array = None):
    # Create an Open3D point cloud from the numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def preprocess_point_cloud(pcd, k=10, estimate_normals=False):
    """
    Preprocess the point cloud: optionally estimate normals and orient them consistently.
    """
    if estimate_normals:
        # Estimate normals
        pcd.estimate_normals()

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=k)

    return pcd


def create_surface_mesh(pcd):
    """
    Create a surface mesh from the point cloud using Poisson reconstruction.
    """
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5
    )

    # Optional: Clean up the mesh
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    return mesh
        
def optimize_stl_mesh(stl_fname):
    gmsh.initialize()
    # Optionally output messages to the terminal
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("SurfaceMesh")
    # Check if the file exists
    if not os.path.isfile(stl_fname):
        logger.error(f"Error: File '{stl_fname}' not found.")
        gmsh.finalize()
        return
    gmsh.merge(stl_fname)
    # Classify surfaces to prepare for meshing
    # Here, we define the angle (in degrees) between two triangles that will be considered sharp
    angle = 60
    # Force the mesh elements to be classified on discrete entities
    # (surfaces, curves) that respect the sharp edges
    gmsh.model.mesh.classifySurfaces(angle * (3.141592653589793 / 180.0), True, False, 0.01, True)
    # Create geometry from the classified mesh
    gmsh.model.mesh.createGeometry()
    # Synchronize the built-in CAD kernel with the Gmsh model
    gmsh.model.geo.synchronize()
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    # Save the mesh to a file
    gmsh.write(stl_fname)
    # Finalize Gmsh
    gmsh.finalize()
    

def remesh_surface(stl_fname, mesh_size=1):
    """
    Remeshes a 3D surface mesh from an STL file with a specified mesh size
    and saves the output in STL format with a "_coarse" suffix.

    Parameters:
    - stl_fname: str to the input STL file.
    - mesh_size: float, characteristic length for mesh elements (higher values for coarser mesh).

    Returns:
    - vertices: numpy array of shape (n_nodes, 3)
    - faces: numpy array of shape (n_faces, 3), indices into vertices
    """
    # Check if the file exists
    if not os.path.isfile(stl_fname):
        logger.error(f"Error: File '{stl_fname}' not found.")
        return None, None

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Enable terminal output

    try:
        gmsh.merge(stl_fname)
        # Classify surfaces to create geometry
        angle = 60 # Angle threshold for feature detection in degrees
        force_parametrizable_patches = True
        include_boundary = True
        curve_angle = 180  # For sewing surfaces

        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180.0, include_boundary,
            force_parametrizable_patches,
            curve_angle * math.pi / 180.0
        )

        # Create geometry from the classified surfaces
        gmsh.model.mesh.createGeometry()

        # Synchronize the model
        gmsh.model.geo.synchronize()

        # Set the specified mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)

        # Extract nodes and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.array(node_coords).reshape(-1, 3)

        # Create a mapping from node tags to indices
        node_index = {tag: idx for idx, tag in enumerate(node_tags)}

        elementType = 2  # 3-node triangle
        _, element_node_tags = gmsh.model.mesh.getElementsByType(elementType)
        element_node_tags = np.array(element_node_tags).reshape(-1, 3)

        # Map node tags to indices to create faces array
        faces = np.array([[node_index[tag] for tag in tri] for tri in element_node_tags])

        # Save the mesh in STL format
        gmsh.write(stl_fname)
        # Return vertices and faces
        return vertices, faces

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None, None

    finally:
        # Finalize Gmsh
        gmsh.finalize()
        
        
def generate_surface_mesh_from_pointclouds(points_cloud, normals, mesh_size=1, output_folder="output", fname_suffix=''):
    logger.info(f'{fname_suffix} points cloud are being preprocessed ...')
    pcd = make_open3d_point_cloud(points_cloud, normals)
    # Preprocess the point cloud without estimating normals
    if normals is not None:
        estimate_normals=False
    else:
        estimate_normals=True
    pcd = preprocess_point_cloud(pcd, estimate_normals=estimate_normals)
    logger.info(f'{fname_suffix} points cloud normals are set from spline derivatives')
    # Create surface mesh from point cloud
    mesh = create_surface_mesh(pcd)
    logger.info(f'{fname_suffix} initial surface is reconstructed based on Poisson surface reconstruction')
    # Compute normals for the mesh (required for STL export)
    mesh.compute_vertex_normals()
    # Save the mesh as STL
    stl_fname = f"{output_folder}/surface_{fname_suffix}.stl" if fname_suffix != '' else f"{output_folder}/surface.stl"
    o3d.io.write_triangle_mesh(stl_fname, mesh)
    optimize_stl_mesh(stl_fname)
    logger.info(f'{fname_suffix} mesh is being optimized')
    vertices, faces = remesh_surface(stl_fname, mesh_size=mesh_size)
    logger.info(f'{fname_suffix} is remeshed based on user mesh_size = {mesh_size}')
    return vertices, faces

def sort_vertices_by_proximity(vertices):
    n = vertices.shape[0]
    sorted_indices = []
    unvisited = set(range(n))
    current_index = 0  # You can choose any starting point
    sorted_indices.append(current_index)
    unvisited.remove(current_index)

    while unvisited:
        current_point = vertices[current_index]
        # Compute distances to all unvisited points
        unvisited_indices = np.array(list(unvisited))
        unvisited_points = vertices[unvisited_indices]
        distances = np.linalg.norm(unvisited_points - current_point, axis=1)
        # Find the nearest unvisited point
        nearest_index = unvisited_indices[np.argmin(distances)]
        sorted_indices.append(nearest_index)
        unvisited.remove(nearest_index)
        current_index = nearest_index

    # Return the sorted vertices
    return vertices[sorted_indices]

def get_base_from_vertices(vertices):
    # Find indices where the z-coordinate is at the maximum value (base)
    idx = np.where(vertices[:, 2] == np.max(vertices[:, 2]))
    vertices_base = vertices[idx]
    vertices_base_sorted = sort_vertices_by_proximity(vertices_base)
    vertices_base_sorted = np.vstack([vertices_base_sorted, vertices_base_sorted[0]])
    return vertices_base_sorted

# %%
# ----------------------------------------------------------------
# -------------- Main functions to create meshes  ----------------
# ----------------------------------------------------------------
def NodeGenerator(
    mask,
    resolution,
    slice_thickness,
    seed_num_base_epi,
    seed_num_base_endo,
    num_z_sections_epi,
    num_z_sections_endo,
    z_sections_flag_epi=0,
    z_sections_flag_endo=0,
    smooth_shax_epi=5,
    smooth_shax_endo=2,
    n_points_lax=16,
    smooth_lax_epi=1,
    smooth_lax_endo=0.8,
):
    mask_epi, mask_endo = get_endo_epi(mask)
    coords_epi = get_coords_from_mask(mask_epi, resolution, slice_thickness)
    coords_endo = get_coords_from_mask(mask_endo, resolution, slice_thickness)
    tck_epi = get_shax_from_coords(
            coords_epi, smooth_shax_epi
        )
    tck_endo = get_shax_from_coords(
        coords_endo, smooth_shax_endo
    )
    sample_points_epi = get_sample_points_from_shax(tck_epi, n_points_lax)
    sample_points_endo = get_sample_points_from_shax(tck_endo, n_points_lax)
    apex_threshold = get_apex_threshold(sample_points_epi, sample_points_endo)
    LAX_points_epi, apex_epi = create_lax_points(
        sample_points_epi, apex_threshold, slice_thickness
    )
    LAX_points_endo, apex_endo = create_lax_points(
        sample_points_endo, apex_threshold, slice_thickness
    )
    tck_lax_epi = get_lax_from_laxpoints(LAX_points_epi, smooth_lax_epi)
    tck_lax_endo = get_lax_from_laxpoints(LAX_points_endo, smooth_lax_endo)
    tck_shax_epi = get_shax_from_lax(
        tck_lax_epi, apex_epi, num_z_sections_epi, z_sections_flag_epi
    )
    tck_shax_endo = get_shax_from_lax(
        tck_lax_endo, apex_endo, num_z_sections_endo, z_sections_flag_endo
    )
    points_cloud_endo, k_apex_endo, normals_endo = create_point_cloud(
        tck_shax_endo, apex_endo, seed_num_base_endo, seed_num_threshold=8
    )
    points_cloud_epi, k_apex_epi, normals_epi =  create_point_cloud(
        tck_shax_epi, apex_epi, seed_num_base_epi, seed_num_threshold=8
    )
    return points_cloud_epi, points_cloud_endo, k_apex_epi, k_apex_endo, normals_epi, normals_endo


def VentricMesh_delaunay(
    points_cloud_epi,
    points_cloud_endo,
    num_mid_layers_base,
    k_apex_epi,
    k_apex_endo,
    scale_for_delauny=1.2,
    save_flag=True,
    filename_suffix="",
    result_folder="",
):
    vertices_epi, faces_epi = create_mesh_slice_by_slice(points_cloud_epi, scale=scale_for_delauny, k_apex=k_apex_epi)
    vertices_endo, faces_endo = create_mesh_slice_by_slice(points_cloud_endo, scale=scale_for_delauny, k_apex=k_apex_endo)
    points_cloud_base = create_base_point_cloud(points_cloud_endo[0], points_cloud_epi[0], num_mid_layers_base)
    vertices_base, faces_base = create_base_mesh(points_cloud_base)
    mesh_merged = merge_meshes(
        vertices_epi, faces_epi, vertices_base, faces_base, vertices_endo, faces_endo
    )
    if filename_suffix == "":
        mesh_merged_filename = result_folder + "Mesh.stl"
    else:
        mesh_merged_filename = result_folder + "Mesh_" + filename_suffix + ".stl"
    if save_flag:
        mesh_merged.save(mesh_merged_filename)
    mesh_epi = create_mesh(vertices_epi, faces_epi)
    mesh_epi_filename = result_folder + "Mesh_epi_" + filename_suffix + ".stl"
    mesh_epi.save(mesh_epi_filename)
    mesh_endo = create_mesh(vertices_endo, faces_endo)
    mesh_endo_filename = result_folder + "Mesh_endo_" + filename_suffix + ".stl"
    mesh_endo.save(mesh_endo_filename)
    mesh_base=create_mesh(vertices_base,faces_base)
    mesh_base_filename=result_folder+'Mesh_base_'+filename_suffix+'.stl'
    mesh_base.save(mesh_base_filename)
    return mesh_merged, mesh_epi, mesh_endo, mesh_base

def VentricMesh_poisson(
    points_cloud_epi,
    points_cloud_endo,
    num_mid_layers_base,
    SurfaceMeshSizeEpi=1,
    SurfaceMeshSizeEndo=1,
    normals_epi = None,
    normals_endo = None,
    save_flag=True,
    filename_suffix="",
    result_folder="",
):
    stacked_points_epi = np.vstack(points_cloud_epi)
    stacked_normals_epi = np.vstack(normals_epi)
    stacked_points_endo = np.vstack(points_cloud_endo)
    stacked_normals_endo= np.vstack(normals_endo)
    vertices_epi, faces_epi = generate_surface_mesh_from_pointclouds(stacked_points_epi, stacked_normals_epi, mesh_size=SurfaceMeshSizeEpi, output_folder=result_folder, fname_suffix='epi')
    vertices_endo, faces_endo = generate_surface_mesh_from_pointclouds(stacked_points_endo, stacked_normals_endo, mesh_size=SurfaceMeshSizeEndo, output_folder=result_folder, fname_suffix='endo')
    
    base_endo = get_base_from_vertices(vertices_endo)
    base_epi = get_base_from_vertices(vertices_epi)
    points_cloud_base = create_base_point_cloud_poisson(base_endo, base_epi, num_mid_layers_base)
    points_cloud_base[0] = base_epi
    points_cloud_base[-1] = base_endo
    vertices_base, faces_base = create_base_mesh(points_cloud_base)
    mesh_merged = merge_meshes(
        vertices_epi, faces_epi, vertices_base, faces_base, vertices_endo, faces_endo
    )
    if filename_suffix == "":
        mesh_merged_filename = result_folder + "Mesh.stl"
    else:
        mesh_merged_filename = result_folder + "Mesh_" + filename_suffix + ".stl"
    if save_flag:
        mesh_merged.save(mesh_merged_filename)
    mesh_epi = create_mesh(vertices_epi, faces_epi)
    mesh_epi_filename = result_folder + "Mesh_epi_" + filename_suffix + ".stl"
    mesh_epi.save(mesh_epi_filename)
    mesh_endo = create_mesh(vertices_endo, faces_endo)
    mesh_endo_filename = result_folder + "Mesh_endo_" + filename_suffix + ".stl"
    mesh_endo.save(mesh_endo_filename)
    mesh_base=create_mesh(vertices_base,faces_base)
    mesh_base_filename=result_folder+'Mesh_base_'+filename_suffix+'.stl'
    mesh_base.save(mesh_base_filename)
    return mesh_epi_filename, mesh_endo_filename, mesh_base_filename
# ----------------------------------------------------------------
# ------------------- Mesh Quality functions  --------------------
# ----------------------------------------------------------------


def triangle_aspect_ratio(vertices):
    """
    Calculate the aspect ratio of a triangle.
    Aspect Ratio: The maximum ratio of the length of a side to the perpendicular distance
    from that side to its opposite node, multiplied by √3/2. √3/2 is a factor based on
    an equilateral triangle.
    """
    # Calculate the lengths of the sides of the triangle
    lengths = np.sqrt(
        np.sum(np.square(np.diff(vertices[np.array([0, 1, 2, 0])], axis=0)), axis=1)
    )
    # Calculate the semi-perimeter
    s = np.sum(lengths) / 2
    # Calculate the area using Heron's formula
    area = np.sqrt(s * np.prod(s - lengths))
    # The shortest altitude is area * 2 / longest side
    shortest_altitude = (area * 2) / np.max(lengths)
    # Aspect ratio is longest side length over shortest altitude
    return np.max(lengths) / shortest_altitude * np.sqrt(3) / 2


def check_mesh_quality(mesh_data, file_path=None):
    """
    Get a mesh file and print out its quality metrics.
    """
    if file_path is not None:
        file = open(file_path, 'w')
    else:
        file = None
    line = "======= Mesh Statistics ======="
    print(line)
    if file:
        file.write(line + '\n')
    # Basic Properties
    line = f"Triangles: {len(mesh_data.vectors)}"
    print(line)
    if file:
        file.write(line + '\n')
    # Triangle Quality
    aspect_ratios = [triangle_aspect_ratio(triangle) for triangle in mesh_data.vectors]
    average_aspect_ratio = np.mean(aspect_ratios)
    std_aspect_ratio = np.std(aspect_ratios)
    
    line = f"Average Aspect Ratio: {np.round(average_aspect_ratio, 3)}"
    print(line)
    if file:
        file.write(line + '\n')
    print()
    line = f"Standard Deviation of Aspect Ratios: {np.round(std_aspect_ratio, 3)}"
    print(line)
    if file:
        file.write(line + '\n')
    # Count the number of triangles with an aspect ratio larger than 5
    num_large_aspect_ratio = sum(1 for ratio in aspect_ratios if ratio > 5)
    line = f"Number of Triangles with Aspect Ratio > 5: {num_large_aspect_ratio}"
    print(line)
    if file:
        file.write(line + '\n')
    line = "==============================="
    print(line)
    print(line)
    if file:
        file.write(line + '\n')
        file.write(line + '\n')
    return aspect_ratios


# %%
def print_mesh_quality_report(n_bins, file_path=None):
    # Retrieve statistics about the mesh
    # num_elem=gmsh.model.mesh.getMaxElementTag()
    elem_tags = gmsh.model.mesh.getElementsByType(
        4
    )  # getting all the tetrahedron eleemnts
    # num_elem=elem_tags[0].shape[0]
    q = gmsh.model.mesh.getElementQualities(elem_tags[0])
    counts, bin_edges = np.histogram(q, bins=n_bins, range=(0, 1))
    
    if file_path is not None:
        file = open(file_path, 'a')
    else:
        file = None
    
    line = "Quality report of the final volumetric mesh:"
    print(line)
    if file:
        file.write(line + '\n')
    for i in range(n_bins):
        line = f"{bin_edges[i]:.2f} < quality < {bin_edges[i+1]:.2f} : {counts[i]:>10} elements"
        print(line)
        if file:
            file.write(line + '\n')

    if file is not None:
        file.close()
        
def generate_3d_mesh_from_stl(stl_path, mesh_path, MeshSizeMin=None, MeshSizeMax=None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.merge(stl_path)
    # Meshing options
    # # Set maximum element size
    if MeshSizeMin is not None:
        gmsh.option.setNumber('Mesh.MeshSizeMin', MeshSizeMin)
    if MeshSizeMax is not None:
        gmsh.option.setNumber('Mesh.MeshSizeMax', MeshSizeMax)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1: Delaunay, 4: Frontal
    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    # Generate 3D mesh
    gmsh.model.mesh.OptimizeNetgen = 1
    gmsh.model.mesh.SurfaceFaces = 1
    gmsh.model.mesh.Algorithm = 1  # (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=bamg, 8=delquad) (Default=2)
    gmsh.model.mesh.Algorithm3D = 4  # (1=Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree) (Default=1)
    gmsh.model.mesh.Recombine3DAll = 0
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_path)
    print("===============================")
    print_mesh_quality_report(10, file_path=stl_path[:-4]+'_report.txt')
    print("===============================")
    print("===============================")
    gmsh.finalize()


def generate_3d_mesh_from_seperate_stl(mesh_epi_filename, mesh_endo_filename, mesh_base_filename, output_mesh_filename,  MeshSizeMin=None, MeshSizeMax=None):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("3D Mesh")
    gmsh.option.setNumber("General.Verbosity", 0)

    # Merge the STL files
    gmsh.merge(mesh_epi_filename)
    gmsh.merge(mesh_endo_filename)
    gmsh.merge(mesh_base_filename)

    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.create_geometry()
    gmsh.model.mesh.create_topology()
    surfaces = gmsh.model.getEntities(2)
    
    gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces], 1)
    vol = gmsh.model.geo.addVolume([1], 1)
    
    physical_groups = {
        "Epi": [1],
        "Endo": [2],
        "Base": [3],
    }
    for name, tag in physical_groups.items():
        p = gmsh.model.addPhysicalGroup(2, tag)
        gmsh.model.setPhysicalName(2, p, name)

    p = gmsh.model.addPhysicalGroup(3, [vol], 9)
    gmsh.model.setPhysicalName(3, p, "Wall")

    if MeshSizeMin is not None:
        gmsh.option.setNumber('Mesh.MeshSizeMin', MeshSizeMin)
    if MeshSizeMax is not None:
        gmsh.option.setNumber('Mesh.MeshSizeMax', MeshSizeMax)
            
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    # Save the mesh to the specified file
    print("===============================")
    gmsh.write(output_mesh_filename)
    print_mesh_quality_report(10, file_path=output_mesh_filename[:-4]+'_report.txt')
    print("===============================")
    # Finalize Gmsh
    gmsh.finalize()