# %%
import numpy as np
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
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)
    K, I, _, T_total = mask.shape
    kernel = np.ones((3, 3), np.uint8)
    mask_epi = np.zeros((K, I, I, T_total))
    mask_endo = np.zeros((K, I, I, T_total))

    for t in range(T_total):
        for k in range(K):
            mask_t = mask[k, :, :, t]
            img = np.uint8(mask_t * 255)
            img_dilated = binary_dilation(img, structure=kernel).astype(img.dtype)
            img_edges = img_dilated - img
            img_edges[img_edges == 2] = 0
            flag, visited, visited_reversed = is_connected(img_edges)
            if flag:
                img_epi = img_edges
                img_endo = np.zeros((I, I))
            else:
                img_epi = np.zeros((I, I), dtype=np.uint8)
                img_endo = np.zeros((I, I), dtype=np.uint8)
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
            mask_epi[k, :, :, t] = img_epi
            mask_endo[k, :, :, t] = img_endo
    return mask_epi, mask_endo


# ----------------------------------------------------------------
# ------------ Creating BSplines and Smooth Contours -------------
# ----------------------------------------------------------------


# % Getting shax bsplines from epi and endo masks
def get_shax_from_mask(mask, resolution, slice_thickness, smooth_level):
    # The smooth_level is based on the area enclosed by the epi/endo points and it should.
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="scipy.interpolate"
    )
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)
    K, I, _, T_total = mask.shape
    t_nurbs = [[] for _ in range(mask.shape[3])]
    c_nurbs = [[] for _ in range(mask.shape[3])]
    k_nurbs = [[] for _ in range(mask.shape[3])]

    for t in tqdm(range(T_total), desc="Creating SHAX Curves", ncols=100):
        for k in range(K):
            img = mask[k, :, :, t]
            coords = coords_from_img(img, resolution)
            # the if conditions is for the endo as there are not alway K
            if len(coords) > 0:
                coords_sorted = sorting_coords(coords, resolution)
                # Adding the first point to the end to make it periodic
                np.vstack((coords_sorted, coords_sorted[0, :]))
                # spline fitting
                z = -(k) * slice_thickness
                z_list = np.ones(coords_sorted.shape[0]) * z
                area = calculate_area_points(coords_sorted)
                tck_tk, u_epi = splprep(
                    [coords_sorted[:, 0], coords_sorted[:, 1], z_list],
                    s=smooth_level * area,
                    per=True,
                    k=3,
                )
                # spline evaluations
                t_nurbs[t].append(tck_tk[0])  # Knot vector
                c_nurbs[t].append(tck_tk[1])  # Coefficients
                k_nurbs[t].append(tck_tk[2])  # Degree
    tck = (t_nurbs, c_nurbs, k_nurbs)
    return tck


# % Creating LAX BSplines from SHAX
def get_sample_points_from_shax(tck_shax, n_points):
    T_total = len(tck_shax[0])
    sample_points = [[] for _ in range(T_total)]
    for t in tqdm(range(T_total), desc="Creating LAX Sample points", ncols=100):
        K = len(tck_shax[0][t])
        # We find the center based on the SHAX of the last slice
        shax_points = get_points_from_tck(tck_shax, t, -1)
        LV_center = np.mean(shax_points[:2], axis=1)
        for k in range(K):
            points = get_n_points_from_shax(n_points, tck_shax, t, k, LV_center)
            sample_points[t].append(points)
    return sample_points


def get_apex_threshold(points_epi, points_endo):
    T_total = len(points_endo)
    threshold = np.zeros(T_total)
    for t in range(T_total):
        K_endo = len(points_endo[t]) - 1
        a_epi = calculate_area_points(points_epi[t][K_endo])
        threshold[t] = a_epi * 0.05
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
    T_total = len(sample_points)
    n_points = len(sample_points[0][0])
    n_curves = int(n_points / 2)
    LAX_points = [[] for _ in range(T_total)]
    apex = np.zeros((T_total, 3))
    # for t in range(T_total):
    for t in tqdm(range(T_total), desc="Creating LAX Curves", ncols=100):
        K = len(sample_points[t])
        # We find the center of the last slice SHAX
        Last_SHAX_points = sample_points[t][-1][:, :2]
        apex[t, :] = get_apex_coords(
            Last_SHAX_points, K, apex_threshold[t], slice_thickness
        )
        # We find the points for each curves of LAX
        for m in range(n_curves):
            points_1 = []
            points_2 = []
            for k in range(K):
                points_1.append(sample_points[t][k][m])
                points_2.append(sample_points[t][k][m + n_curves])
            points_1 = np.array(points_1)
            points_2 = np.array(points_2[::-1])
            points = np.vstack((points_1, apex[t, :], points_2))
            LAX_points[t].append(points)
    return LAX_points, apex


def get_weights_for_lax(K, weight_factor):
    W_vector = np.ones(K * 2 + 1)
    W_vector[0] = weight_factor
    W_vector[K] = weight_factor
    W_vector[-1] = weight_factor
    return W_vector


def get_lax_from_laxpoints(LAX_points, smooth_level):
    T_total = len(LAX_points)
    n_curves = len(LAX_points[0])
    t_nurbs = [[] for _ in range(T_total)]
    c_nurbs = [[] for _ in range(T_total)]
    k_nurbs = [[] for _ in range(T_total)]
    for t in range(T_total):
        for n in range(n_curves):
            K = int((len(LAX_points[t][n]) - 1) / 2)
            # We use weights to ensure that all LAX pass through base and apex
            W_vector = get_weights_for_lax(K, 1000)
            # spline fitting
            tck_tk, u_epi = splprep(
                [
                    LAX_points[t][n][:, 0],
                    LAX_points[t][n][:, 1],
                    LAX_points[t][n][:, 2],
                ],
                w=W_vector,
                s=smooth_level,
                per=False,
                k=3,
            )
            # spline evaluations
            t_nurbs[t].append(tck_tk[0])  # Knot vector
            c_nurbs[t].append(tck_tk[1])  # Coefficients
            k_nurbs[t].append(tck_tk[2])  # Degree
    tck = (t_nurbs, c_nurbs, k_nurbs)
    return tck


def get_shax_points_from_lax(tck_lax, t, z_section):
    n_curves = len(tck_lax[0][t])
    shax_points = np.zeros((n_curves * 2, 3))
    for n in range(n_curves):
        tck_lax_tn = (tck_lax[0][t][n], tck_lax[1][t][n], tck_lax[2][t][n])
        points = splev(np.linspace(0, 1, 10000), tck_lax_tn)
        points = np.array(points)
        apex_ind = np.argmin(points[2, :])
        idx = (np.abs(points[2, :apex_ind] - z_section)).argmin()
        shax_points[n, :] = points[:, idx]
        idx = (np.abs(points[2, apex_ind:] - z_section)).argmin()
        shax_points[n + n_curves, :] = points[:, idx + apex_ind]
    return shax_points


def get_shax_area_from_lax(tck_lax, t, apex, num_sections):
    n_LAX = len(tck_lax[0][t][0])
    shax_points = np.zeros((n_LAX * 2, 3))
    z_list = np.linspace(0, apex[t, 2], num_sections)
    area_shax = np.zeros(num_sections)
    # radii_shax=np.zeros(num_sections)
    for k in range(len(z_list)):
        shax_points = get_shax_points_from_lax(tck_lax, t, z_list[k])
        tck_tk, u_epi = splprep(
            [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
            s=0,
            per=True,
            k=3,
        )
        area_shax[k] = calculate_area_b_spline(tck_tk)
    return area_shax


def create_z_sections_for_shax(tck_lax, apex, t, num_sections):
    # The area is cacluated for num_sections-1 as the base will be added at the end
    area = get_shax_area_from_lax(tck_lax, t, apex, num_sections - 1)
    area_norm = area / sum(area)
    z_sections = np.cumsum(area_norm * apex[t, 2])
    z_sections = np.hstack([0, z_sections])
    return z_sections


def get_shax_from_lax(tck_lax, apex, num_sections, z_sections_flag=0):
    T_total = len(tck_lax[0])
    # n_curves = len(tck_lax[0][0])
    t_nurbs = [[] for _ in range(T_total)]
    c_nurbs = [[] for _ in range(T_total)]
    k_nurbs = [[] for _ in range(T_total)]
    for t in tqdm(
        range(T_total), desc="SHAX Allignment with respect to Generated LAX", ncols=100
    ):
        if z_sections_flag == 1:
            z_sections = create_z_sections_for_shax(tck_lax, apex, t, num_sections)
        elif z_sections_flag == 0:
            z_sections = np.linspace(0, apex[t, 2], num_sections)
        for z in z_sections:
            shax_points = get_shax_points_from_lax(tck_lax, t, z)
            tck_epi_tk, u_epi = splprep(
                [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
                s=0,
                per=True,
                k=3,
            )
            t_nurbs[t].append(tck_epi_tk[0])  # Knot vector
            c_nurbs[t].append(tck_epi_tk[1])  # Coefficients
            k_nurbs[t].append(tck_epi_tk[2])  # Degree
    tck_shax = (t_nurbs, c_nurbs, k_nurbs)
    return tck_shax


# ----------------------------------------------------------------
# --------- Creating Points Cloud based on SHAX from LAX ---------
# ----------------------------------------------------------------


def equally_spaced_points_on_spline(tck_tk, N):
    # Evaluate the spline over a fine grid
    n_points = 1000
    t = np.linspace(0, 1, n_points)
    x, y, z = splev(t, tck_tk)
    points = np.vstack((x, y, z)).T
    # Compute the cumulative arc length at each point
    diff_points = np.diff(points, axis=0)
    arc_lengths = np.sqrt(
        diff_points[:, 0] ** 2 + diff_points[:, 1] ** 2 + diff_points[:, 2] ** 2
    )
    cumulative_lengths = np.zeros(n_points)
    cumulative_lengths[1:] = np.cumsum(arc_lengths)
    total_length = cumulative_lengths[-1]
    segment_length = total_length / N
    # Find the t values that correspond to the segments with this segment lengths
    equally_spaced_t_values = np.zeros(N)
    for i in range(N):
        target_length = i * segment_length
        idx = np.where(cumulative_lengths >= target_length)[0][0]
        equally_spaced_t_values[i] = t[idx]
    x, y, z = splev(equally_spaced_t_values, tck_tk)
    points_equal_spaced = np.vstack((x, y, z)).T
    return points_equal_spaced


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


def create_apex_shax(apex_tck_lax, num_apex_slices, z_sections):
    t_nurbs, c_nurbs, k_nurbs = [], [], []
    for z in z_sections:
        shax_points = get_apex_shax_points_from_lax(apex_tck_lax, z)
        tck_epi_tk, u_epi = splprep(
            [shax_points[:, 0], shax_points[:, 1], shax_points[:, 2]],
            s=0,
            per=True,
            k=3,
        )
        t_nurbs.append(tck_epi_tk[0])  # Knot vector
        c_nurbs.append(tck_epi_tk[1])  # Coefficients
        k_nurbs.append(tck_epi_tk[2])  # Degree
    tck_apex_shax = (t_nurbs, c_nurbs, k_nurbs)
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


def get_num_apex_slices(point_cloud, t, k, apex, seed_num_threshold):
    num_points_last_slice = point_cloud[t][k - 1].shape[0]
    num_apex_slices_from_points = int(num_points_last_slice / seed_num_threshold) + 1
    slice_thickness = point_cloud[t][k - 2][0, 2] - point_cloud[t][k - 1][0, 2]
    apex_thickness = point_cloud[t][k - 1][0, 2] - apex[t][2]
    num_apex_slices_from_thickness = int(apex_thickness / slice_thickness)
    num_apex_slices = max(num_apex_slices_from_points, num_apex_slices_from_thickness)
    return num_apex_slices


def check_apex_shift(apex, center):
    apex_shift = np.sqrt((center[0] - apex[0]) ** 2 + (center[1] - apex[1]) ** 2)
    if (apex_shift) > 0.5:
        logger.warning(f"Apex was shifted {np.round(apex_shift,4)} mm")


def find_midpoints(last_slice_points, center, m):
    mid_points = []
    steps = np.linspace(0, 1, m + 2)[
        :-1
    ]  # m equally spaced steps between 0 and 1, excluding 0 and 1
    for step in steps:
        interpolated_points = last_slice_points * (1 - step) + center * step
        mid_points.append(interpolated_points)
    return mid_points


def create_apex_point_cloud(point_cloud, t, k, tck_shax, apex, seed_num_threshold):
    K = len(tck_shax[0][t])
    num_points_last_slice = point_cloud[t][k - 1].shape[0]
    last_slice_points = point_cloud[t][k - 1]
    # if K-k<2:
    # logger.warning("The number of shax slices are too low, Try to re-run get_shax_from_lax with higher value for num_z_sections")
    center = np.mean(last_slice_points, axis=0)
    center[2] = apex[t][2]
    check_apex_shift(apex[t], center)

    scale_factor = (
        num_points_last_slice / seed_num_threshold
    )  # if the area is too big it should be scale so more intermediatry slices are considerd
    apex_slices = np.floor(np.linspace(num_points_last_slice, 4, int(scale_factor * 4)))

    m = len(apex_slices) - 1
    old_shax_points = find_midpoints(last_slice_points, center, m)
    # else:
    #     num_points_last_slice_even = int(num_points_last_slice / 2) * 2 # ensuring that the num_points is a even so we could have n/2 lax curves for apex
    #     old_shax_points = []
    #     for l in range(K-k+1):
    #         tck_tk = (tck_shax[0][t][k-1+l], tck_shax[1][t][k-1+l], tck_shax[2][t][k-1+l])
    #         slice_points = equally_spaced_points_on_spline(tck_tk, num_points_last_slice_even)
    #         slice_points[:, 2] = np.mean(slice_points[:, 2])
    #         old_shax_points.append(slice_points)
    #         old_shax_points = pop_too_close_shax(old_shax_points)
    #     # defining the new apex point based on the center of the last shax point
    #     center = np.mean(old_shax_points, axis=1)[-1]
    #     check_apex_shift(apex[t],center)

    apex_lax_points = create_apex_lax_points(old_shax_points)
    apex_tck_lax = get_apex_lax(apex_lax_points)
    num_apex_slices = get_num_apex_slices(point_cloud, t, k, apex, seed_num_threshold)

    z_last_section = old_shax_points[0][0, 2]
    z_apex = center[2]
    z_sections = np.linspace(z_last_section, z_apex, num_apex_slices + 2)[1:-1]
    tck_apex_shax = create_apex_shax(apex_tck_lax, num_apex_slices, z_sections)

    apex_seed_num = np.floor(
        np.linspace(num_points_last_slice, 4, len(z_sections) + 1)
    )[1:]
    for n, apex_seed_num_k in enumerate(apex_seed_num):
        tck_apex_shax_k = (
            tck_apex_shax[0][n],
            tck_apex_shax[1][n],
            tck_apex_shax[2][n],
        )
        points = equally_spaced_points_on_spline(tck_apex_shax_k, int(apex_seed_num_k))
        points[:, 2] = np.mean(points[:, 2])
        point_cloud[t].append(points)

    point_cloud[t].append(center)

    return point_cloud

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


def create_point_cloud(tck_shax, apex, seed_num_base=30, seed_num_threshold=8):
    T = len(tck_shax[0])
    point_cloud = [[] for t in range(T)]
    for t in tqdm(range(T), desc="Creating pointcloud for mesh generation", ncols=100):
        K = len(tck_shax[0][t])
        area_shax = np.zeros(K)
        for k in range(K):
            tck_tk = (tck_shax[0][t][k], tck_shax[1][t][k], tck_shax[2][t][k])
            area_shax[k] = calculate_area_b_spline(tck_tk)
            seed_num_k = int(np.cbrt(area_shax[k] / area_shax[0]) * seed_num_base)
            if seed_num_k < seed_num_threshold:
                point_cloud = create_apex_point_cloud(
                    point_cloud, t, k, tck_shax, apex, seed_num_threshold
                )
                break
            else:
                points = equally_spaced_points_on_spline(tck_tk, seed_num_k)
                # ensuring that base is always at z=0
                if k == 0:
                    points[:, 2] = 0
                else:
                    points[:, 2] = np.mean(points[:, 2])
                point_cloud[t].append(points)

    return point_cloud


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
    adjusted_slice1 = expand_slice(flat_slice1, scale)
    combined_slice = np.vstack([adjusted_slice1, flat_slice2])
    # Perform Delaunay triangulation
    tri = Delaunay(combined_slice)
    threshold = adjusted_slice1.shape[0]
    faces = filter_simplices(tri.simplices, threshold)
    faces = clean_faces(faces, flat_slice1.shape[0])
    return faces


def create_mesh_slice_by_slice(point_cloud, scale=2):
    vertices = []
    faces = []
    points_cloud_aligned = align_points(point_cloud)
    num_shax = len(points_cloud_aligned) - 1
    for k in range(num_shax):
        slice1 = np.array(points_cloud_aligned[k])
        slice2 = np.array(points_cloud_aligned[k + 1])
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

    u_common = np.linspace(0, 1, num_points)
    points_inner = np.array(splev(u_common, tck_inner))
    points_outer = np.array(splev(u_common, tck_outer))

    # Create intermediate control points from epi to endo
    mid_layers_points = []
    for i in reversed(range(1, num_mid_layers + 1)):
        fraction = i / (num_mid_layers + 1)
        mid_points = (1 - fraction) * points_inner + fraction * points_outer
        mid_layers_points.append(mid_points)

    # Generate new spline representations for the mid-layers
    tck_layers = []
    tck_layers.append(tck_outer)
    for points in mid_layers_points:
        tck, _ = splprep([points[0], points[1], points[2]], s=0, per=True, k=3)
        tck_layers.append(tck)
    tck_layers.append(tck_inner)
    return tck_layers


def create_base_point_cloud(points_endo, points_epi, num_mid_layers=1):
    T_total = len(points_endo)
    base_points_cloud = [[] for _ in range(T_total)]
    for t in range(T_total):
        base_endo = points_endo[t][0]
        base_epi = points_epi[t][0]
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
            base_points_cloud[t].append(points)
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
    tck_epi = get_shax_from_mask(mask_epi, resolution, slice_thickness, smooth_shax_epi)
    tck_endo = get_shax_from_mask(
        mask_endo, resolution, slice_thickness, smooth_shax_endo
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
    points_cloud_endo = create_point_cloud(
        tck_shax_endo, apex_endo, seed_num_base_endo, seed_num_threshold=8
    )
    points_cloud_epi = create_point_cloud(
        tck_shax_epi, apex_epi, seed_num_base_epi, seed_num_threshold=8
    )
    return points_cloud_epi, points_cloud_endo


def VentricMesh(
    points_cloud_epi,
    points_cloud_endo,
    t_mesh,
    num_mid_layers_base,
    save_flag=True,
    filename_suffix="",
    result_folder="",
):
    vertices_epi, faces_epi = create_mesh_slice_by_slice(points_cloud_epi[t_mesh])
    vertices_endo, faces_endo = create_mesh_slice_by_slice(points_cloud_endo[t_mesh])
    points_cloud_base = create_base_point_cloud(
        points_cloud_endo, points_cloud_epi, num_mid_layers_base
    )
    vertices_base, faces_base = create_base_mesh(points_cloud_base[t_mesh])
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
    # mesh_base=create_mesh(vertices_base,faces_base)
    # mesh_base_filename=result_folder+'Mesh_base_'+filename_suffix+'.stl'
    # mesh_base.save(mesh_base_filename)
    return mesh_merged


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


def check_mesh_quality(mesh_data):
    """
    Get a mesh file and print out its quality metrics.
    """
    print("===============================")
    print("===============================")
    # Basic Properties
    print("Triangles:", len(mesh_data.vectors))
    # Triangle Quality
    aspect_ratios = [triangle_aspect_ratio(triangle) for triangle in mesh_data.vectors]
    average_aspect_ratio = np.mean(aspect_ratios)
    std_aspect_ratio = np.std(aspect_ratios)
    print("Average Aspect Ratio:", np.round(average_aspect_ratio, 3))
    print("Standard Deviation of Aspect Ratios:", np.round(std_aspect_ratio, 3))
    # Count the number of triangles with an aspect ratio larger than 5
    num_large_aspect_ratio = sum(1 for ratio in aspect_ratios if ratio > 5)
    print("Number of Triangles with Aspect Ratio > 5:", num_large_aspect_ratio)
    print("===============================")
    print("===============================")
    return aspect_ratios


# %%
def print_mesh_quality_report(n_bins):
    # Retrieve statistics about the mesh
    # num_elem=gmsh.model.mesh.getMaxElementTag()
    elem_tags = gmsh.model.mesh.getElementsByType(
        4
    )  # getting all the tetrahedron eleemnts
    # num_elem=elem_tags[0].shape[0]
    q = gmsh.model.mesh.getElementQualities(elem_tags[0])
    counts, bin_edges = np.histogram(q, bins=n_bins, range=(0, 1))
    for i in range(n_bins):
        print(
            f"{bin_edges[i]:.2f} < quality < {bin_edges[i+1]:.2f} : {counts[i]:>10} elements"
        )


def generate_3d_mesh_from_stl(stl_path, mesh_path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.merge(stl_path)
    # Meshing options
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
    print("Quality report of the final volumetri mesh:")
    print_mesh_quality_report(10)
    print("===============================")
    print("===============================")
    gmsh.finalize()
