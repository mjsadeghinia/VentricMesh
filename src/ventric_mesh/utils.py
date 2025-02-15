"""
Useful functions for creating the mesh

Last Modified: 18.01.2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep
import math
import plotly.graph_objects as go
from scipy.interpolate import BSpline
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from structlog import get_logger

logger = get_logger()

# ----------------------------------------------------------------
# ----------------------    Utilities    -------------------------
# ----------------------------------------------------------------


# Definig function is_connected to checking if there is unconnected pixels
# from a long axis view. Here a function is defined to check if there is any gap
# in the image or not
def is_connected(matrix):
    flag = False
    visited = set()
    visited_reversed = set()
    rows, cols = len(matrix), len(matrix[0])

    # Defining depth first search (DFS) for traversing the pixels It gets the r
    # and c of a True value pixel and looks for the neighboutring pixels to add
    # it to the set of visited. the flg is a flag for wether it should it use
    # visited (1) or visited_reversed (0) set. The latter is used in case of an opening.
    #
    def dfs(r, c, flg):
        if flg:
            if (
                0 <= r < rows
                and 0 <= c < cols
                and matrix[r][c]
                and (r, c) not in visited
            ):
                visited.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc, 1)
        else:
            if (
                0 <= r < rows
                and 0 <= c < cols
                and matrix[r][c]
                and (r, c) not in visited_reversed
            ):
                visited_reversed.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc, 0)

    # Find first True value and start DFS from there
    for r in range(rows):
        if flag:
            break
        for c in range(cols):
            if matrix[r][c]:
                dfs(r, c, 1)
                flag = True
                break

    # Check for any unvisited True values
    indices = np.argwhere(matrix)
    index_set = set(map(tuple, indices))
    if visited == index_set:
        return True, visited, visited_reversed
    else:
        flag = False
        for c in range(cols - 1, -1, -1):
            if flag:
                break
            for r in range(rows):
                if matrix[r][c] and (r, c) not in visited:
                    dfs(r, c, 0)
                    flag = True
                    break

        return False, visited, visited_reversed


# Getting a coordinates of a binary image with correct value So when ploting it would be similiar to the image itself
def coords_from_img(img, resolution):
    I = img.shape[0]
    temp = np.argwhere(img == 1) * resolution
    coords = np.zeros((len(temp), 2))
    coords[:, 0] = temp[:, 1]
    coords[:, 1] = temp[:, 0]
    coords[:, 1] = I * resolution - coords[:, 1]
    return coords


# function to sort the coords based on the closest neighbouring point. This makes it possible to fit using nurbs
def sorting_coords(coords, resolution):
    points = [list(coord) for coord in coords]
    coords_sorted = [points[0]]
    used_indices = {0}
    for _ in range(1, len(points)):
        current_point = coords_sorted[-1]
        closest_point_index = find_closest_point(
            current_point, points, used_indices, resolution
        )
        if closest_point_index >= 0:
            coords_sorted.append(points[closest_point_index])
            used_indices.add(closest_point_index)
    return np.array(coords_sorted)


# functions for sorting the points based on the nearest neighbor and fitting the splines


def find_closest_point(current_point, points, used_indices, resolution):
    min_distance = float("inf")
    max_distance = resolution * 2 * np.sqrt(2)
    closest_point_index = -1
    for i, point in enumerate(points):
        if i not in used_indices:
            distance = np.linalg.norm(np.array(current_point) - np.array(point))
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_point_index = i
    return closest_point_index


# geting 1000 points from a tck for a specific time and slice k
def get_points_from_tck(tck, k):
    tck_k = tck[k]
    points = splev(np.linspace(0, 1, 100), tck_k)
    return points


def closest_point_to_line(coefficients, coords):
    """
    Finds the point closest to the given line from a set of coordinates.

    :param coefficients: Coefficients of the line (slope m and intercept b).
    :param coords: A numpy array of shape (m, 2) containing m points (x, y).
    :return: Coordinates of the point closest to the line.
    """
    m, b = coefficients

    def perpendicular_distance(x0, y0):
        """Calculate the perpendicular distance from a point to the line."""
        return abs(m * x0 - y0 + b) / np.sqrt(m**2 + 1)

    distances = np.apply_along_axis(
        lambda point: perpendicular_distance(point[0], point[1]), 1, coords
    )
    min_distance_index = np.argmin(distances)
    return coords[min_distance_index]


def bspline_angle_intersection(
    tck, angle_deg, side="upper", plt_flag="False", center=None
):
    """
    Finds the point of the intersection between the bspline and a line from the center of coords oriented at angle

    :param tck:             A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    :param angle_deg:       Angle of the line from x axis.
    :param side:            As the the bspline is a closed loop the intersection will be two points, this parameter indicate the 'upper' or 'lower' side should be returned.
    :param plt:             To plot the line and bspline
    :return:                Coordinates of the intersection point.
    """
    coords = splev(np.linspace(0, 1, 1000), tck)
    coords = np.array(coords).T
    path = Path(coords[:,:2])
    is_inside = path.contains_point(center)
    if center is None or not is_inside:
        center_coords = np.mean(coords[:,:2], axis=0)
        logger.warning(f"LV center is outside of the shax coordinates")
        logger.warning(f"LV center is shifted from {center} to {center_coords}")
    else:
        center_coords = center
    find_line_coefficients = lambda center_coords, theta: (
        math.tan(math.radians(theta)),
        center_coords[1] - math.tan(math.radians(theta)) * center_coords[0],
    )
    coefficients = find_line_coefficients(center_coords, angle_deg)
    if side == "upper":
        if angle_deg == 0.0:
            coords_side = coords[coords[:, 0] > center_coords[0]]
        elif angle_deg == 180.0:
            coords_side = coords[coords[:, 0] < center_coords[0]]
        else:
            coords_side = coords[coords[:, 1] > center_coords[1]]
    elif side == "lower":
        if angle_deg == 0.0:
            coords_side = coords[coords[:, 0] < center_coords[0]]
        elif angle_deg == 180.0:
            coords_side = coords[coords[:, 0] > center_coords[0]]
        else:
            coords_side = coords[coords[:, 1] < center_coords[1]]
    point = closest_point_to_line(coefficients, coords_side)
    if plt_flag:
        plt.plot(coords[:, 0], coords[:, 1])
        plt.plot(center_coords[0], center_coords[1], "go")
        plt.plot(point[0], point[1], "ro")
    return point


def sample_bspline_from_angles(tck, n_points, side="upper", center=None):
    points = np.zeros((n_points, 3))
    angles_deg = np.linspace(0, 180, n_points)
    for i in range(n_points):
        points[i, :] = bspline_angle_intersection(
            tck, angles_deg[i], side, plt_flag=False, center=center
        )
    return points


def get_n_points_from_shax(n_points, tck_k, LV_center):
    """
    Divide the shax into several slices with equal angles to get n_points, which should be an even number

    :param n_points:        The number of the points
    :param tck:             A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline for all the time and slices
    :param t,k:             the time and slice number
    :param LV_center:       The coords of the center
    :return:                Coordinates of the points.
    """
    points_upper = sample_bspline_from_angles(
        tck_k, int(n_points / 2) + 1, side="upper", center=LV_center
    )
    points_lower = sample_bspline_from_angles(
        tck_k, int(n_points / 2) + 1, side="lower", center=LV_center
    )
    points = np.vstack([points_upper[:-1], points_lower[:-1]])
    return points


def calculate_area_b_spline(tck):
    """
    Calculate the area enclosed by a closed B-spline curve with corrected tck format.

    :param tck:     A tuple representing the vector of knots, a 2D array of B-spline coefficients for x and y,
                    and the degree of the spline. It should be in the format (t, c, k) where:
                    - t is the vector of knots.
                    - c is a 2D array of B-spline coefficients, with the first row for x-coordinates and the second for y-coordinates.
                    - k is the degree of the spline.

    :return area:   The area enclosed by the curve.
    """
    t, c, k = tck
    c_x, c_y = c[0], c[1]

    # Check the compatibility of knots, coefficients, and degree
    if len(t) != len(c_x) + k + 1:
        raise ValueError("Knots, coefficients and degree are inconsistent.")

    num_points = 1000
    spline_x = BSpline(t, c_x, k)
    spline_y = BSpline(t, c_y, k)
    u = np.linspace(t[k], t[-k - 1], num_points)
    x = spline_x(u)
    y = spline_y(u)

    # Use the shoelace formula to calculate the area
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


def calculate_area_points(points):
    if points.ndim == 1:
        return 0
    else:
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        # Remove consecutive duplicates
        unique_points = points[np.diff(points, axis=0, prepend=np.nan).any(axis=1)]
        tck_tk, u_epi = splprep([unique_points[:, 0], unique_points[:, 1]], s=0, per=True, k=3)
        area = calculate_area_b_spline(tck_tk)
        return area


# ----------------------------------------------------------------
# ----------------------    Plotting    --------------------------
# ----------------------------------------------------------------
# Defining a function to overlay epi and endo edges on a mask (img)
def image_overlay(img, epi, endo):
    new_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:
                new_image[i, j] = [255, 255, 255]
            elif endo[i, j] != 0:
                new_image[i, j] = [0, 0, 255]
            elif epi[i, j] != 0:
                new_image[i, j] = [255, 0, 0]
    return new_image


def plot_spline(ax, tck):
    points = splev(np.linspace(0, 1, 1000), tck)
    ax.plot(points[0], points[1], "r-")
    return ax


def plot_shax_with_coords(coords, tck, k, new_plot=False, color="r"):
    if new_plot or not hasattr(plot_shax_with_coords, "fig"):
        plot_shax_with_coords.fig, plot_shax_with_coords.ax = plt.subplots()
        plot_shax_with_coords.ax.cla()  # Clear the previous plot if new_plot is True
    points = get_points_from_tck(tck, k)
    plot_shax_with_coords.ax.plot(points[0], points[1], color + "-")
    plot_shax_with_coords.ax.scatter(coords[k][:, 0], coords[k][:, 1], s=1, c=color)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.draw()


def plot_3d_SHAX(t, slice_thickness, tck_epi, tck_endo=None):
    """
    Plots the 3D SHAX spline shapes for a given t, where each k corresponds to a z location.
    Returns the matplotlib axis for further plotting.
    :param k:           Index for the specific set of data
    :param t:           Time index
    :param tck_epi:     epi splines data stored as (t_nurbs_epi,c_nurbs_epi,k_nurbs_epi)
    :param tck_endo:    endo splines data stored as (t_nurbs_endo,c_nurbs_endo,k_nurbs_endo)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    K = len(tck_epi[0][0])
    for k in range(K):
        z = -k * slice_thickness
        # Plot epicardial spline
        tck_epi_tk = (tck_epi[0][t][k], tck_epi[1][t][k], tck_epi[2][t][k])
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_tk)
        ax.plot(
            new_points_epi[0],
            new_points_epi[1],
            zs=z,
            zdir="z",
            label=f"Epi k={k}" if k == 0 else None,
            color="r",
        )
        # Plot endocardial spline if data is available
        if tck_endo and len(tck_endo[0][t]) > k:
            tck_endo_tk = (tck_endo[0][t][k], tck_endo[1][t][k], tck_endo[2][t][k])
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_tk)
            ax.plot(
                new_points_endo[0],
                new_points_endo[1],
                zs=z,
                zdir="z",
                label=f"Endo k={k}" if k == 0 else None,
                color="b",
            )

    ax.set_title(f"3D Spline Shapes (SHAX) for t={t}")

    return ax


def plot_3d_LAX(ax, n_list, tck_epi, tck_endo=None):
    """
    Plots the 3D LAX spline shapes for a given t, where n corresponds the list of curve numbers.
    :param k:           Index for the specific set of data
    :param t:           Time index
    :param tck_epi:     epi splines data stored as (t_nurbs_epi,c_nurbs_epi,k_nurbs_epi)
    :param tck_endo:    endo splines data stored as (t_nurbs_endo,c_nurbs_endo,k_nurbs_endo)
    """

    for n in n_list:
        # Plot epicardial spline
        tck_epi_n = tck_epi[n]
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_n)
        ax.plot(
            new_points_epi[0], new_points_epi[1], new_points_epi[2], zdir="z", color="r"
        )
        # Plot endocardial spline if data is available
        if tck_endo:
            tck_endo_n = tck_endo[n]
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_n)
            ax.plot(
                new_points_endo[0],
                new_points_endo[1],
                new_points_epi[2],
                zdir="z",
                color="b",
            )

    ax.set_title(f"3D Spline Shapes (LAX)")

    return ax

def plotly_3d_LAX(fig, n_list, tck_epi, tck_endo=None):
    """
    Plots the 3D LAX spline shapes for a given t, where n corresponds the list of curve numbers.
    :param k:           Index for the specific set of data
    :param t:           Time index
    :param tck_epi:     epi splines data stored as (t_nurbs_epi,c_nurbs_epi,k_nurbs_epi)
    :param tck_endo:    endo splines data stored as (t_nurbs_endo,c_nurbs_endo,k_nurbs_endo)
    """

    for n in n_list:
        # Plot epicardial spline
        tck_epi_n = tck_epi[n]
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_n)
        fig.add_trace(
            go.Scatter3d(
                x=new_points_epi[0],
                y=new_points_epi[1],
                z=new_points_epi[2],
                showlegend=False,
                mode="lines",
                name=f"SHAX Epi n={n}",
                line=dict(color="red"),
            )
        )
        # Plot endocardial spline if data is available
        if tck_endo:
            tck_endo_n = tck_endo[n]
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_n)
            fig.add_trace(
            go.Scatter3d(
                x=new_points_endo[0],
                y=new_points_endo[1],
                z=new_points_endo[2],
                showlegend=False,
                mode="lines",
                name=f"SHAX Endo n={n}",
                line=dict(color="red"),
            )
        )

    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=2)))
    return fig

def plotly_3d_contours(
    fig, tck_shax_epi, tck_lax_epi, tck_shax_endo=None, tck_lax_endo=None
):
    k_shax = len(tck_shax_epi)
    for k in range(k_shax):
        tck_epi_k = tck_shax_epi[k]
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_k)
        fig.add_trace(
            go.Scatter3d(
                x=new_points_epi[0],
                y=new_points_epi[1],
                z=new_points_epi[2],
                showlegend=False,
                mode="lines",
                name=f"SHAX Epi k={k}",
                line=dict(color="red"),
            )
        )
        if tck_shax_endo and len(tck_shax_endo) > k:
            tck_endo_k = tck_shax_endo[k]
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_k)
            fig.add_trace(
                go.Scatter3d(
                    x=new_points_endo[0],
                    y=new_points_endo[1],
                    z=new_points_endo[2],
                    showlegend=False,
                    mode="lines",
                    name=f"SHAX Epi k={k}",
                    line=dict(color="blue"),
                )
            )

    # Plot LAX splines
    n_lax = len(tck_lax_epi)
    for n in range(n_lax):
        lax_tck_epi_n = tck_lax_epi[n]
        lax_new_points_epi = splev(np.linspace(0, 1, 1000), lax_tck_epi_n)
        fig.add_trace(
            go.Scatter3d(
                x=lax_new_points_epi[0],
                y=lax_new_points_epi[1],
                z=lax_new_points_epi[2],
                showlegend=False,
                mode="lines",
                name=f"LAX Epi {n}",
                line=dict(color="red"),
            )
        )
        if tck_lax_endo:
            lax_tck_endo_n = tck_lax_endo[n]
            lax_new_points_endo = splev(np.linspace(0, 1, 1000), lax_tck_endo_n)
            fig.add_trace(
                go.Scatter3d(
                    x=lax_new_points_endo[0],
                    y=lax_new_points_endo[1],
                    z=lax_new_points_endo[2],
                    showlegend=False,
                    mode="lines",
                    name=f"LAX Endo {n}",
                    line=dict(color="blue"),
                )
            )

    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=2)))
    return fig


def plot_delaunay_3d_plotly(simplices, points):
    fig = go.Figure()

    # Adding triangles (simplices) to the plot
    for simplex in simplices:
        x = [points[simplex[i]][0] for i in range(3)] + [points[simplex[0]][0]]
        y = [points[simplex[i]][1] for i in range(3)] + [points[simplex[0]][1]]
        z = [points[simplex[i]][2] for i in range(3)] + [points[simplex[0]][2]]

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color="blue", opacity=0.50))

    # Setting labels and titles
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Delaunay Triangulation",
    )

    fig.show()


def plot_delaunay_2d(vertices, points):
    fig, ax = plt.subplots()

    # Plotting the Delaunay triangles
    for simplex in vertices:
        triangle = [
            points[simplex[0]],
            points[simplex[1]],
            points[simplex[2]],
            points[simplex[0]],
        ]  # Closing the triangle
        triangle_array = np.array(triangle)
        ax.plot(
            triangle_array[:, 0], triangle_array[:, 1], "k-"
        )  # 'k-' means black line

    # Setting the limits of the plot based on the points
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Delaunay Triangulation")

    plt.show()


def plot_delaunay_3d(vertices, points):
    # Perform Delaunay triangulation
    # tri = Delaunay(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plotting the Delaunay triangles
    for simplex in vertices:
        triangle = [
            points[simplex[0]],
            points[simplex[1]],
            points[simplex[2]],
            points[simplex[0]],
        ]  # Closing the triangle
        ax.add_collection3d(Poly3DCollection([triangle]))

    # Setting the limits of the plot based on the points
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))
    ax.set_zlim(np.min(points[:, 2]), np.max(points[:, 2]))

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Delaunay Triangulation")

    plt.show()


def plot_3d_points_on_figure(data_array, fig=None):
    """
    Plot 3D points from a NumPy array on a given Plotly figure.
    If no figure is provided, a new one is created.

    Parameters:
    data_array (np.array): A NumPy array with shape (n, 3) where n is the number of points.
                           Each row represents a point in 3D space (x, y, z).
    fig (go.Figure): Optional. A Plotly figure object to plot the points on.
    """

    # Check if figure is provided, else create a new one
    if fig is None:
        fig = go.Figure()

    # Extract x, y, z coordinates
    if len(data_array.shape) == 1:
        x = np.array(data_array[0])
        y = np.array(data_array[1])
        z = np.array(data_array[2])
    else:
        x = data_array[:, 0]
        y = data_array[:, 1]
        z = data_array[:, 2]

    # Add the points to the figure
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.8,
            ),
        )
    )

    # Set plot layout if this is the first plot
    if len(fig.data) == 1:
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

    return fig



def plotly_3d_base_splines(tck_layers, fig=None, color = "black"):
    
    # Check if figure is provided, else create a new one
    if fig is None:
        fig = go.Figure()
    
    for tck in tck_layers:
        # Plot epicardial spline
        points = splev(np.linspace(0, 1, 1000), tck)
        fig.add_trace(
            go.Scatter3d(
                x=points[0],
                y=points[1],
                z=points[2],
                showlegend=False,
                mode="lines",
                line=dict(color=color),
            )
        )
    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=2)))
    return fig



def plot_coords_and_mesh(coords_epi, coords_endo, mesh_epi_filename, mesh_endo_filename, plot_edges=True, fig=None):
    """
    Plots the coords_epi, coords_endo, mesh_epi, and mesh_endo in Plotly with line colors.

    Parameters:
    - coords_epi: list of arrays, each array containing the coordinates from a slice.
                  Each array is of shape (n_points, 3).
    - coords_endo: list of arrays, similar to coords_epi.
    - mesh_epi: mesh.Mesh object (from stl.mesh import Mesh)
    - mesh_endo: mesh.Mesh object (from stl.mesh import Mesh)
    """
    from stl import mesh
    
    mesh_epi = mesh.Mesh.from_file(mesh_epi_filename)
    mesh_endo = mesh.Mesh.from_file(mesh_endo_filename)
    
    # Check if figure is provided, else create a new one
    if fig is None:
        fig = go.Figure()
    
    # Plot coords_epi
    for k, coords_k in enumerate(coords_epi):
        if coords_k.shape[0] == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=coords_k[:, 0],
            y=coords_k[:, 1],
            z=coords_k[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name=f'Coords Epi Slice {k}'
        ))

    # Plot coords_endo
    for k, coords_k in enumerate(coords_endo):
        if coords_k.shape[0] == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=coords_k[:, 0],
            y=coords_k[:, 1],
            z=coords_k[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue'),
            name=f'Coords Endo Slice {k}'
        ))

    # Prepare mesh_epi for plotting
    vectors_epi = mesh_epi.vectors  # Shape (n_facets, 3, 3)
    vertices_epi = vectors_epi.reshape(-1, 3)
    # Remove duplicate vertices
    vertices_epi_unique, index_unique_epi = np.unique(vertices_epi, axis=0, return_inverse=True)
    faces_epi = index_unique_epi.reshape(-1, 3)
    
    # Plot mesh_epi with faces
    fig.add_trace(go.Mesh3d(
        x=vertices_epi_unique[:, 0],
        y=vertices_epi_unique[:, 1],
        z=vertices_epi_unique[:, 2],
        i=faces_epi[:, 0],
        j=faces_epi[:, 1],
        k=faces_epi[:, 2],
        color='red',
        opacity=0.5,
        name='Mesh Epi',
        flatshading=True
    ))

    # Plot mesh_epi edges
    if plot_edges:
        fig = add_mesh_edges_to_figure(fig, vertices_epi_unique, faces_epi, line_color='black', name='Mesh Epi Edges')

    # Prepare mesh_endo for plotting
    vectors_endo = mesh_endo.vectors  # Shape (n_facets, 3, 3)
    vertices_endo = vectors_endo.reshape(-1, 3)
    # Remove duplicate vertices
    vertices_endo_unique, index_unique_endo = np.unique(vertices_endo, axis=0, return_inverse=True)
    faces_endo = index_unique_endo.reshape(-1, 3)
    
    # Plot mesh_endo with faces
    fig.add_trace(go.Mesh3d(
        x=vertices_endo_unique[:, 0],
        y=vertices_endo_unique[:, 1],
        z=vertices_endo_unique[:, 2],
        i=faces_endo[:, 0],
        j=faces_endo[:, 1],
        k=faces_endo[:, 2],
        color='blue',
        opacity=0.5,
        name='Mesh Endo',
        flatshading=True
    ))

    # Plot mesh_endo edges
    if plot_edges:
        fig = add_mesh_edges_to_figure(fig, vertices_endo_unique, faces_endo, line_color='black', name='Mesh Endo Edges')

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Coords and Mesh Visualization with Edge Lines'
    )

    return fig

def add_mesh_edges_to_figure(fig, vertices, faces, line_color='black', name='Mesh Edges'):
    """
    Adds mesh edges as lines to the figure.

    Parameters:
    - fig: Plotly Figure object to add the edges to.
    - vertices: Array of vertices, shape (n_vertices, 3).
    - faces: Array of face indices, shape (n_faces, 3).
    - line_color: Color of the lines.
    - name: Name of the trace in the legend.
    """
    
    # Create a set to store unique edges
    edges = set()

    # Iterate over faces and extract edges
    for face in faces:
        # For each face (triangle), get all edge pairs
        edge_pairs = [(face[i], face[(i + 1) % 3]) for i in range(3)]
        for edge in edge_pairs:
            # Sort the vertices indices to avoid duplicates (e.g., (1,2) and (2,1))
            sorted_edge = tuple(sorted(edge))
            edges.add(sorted_edge)
    
    # Prepare lists to hold edge coordinates
    x_edges = []
    y_edges = []
    z_edges = []

    # Build the coordinates for the edges
    for edge in edges:
        x_coords = [vertices[edge[0], 0], vertices[edge[1], 0], None]
        y_coords = [vertices[edge[0], 1], vertices[edge[1], 1], None]
        z_coords = [vertices[edge[0], 2], vertices[edge[1], 2], None]
        x_edges.extend(x_coords)
        y_edges.extend(y_coords)
        z_edges.extend(z_coords)

    # Add the edges as a Scatter3d line trace
    fig.add_trace(go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color=line_color, width=1),
        name=name,
        showlegend=False
    ))

    return fig



def calculate_error_between_coords_and_mesh(coords, stl_mesh_filename):
    import trimesh
    from stl import mesh
    
    stl_mesh = mesh.Mesh.from_file(stl_mesh_filename)
    # Extract vertices and faces
    vectors = stl_mesh.vectors  # Shape (n_facets, 3, 3)
    vertices = vectors.reshape(-1, 3)  # Flatten vertices
    unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)

    # Create a trimesh object
    trimesh_mesh = trimesh.Trimesh(vertices=unique_vertices, faces=faces, process=False)

    # Compute the signed distance from points to mesh
    coords_array = np.vstack(coords)
    distances = trimesh.proximity.signed_distance(trimesh_mesh, coords_array)

    errors = np.abs(distances)
    
    return errors


def plot_error_histogram(errors, fname, color, xlim, ylim, title_prefix, resolution=""):

    avg_error = np.mean(errors)
    std_error  = np.std(errors)
    if not resolution == "":
        resolution = np.round(resolution,3)
        line = f'{title_prefix} Error Distribution (Avg: {avg_error:.2f} ± {std_error:.2f}) - (Data Res:{resolution}) '
    else:
        line = f'{title_prefix} Error Distribution (Avg: {avg_error:.2f} ± {std_error:.2f})'
    # Plot error distribution histogram
    plt.figure()
    plt.hist(errors, bins=30, edgecolor='black', color=color)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(line)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(fname)
    plt.close()

def save_error_distribution_report(errors, file_path, n_bins=10, surface_name = "", resolution=""):
    # Generate histogram data
    counts, bin_edges = np.histogram(errors, bins=n_bins, range=(np.min(errors), np.max(errors)))
    file = open(file_path, 'w')
    line = "======= Mesh Statistics ======="
    file.write(line + '\n')
    if not surface_name == "":
        line = f"{surface_name}"
        if not resolution == "":
            resolution = np.round(resolution,3)
            line = f"{surface_name} with image resolution of {resolution}"
        file.write(line + '\n')
    line = f"Original coords vs surface mesh error distribution report:"
    file.write(line + '\n')
    line = "-------------------------------"
    file.write(line + '\n')
    total_errors = len(errors)
    cumulative_percentage = 0
    for i in range(n_bins):
        bin_count = counts[i]
        bin_percentage = (bin_count / total_errors) * 100
        cumulative_percentage += bin_percentage
        line = (f"{bin_edges[i]:.4f} ≤ error < {bin_edges[i+1]:.4f} : "
                f"{bin_count:>10} instances ({bin_percentage:6.2f}%), "
                f"cumulative: {cumulative_percentage:6.2f}%")
        file.write(line + '\n')
    file.close()
