"""
Useful functions for creating the mesh

Last Modified: 18.01.2024
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep
import math
import plotly.graph_objects as go
from scipy.interpolate import BSpline

#----------------------------------------------------------------
#------------------------    Path    ----------------------------
#---------------------------------------------------------------- 
def get_path():
    """Get the full path of the current notebook or script."""
    try:
        # This will work for Jupyter notebooks
        notebook_path = IPython.core.getipython().getoutput('pwd')[0]
        return notebook_path
    except:
        # This will work for Python scripts
        script_path = os.path.abspath(__file__)
        script_folder=os.path.dirname(script_path)
        return script_folder
    
    
#----------------------------------------------------------------
#----------------------    Utilities    -------------------------
#---------------------------------------------------------------- 


# Definig function is_connected to checking if there is unconnected pixels
# from a long axis view. Here a function is defined to check if there is any gap
# in the image or not
def is_connected(matrix):
    flag=False
    visited = set()
    visited_reversed = set()
    rows, cols = len(matrix), len(matrix[0])
    # Defining depth first search (DFS) for traversing the pixels It gets the r
    # and c of a True value pixel and looks for the neighboutring pixels to add
    # it to the set of visited. the flg is a flag for wether it should it use
    # visited (1) or visited_reversed (0) set. The latter is used in case of an opening.
    #  
    def dfs(r, c,flg):
        if flg:
            if 0 <= r < rows and 0 <= c < cols and matrix[r][c] and (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc,1)
        else:
            if 0 <= r < rows and 0 <= c < cols and matrix[r][c] and (r, c) not in visited_reversed:
                visited_reversed.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc,0)

    # Find first True value and start DFS from there
    for r in range(rows):
        if flag:
            break
        for c in range(cols):
            if matrix[r][c]:
                dfs(r, c,1)
                flag=True
                break
        
    # Check for any unvisited True values
    indices = np.argwhere(matrix)
    index_set = set(map(tuple, indices))
    if visited==index_set:
        return True, visited, visited_reversed
    else:
        flag=False
        for c in range(cols-1, -1, -1):
            if flag:
                break
            for r in range(rows):
                if matrix[r][c] and (r, c) not in visited:
                    dfs(r, c,0)
                    flag=True
                    break

        return False, visited, visited_reversed
    

#Getting a coordinates of a binary image with correct value So when ploting it would be similiar to the image itself
def coords_from_img(img,resolution,I):
    temp=np.argwhere(img == 1)*resolution 
    coords=np.zeros((len(temp),2))
    coords[:,0]=temp[:,1]
    coords[:,1]=temp[:,0]
    coords[:,1]=I*resolution-coords[:,1]
    return coords


# function to sort the coords based on the closest neighbouring point. This makes it possible to fit using nurbs
def sorting_coords(coords):
    points = [list(coord) for coord in coords]
    coords_sorted = [points[0]]
    used_indices = {0}
    for _ in range(1, len(points)):
        current_point = coords_sorted[-1]
        closest_point_index = find_closest_point(current_point, points, used_indices)
        if closest_point_index >= 0:
            coords_sorted.append(points[closest_point_index])
            used_indices.add(closest_point_index)
    return np.array(coords_sorted)

# functions for sorting the points based on the nearest neighbor and fitting the splines

def find_closest_point(current_point, points, used_indices):
    min_distance = float('inf')
    closest_point_index = -1
    for i, point in enumerate(points):
        if i not in used_indices:
            distance = np.linalg.norm(np.array(current_point) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i
    return closest_point_index

# geting 1000 points from a tck for a specific time and slice k
def get_points_from_tck(tck,t,k):
    tck_tk = (tck[0][t][k], tck[1][t][k], tck[2][t][k])
    points = splev(np.linspace(0, 1, 100), tck_tk)
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
    
    distances = np.apply_along_axis(lambda point: perpendicular_distance(point[0], point[1]), 1, coords)
    min_distance_index = np.argmin(distances)
    return coords[min_distance_index]

def bspline_angle_intersection(tck,angle_deg, side='upper', plt_flag='False', center=None):
    """
    Finds the point of the intersection between the bspline and a line from the center of coords oriented at angle

    :param tck:             A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    :param angle_deg:       Angle of the line from x axis.
    :param side:            As the the bspline is a closed loop the intersection will be two points, this parameter indicate the 'upper' or 'lower' side should be returned.
    :param plt:             To plot the line and bspline
    :return:                Coordinates of the intersection point.
    """
    coords = splev(np.linspace(0, 1, 1000), tck)
    coords=np.array(coords).T
    if center is None:
        center_coords = np.mean(coords, axis=0)
    else:
        center_coords=center
    find_line_coefficients = lambda center_coords, theta: (math.tan(math.radians(theta)), center_coords[1] - math.tan(math.radians(theta)) * center_coords[0])
    coefficients = find_line_coefficients(center_coords,angle_deg)
    if side=='upper':
        if angle_deg==0.0:
            coords_side = coords[coords[:, 0] > center_coords[0]]
        elif angle_deg==180.0:
            coords_side = coords[coords[:, 0] < center_coords[0]]
        else:
            coords_side = coords[coords[:, 1] > center_coords[1]]
    elif side=='lower':
        if angle_deg==0.0:
            coords_side = coords[coords[:, 0] < center_coords[0]]
        elif angle_deg==180.0:
            coords_side = coords[coords[:, 0] > center_coords[0]]
        else:
            coords_side = coords[coords[:, 1] < center_coords[1]]
    point=closest_point_to_line(coefficients, coords_side)
    if plt_flag==True:
        plt.plot(coords[:,0],coords[:,1])
        plt.plot(center_coords[0],center_coords[1],'go')
        plt.plot(point[0],point[1],'ro')
    return point

def sample_bspline_from_angles(tck,n_points,side='upper', center=None):  
    points=np.zeros((n_points,3))      
    angles_deg=np.linspace(0,180,n_points)
    for i in range(n_points):
        points[i,:]=bspline_angle_intersection(tck,angles_deg[i], side, plt_flag=False,center=center)
    return points

def get_n_points_from_shax(n_points,tck,t,k,LV_center):
    """
    Divide the shax into several slices with equal angles to get n_points, which should be an even number

    :param n_points:        The number of the points
    :param tck:             A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline for all the time and slices
    :param t,k:             the time and slice number
    :param LV_center:       The coords of the center
    :return:                Coordinates of the points.
    """
    tck_tk = (tck[0][t][k], tck[1][t][k], tck[2][t][k])
    points_upper=sample_bspline_from_angles(tck_tk,int(n_points/2)+1,side='upper',center=LV_center)    
    points_lower=sample_bspline_from_angles(tck_tk,int(n_points/2)+1,side='lower',center=LV_center)
    points=np.vstack([points_upper[:-1],points_lower[:-1]])
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
    u = np.linspace(t[k], t[-k-1], num_points)
    x = spline_x(u)
    y = spline_y(u)

    # Use the shoelace formula to calculate the area
    area = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


def calculate_area_points(points):
    tck_tk, u_epi = splprep([points[:,0],points[:,1]], s=1, per=True, k=3) 
    area=calculate_area_b_spline(tck_tk) 
    return area
    
    
#----------------------------------------------------------------
#----------------------    Plotting    --------------------------
#----------------------------------------------------------------   
#Defining a function to overlay epi and endo edges on a mask (img)
def image_overlay(img,epi,endo):
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

def plot_spline(ax,tck):
    points = splev(np.linspace(0, 1, 1000), tck)
    ax.plot(points[0], points[1], 'r-')
    return ax

def plot_shax_with_coords(ax,mask,tck,t,k,resolution):
    K, I, _, T_total = mask.shape
    coords=coords_from_img(mask[k,:,:,t],resolution,I)
    points=get_points_from_tck(tck,t,k)
    ax.plot(points[0], points[1], 'r-')
    ax.plot(coords[:,0], coords[:,1],'ro',markersize=1)
    plt.gca().set_aspect('equal', adjustable='box')


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
    ax = fig.add_subplot(111, projection='3d')

    for k in range(10):
        z = -k * slice_thickness
        # Plot epicardial spline
        tck_epi_tk = (tck_epi[0][t][k], tck_epi[1][t][k], tck_epi[2][t][k])
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_tk)
        ax.plot(new_points_epi[0], new_points_epi[1], zs=z, zdir='z', label=f'Epi k={k}' if k == 0 else None, color='r')
        # Plot endocardial spline if data is available
        if tck_endo and len(tck_endo[0][t]) > k:
            tck_endo_tk = (tck_endo[0][t][k], tck_endo[1][t][k], tck_endo[2][t][k])
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_tk)
            ax.plot(new_points_endo[0], new_points_endo[1], zs=z, zdir='z', label=f'Endo k={k}' if k == 0 else None, color='b')
            
    ax.set_title(f"3D Spline Shapes (SHAX) for t={t}")

    return ax

def plot_3d_LAX(ax,t,n_list, tck_epi, tck_endo=None):
    """
    Plots the 3D LAX spline shapes for a given t, where n corresponds the list of curve numbers.
    :param k:           Index for the specific set of data
    :param t:           Time index
    :param tck_epi:     epi splines data stored as (t_nurbs_epi,c_nurbs_epi,k_nurbs_epi)
    :param tck_endo:    endo splines data stored as (t_nurbs_endo,c_nurbs_endo,k_nurbs_endo)
    """

    for n in n_list:
        # Plot epicardial spline
        tck_epi_tk = (tck_epi[0][t][n], tck_epi[1][t][n], tck_epi[2][t][n])
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_tk)
        ax.plot(new_points_epi[0], new_points_epi[1], new_points_epi[2], zdir='z', color='r')
        # Plot endocardial spline if data is available
        if tck_endo:
            tck_endo_tk = (tck_endo[0][t][n], tck_endo[1][t][n], tck_endo[2][t][n])
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_tk)
            ax.plot(new_points_endo[0], new_points_endo[1], new_points_epi[2], zdir='z', color='b')
            
    ax.set_title(f"3D Spline Shapes (LAX) for t={t}")

    return ax

def plotly_3d_contours(fig, t, tck_shax_epi, tck_lax_epi, tck_shax_endo=None, tck_lax_endo=None):
    
    k_shax = len(tck_shax_epi[0][0])
    for k in range(k_shax):
        tck_epi_tk = (tck_shax_epi[0][t][k], tck_shax_epi[1][t][k], tck_shax_epi[2][t][k])
        new_points_epi = splev(np.linspace(0, 1, 1000), tck_epi_tk)
        fig.add_trace(go.Scatter3d(x=new_points_epi[0], y=new_points_epi[1], z=new_points_epi[2],showlegend=False, mode='lines', name=f'SHAX Epi k={k}', line=dict(color='red')))
        if tck_shax_endo and len(tck_shax_endo[0][t]) > k:
            tck_endo_tk = (tck_shax_endo[0][t][k], tck_shax_endo[1][t][k], tck_shax_endo[2][t][k])
            new_points_endo = splev(np.linspace(0, 1, 1000), tck_endo_tk)
            fig.add_trace(go.Scatter3d(x=new_points_endo[0], y=new_points_endo[1], z=new_points_endo[2],showlegend=False, mode='lines', name=f'SHAX Epi k={k}', line=dict(color='blue')))
    
    
    # Plot LAX splines
    n_lax = len(tck_lax_epi[0][0])
    for n in range(n_lax):
        lax_tck_epi_tk = (tck_lax_epi[0][t][n], tck_lax_epi[1][t][n], tck_lax_epi[2][t][n])
        lax_new_points_epi = splev(np.linspace(0, 1, 1000), lax_tck_epi_tk)
        fig.add_trace(go.Scatter3d(x=lax_new_points_epi[0], y=lax_new_points_epi[1], z=lax_new_points_epi[2],showlegend=False, mode='lines', name=f'LAX Epi {n}', line=dict(color='red')))
        if tck_lax_endo:
            lax_tck_endo_tk = (tck_lax_endo[0][t][n], tck_lax_endo[1][t][n], tck_lax_endo[2][t][n])
            lax_new_points_endo = splev(np.linspace(0, 1, 1000), lax_tck_endo_tk)
            fig.add_trace(go.Scatter3d(x=lax_new_points_endo[0], y=lax_new_points_endo[1], z=lax_new_points_endo[2],showlegend=False, mode='lines', name=f'LAX Endo {n}', line=dict(color='blue')))

    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=2)))
    return fig


def plot_delaunay_3d_plotly(simplices, points):
    fig = go.Figure()

    # Adding triangles (simplices) to the plot
    for simplex in simplices:
        x = [points[simplex[i]][0] for i in range(3)] + [points[simplex[0]][0]]
        y = [points[simplex[i]][1] for i in range(3)] + [points[simplex[0]][1]]
        z = [points[simplex[i]][2] for i in range(3)] + [points[simplex[0]][2]]
        
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color='blue',
            opacity=0.50
        ))

    # Setting labels and titles
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Delaunay Triangulation'
    )

    fig.show()

def plot_delaunay_2d(vertices, points):
    fig, ax = plt.subplots()

    # Plotting the Delaunay triangles
    for simplex in vertices:
        triangle = [points[simplex[0]], points[simplex[1]], points[simplex[2]], points[simplex[0]]]  # Closing the triangle
        triangle_array = np.array(triangle)
        ax.plot(triangle_array[:, 0], triangle_array[:, 1], 'k-')  # 'k-' means black line

    # Setting the limits of the plot based on the points
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Delaunay Triangulation')

    plt.show()
    
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_delaunay_3d(vertices,points):
    # Perform Delaunay triangulation
    # tri = Delaunay(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the Delaunay triangles
    for simplex in vertices:
        triangle = [points[simplex[0]], points[simplex[1]], points[simplex[2]], points[simplex[0]]]  # Closing the triangle
        ax.add_collection3d(Poly3DCollection([triangle]))

    # Setting the limits of the plot based on the points
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))
    ax.set_zlim(np.min(points[:, 2]), np.max(points[:, 2]))

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Delaunay Triangulation')

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
    x = data_array[:, 0]
    y = data_array[:, 1]
    z = data_array[:, 2]

    # Add the points to the figure
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
        )
    ))

    # Set plot layout if this is the first plot
    if len(fig.data) == 1:
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

    return fig
