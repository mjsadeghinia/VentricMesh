#%% The mask file should be an np.array of size K*I*I*T, where K is the slices and I is the image size of each slice and T is the time. 
from mesh_utils import *
from utils import *
import h5py

sample_ID='example_mask'
data_folder=get_path()+'/00_Data/'+sample_ID
result_folder=get_path()+'/00_Data/'+sample_ID+'/01_SHAX/'

with h5py.File(sample_ID+'.h5', "r") as f:
    # Read datasets
    LVmask = f["LVmask_closed"][:]
    T_array = f["T_array"][:]
    T_end=T_array.shape[0]
    metadata = {key: value for key, value in f.attrs.items()}
    slice_thickness=metadata['slice_thickness'] # in mm
    resolution=metadata['resolution']*10        # to convert to mm/pixel



num_z_sections_endo=20
num_z_sections_epi=20
seed_num_base_endo=12
seed_num_base_epi=20
num_mid_layers_base=2
t_mesh=-1
filename_suffix='ED'

points_cloud_epi,points_cloud_endo=NodeGenerator(LVmask,resolution,slice_thickness,seed_num_base_epi,seed_num_base_endo,num_z_sections_epi,num_z_sections_endo)
LVmesh=VentricMesh(points_cloud_epi,points_cloud_endo,t_mesh,num_mid_layers_base,filename_suffix,result_folder='')



#%% Then we use gmesh to create the 3D mesh from the generated 3D surface mesh
import gmsh
def generate_3d_mesh_from_stl(stl_path, mesh_path):
    gmsh.initialize()
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
    gmsh.model.mesh.Algorithm    = 1 # (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=bamg, 8=delquad) (Default=2)
    gmsh.model.mesh.Algorithm3D    = 4 # (1=Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree) (Default=1)
    gmsh.model.mesh.Recombine3DAll = 0
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_path)
    gmsh.finalize()
    
generate_3d_mesh_from_stl('Mesh_'+filename_suffix+'.stl', 'Mesh_'+filename_suffix+'_3D.msh')

# %%
