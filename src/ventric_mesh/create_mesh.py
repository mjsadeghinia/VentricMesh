#%%

import h5py
import os
# from pathlib import Path
import argparse

from .mesh_utils import *
from .utils import *

# here = Path(__file__).absolute().parent

#%%
def read_data_h5(file_dir):
    with h5py.File(file_dir, "r") as f:
        # Read datasets
        LVmask = f["LVmask"][:]
        T = f["T"][:]
        metadata = {key: value for key, value in f.attrs.items()}
        slice_thickness=metadata['slice_thickness'] # in mm
        resolution=metadata['resolution']           # to convert to mm/pixel
    return LVmask,T,slice_thickness,resolution  

#%%
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        default='D3-2_sample.h5',
        help="the data file name",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default='ED',
        help="The filename suffix for the saved mesh, e.g., ED or ES",
    )
    parser.add_argument(
        "--num_z_sections_endo",
        default=15,
        type=int,
        help="The number of z slices (longitudinal) of the final mesh for endocardium",
    )
    parser.add_argument(
        "--num_z_sections_epi",
        default=20,
        type=int,
        help="The number of z slices (longitudinal) of the final mesh for epicardium",
    )
    parser.add_argument(
        "--seed_num_base_endo",
        default=12,
        type=int,
        help="The number nodes on the base of the endocardium",
    )
    parser.add_argument(
        "--seed_num_base_epi",
        default=20,
        type=int,
        help="The number nodes on the base of the epicardium",
    )
    parser.add_argument(
        "--num_mid_layers_base",
        default=2,
        type=int,
        help="The number middle layers between epicardum and endocardium on the base",
    )
    parser.add_argument(
        "-t",
        "--time",
        default=1,
        type=float,
        help="The normalized time of the cardiac cylce for the mesh, 0 is early systole and 1 is end diastole. ",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=os.getcwd()+'/',
        type=str,
        help="The output directory to save the files. ",
    )
    args = parser.parse_args()
    
    file_dir= os.getcwd()+'/'+args.filename
    num_z_sections_endo=args.num_z_sections_endo
    num_z_sections_epi=args.num_z_sections_epi
    seed_num_base_endo=args.seed_num_base_endo
    seed_num_base_epi=args.seed_num_base_epi
    num_mid_layers_base=args.num_mid_layers_base
    filename_suffix=args.suffix
    output_dir=os.getcwd()+args.output_dir+'/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    LVmask,T,slice_thickness,resolution=read_data_h5(file_dir)
    T_end=len(T)
    t_mesh=int((args.time)*T_end)-1
    
    points_cloud_epi,points_cloud_endo=NodeGenerator(LVmask,resolution,slice_thickness,seed_num_base_epi,seed_num_base_endo,num_z_sections_epi,num_z_sections_endo)
    LVmesh=VentricMesh(points_cloud_epi,points_cloud_endo,t_mesh,num_mid_layers_base,filename_suffix,result_folder=output_dir)
    generate_3d_mesh_from_stl(output_dir+'Mesh_'+filename_suffix+'.stl', output_dir+'Mesh_'+filename_suffix+'_3D.msh')
    
#%%   
if __name__ == "__main__":
    main()