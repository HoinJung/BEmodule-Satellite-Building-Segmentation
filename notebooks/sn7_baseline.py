
# Set prediction and image directories (edit appropriately)
pred_top_dir = '../inference_out/sn7_baseline_preds'
im_top_dir = './../../test_public'

from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import skimage.io
import tqdm
import glob
import math
import gdal
import time
import sys
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES']='2'

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib
# matplotlib.use('Agg') # non-interactive

import solaris as sol
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 

# import from data_postproc_funcs
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from sn7_baseline_postproc_funcs import map_wrapper, multithread_polys,         calculate_iou, track_footprint_identifiers,         sn7_convert_geojsons_to_csv


# --------
# ### 3.A. Group predictions by AOI

# In[2]:


raw_name = '201020_bear_only_dhdn_pixelwise_work/'
grouped_name = 'grouped/'
im_list = sorted([z for z in os.listdir(os.path.join(pred_top_dir, raw_name)) if z.endswith('.tif')])
df = pd.DataFrame({'image': im_list})
roots = [z.split('mosaic_')[-1].split('.tif')[0] for z in df['image'].values]
df['root'] = roots
# copy files
for idx, row in df.iterrows():
    in_path_tmp = os.path.join(pred_top_dir, raw_name, row['image'])
    out_dir_tmp = os.path.join(pred_top_dir, grouped_name, row['root'], 'masks')
    os.makedirs(out_dir_tmp, exist_ok=True)
    cmd = 'cp ' + in_path_tmp + ' ' + out_dir_tmp
    print("cmd:", cmd)
    os.system(cmd)    


# ------
# ## 3.C. Extract building footprint geometries for all AOIs

# In[3]:


# Get all geoms for all aois (mult-threaded)

min_area = 3.5   # in pixels (4 is standard)
simplify = False
bg_threshold = 0  
output_type = 'geojson'
aois = sorted([f for f in os.listdir(os.path.join(pred_top_dir, 'grouped')) if os.path.isdir(os.path.join(pred_top_dir, 'grouped', f))])


# set params
params = []
for i, aoi in enumerate(aois):
    print(i, "/", len(aois), aoi)   
    outdir = os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons')
    os.makedirs(outdir, exist_ok=True)
    pred_files = sorted([os.path.join(pred_top_dir, 'grouped', aoi, 'masks', f)
                for f in sorted(os.listdir(os.path.join(pred_top_dir, 'grouped', aoi, 'masks')))
                if f.endswith('.tif')])
    for j, p in enumerate(pred_files):
        name = os.path.basename(p)
        # print(i, j, name)
        output_path_pred = os.path.join(outdir,  name.split('.tif')[0] + '.geojson')
        # get pred geoms
        if not os.path.exists(output_path_pred):
            pred_image = skimage.io.imread(p)#[:,:,0]
            params.append([pred_image, min_area, output_path_pred,
                          output_type, bg_threshold, simplify])        

print("Execute!")
print("len params:", len(params))
n_threads = 10
pool = multiprocessing.Pool(n_threads)
_ = pool.map(multithread_polys, params)


# ----------
# ## 3.D. Track building identifiers
# 
# Now we assign a unique identifier to each building, and propogate that identifier through the data cube.

# In[4]:


# This takes awhile, so multi-thread it

min_iou = 0.2
iou_field = 'iou_score'
id_field = 'Id'
reverse_order = False
verbose = True
super_verbose = False
n_threads = 10

json_dir_name = 'pred_jsons/'
out_dir_name = 'pred_jsons_match/'
aois = sorted([f for f in os.listdir(os.path.join(pred_top_dir, 'grouped')) 
               if os.path.isdir(os.path.join(pred_top_dir, 'grouped', f))])
print("aois:", aois)

print("Gather data for matching...")
params = []
for aoi in aois:
    print(aoi)
    json_dir = os.path.join(pred_top_dir, 'grouped', aoi, json_dir_name)
    out_dir = os.path.join(pred_top_dir, 'grouped', aoi, out_dir_name)
    
    # check if we started matching...
    if os.path.exists(out_dir):
        # print("  outdir exists:", outdir)
        json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))])
        out_files_tmp = sorted([z for z in os.listdir(out_dir) if z.endswith('.geojson')])
        if len(out_files_tmp) > 0:
            if len(out_files_tmp) == len(json_files):
                print("Dir:", os.path.basename(out_dir), "N files:", len(json_files), 
                      "directory matching completed, skipping...")
                continue
            elif len(out_files_tmp) != len(json_files):
                # raise Exception("Incomplete matching in:", out_dir, "with N =", len(out_files_tmp), 
                #                 "files (should have N_gt =", 
                #                 len(json_files), "), need to purge this folder and restart matching!")
                print("Incomplete matching in:", out_dir, "with N =", len(out_files_tmp), 
                                "files (should have N_gt =", 
                                len(json_files), "), purging this folder and restarting matching!")
                purge_cmd = 'rm -r ' + out_dir
                print("  purge_cmd:", purge_cmd)
                if len(out_dir) > 20:
                    purge_cmd = 'rm -r ' + out_dir
                else:
                    raise Exception("out_dir too short, maybe deleting something unintentionally...")
                    break
                os.system(purge_cmd)
            else:
                pass

    params.append([track_footprint_identifiers, json_dir,  out_dir, min_iou, 
                   iou_field, id_field, reverse_order, verbose, super_verbose])    

print("Len params:", len(params))


# In[5]:


print("Execute!")
n_threads = 10
pool = multiprocessing.Pool(n_threads)
_ = pool.map(map_wrapper, params)


# In[ ]:


# --------
# ## 3.E. Make proposal CSV 
# This is necessary for scoring with the [SCOT metric](https://github.com/CosmiQ/solaris/blob/master/solaris/eval/scot.py).

# In[ ]:


# Make proposal csv

out_dir_csv = os.path.join(pred_top_dir, 'csvs')
os.makedirs(out_dir_csv, exist_ok=True)
prop_file = os.path.join(out_dir_csv, 'solution.csv')

aoi_dirs = sorted([os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match')                    for aoi in os.listdir(os.path.join(pred_top_dir, 'grouped'))                    if os.path.isdir(os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match'))])
print("aoi_dirs:", aoi_dirs)

# Execute
if not os.path.exists(prop_file):
    net_df = sn7_convert_geojsons_to_csv(aoi_dirs, prop_file, 'proposal')

print("prop_file:", prop_file)


# --------
# 
# ## 4. Conclusions
# 
# The notebook above walks through all the steps to train and test a model that extracts building footprints with (hopefully) persistent unique identifiers (i.e. addresses) from a deep temporal stack of medium resolution satellite imagery.  
# 
# The model proposed here achieves a [SCOT](https://github.com/CosmiQ/solaris/blob/master/solaris/eval/scot.py) score of 0.158 on the SpaceNet 7 test_public data.
# 

# In[ ]:




