import sys
sys.path.insert(0,r'C:\Users\Yawar\Documents\FRM-II\c++\GridReconstructionPy\x64\Release')

import gridrecon
arr = gridrecon.log_filter(3, 3)

print (arr)
print(type(arr))

print("-" * 40)

import pyfits
import numpy as np
xmin=200 # As measured in ImageJ (hopefully)
xmax=1000
ymin=45
ymax=2000

img_path = 'C:\\Users\\Yawar\\Documents\\FRM-II\\data\\ob\\00155908.fits'
img =  np.ascontiguousarray(pyfits.open(img_path)[0].data[ymin:ymax,xmin:xmax], dtype=np.float32)
out = np.ascontiguousarray(np.zeros(img.shape, dtype=np.float32), dtype=np.float32)
print(gridrecon.gam_rem_adp_log(img, out, 50,100,200,0.8))

print(out)


