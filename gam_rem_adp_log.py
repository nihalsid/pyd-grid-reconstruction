# -*- coding: utf-8 -*-
""" gamma spots removal for Neutron radiography and tomography images
using 'find and replace' strategy, this is a discriminative method,
better than unique threshold substitution

Find the gamma spots by laplacian or LOG operation, which are usually utilized for
edge finding. Those greatly changing area will produce high level in the resultant image. This can work very well for those high level gamma spots.
For those relatively low level gamma spots, a lower threshold is needed. But,
some steep edges also have very high values. This make it a hard choice: whether filter out those
low level noise at some sacrifice of the edge info, or keep the edge intact as well as those low level noise.

adptive thresholding : med3(log)+ thr
based on the fact that: the edges of the obj usually have a width of more than 5 pixels,
while the gamma spots usually involve less then 3 pixels in width. By filtering the
resultant image from LOG(laplacian of gaussian) filtering with a 3by3 median kernel, the edges
will survive, while most of the gamma pixels lost their magnitude drastically, some are wiped out.
Adaptive size of median filter:
   3 by 3 for small spots, 5 by 5 for medium ones, 7 by 7 for high level ones.

written by Hongyun Li, visiting physicist at FRM2,TUM,Germany, Feb 09, 2006
Contact info:
hongyunlee@yahoo.com, or lihongyun03@mails.tsinghua.edu.cn
Northwest Institute of Nuclear Technology, China

ported to python by Michael Schulz, michael.schulz@frm2.tum.de    
"""

import logging
import numpy as np
from scipy import signal
from scipy import ndimage
from numba import jit

@jit
def log(N, sigma):
    """Laplacian of Gaussian.
    
    :param N: kernel size.
    :type N: :class:'int'
    :param sigma: width
    :type sigma: :class:'float'
    """
    if(np.mod(N,2) == 0):
        N=N+1
    n2=N//2
    g=np.zeros([N,N])
    log=np.zeros([N,N])
    sigma=float(sigma)
    for i in np.arange(-n2, n2+1):
        for j in np.arange(-n2, n2+1):
           g[i+n2,j+n2]=np.exp(-(np.power(i,2)+np.power(j,2))/(2.0*np.power(sigma,2)))
    sumg=np.sum(g)
    for i in np.arange(-n2, n2+1):
        for j in np.arange(-n2, n2+1):
           log[i+n2,j+n2]=(np.power(i,2)+np.power(j,2)-2*np.power(sigma,2))*g[i+n2,j+n2]/(2.0*3.1415926*np.power(sigma,6)*sumg)
    return log

@jit
def gam_rem_adp_log(img, thr3=25, thr5=100, thr7=400, sig_log=0.8):
    f_log=-log(9,sig_log)      #create the kernel of LOG filter
    img_log = signal.fftconvolve(img, f_log, mode = 'same')  # do the LOG filter
    #    img_logm3 = signal.medfilt(img_log,3)        # this is much slower!
    img_logm3 = ndimage.median_filter(img_log,(3,3))        # median the LOG edge enhanced image, 3 by 3 is good enough

    #----------substitute only those pixels whose values are greater than adaptive threshold
    #----------  ,which is set to median(log(img))+thr, where thr is a predetermined constant chosed
    #----------  to be best fitted for specific noise charateristics by user.
    #Adaptive filter size:
    #"Opening" operator:
    imgthr3 = np.greater(img_log, img_logm3+thr3)
    imgthr5 = np.greater(img_log, img_logm3+thr5)
    imgthr7 = np.greater(img_log, img_logm3+thr7)


    # we found that some of the edge pixels are not removed. so we dilate the map7
    if np.sum(imgthr7) >= 0:
        s = np.ones([3,3])     #the dilate structure
        single7 = signal.convolve2d(imgthr7*255, s/np.sum(s), mode = 'same') < 30 # boxcar smoothing plus threshold to identify single pixel spots
        #        single7 = signal.fftconvolve(imgthr7.astype(float)*255, s/np.sum(s), mode = 'same') < 30 # this is a little bit slower in this case. why?
        single7 = single7 & imgthr7     # those single pixels in threshold map7
        imgthr7 = np.logical_xor(imgthr7, single7)    # take these out of map7 before dilation
        imgthr7 = np.logical_or(ndimage.binary_dilation(imgthr7,s),single7) # and add them again after dilation
    
    imgthr5 = np.logical_xor((imgthr5|imgthr7),imgthr7)

    imgthr3 = np.logical_xor(imgthr3,imgthr5)
    
#     clean the border of map5 and map7, otherwise there might be out of range error
#     when doing the replacement
    imgthr7[:3,:]=False  
    imgthr7[:,:3]=False 
    imgthr7[:,-3:]=False 
    imgthr7[-3:,:]=False

    imgthr5[:2,:]=False
    imgthr5[:,:2]=False
    imgthr5[:,-2:]=False 
    imgthr5[-2:,:]=False

    img_adp = np.copy(img)
    imgm3 = ndimage.median_filter(img,(3,3))        #3 by 3 median filtering, as substitution image

    thr3list = np.nonzero(imgthr3)
    n3 = np.shape(thr3list)[1]
    img_adp[thr3list]=imgm3[thr3list]


    index = np.nonzero(imgthr5)
    n5 = np.shape(index)[1]
    for i in range(n5):
        img_adp[index[0][i],index[1][i]]=np.median(img[index[0][i]-2:index[0][i]+3, index[1][i]-2:index[1][i]+3])
        
    index = np.nonzero(imgthr7)
    n7 = np.shape(index)[1]
    for i in range(n7):
        img_adp[index[0][i],index[1][i]]=np.median(img[index[0][i]-3:index[0][i]+4, index[1][i]-3:index[1][i]+4])
    ntot = n3 + n5 + n7  
    sz = np.shape(img)
    m = sz[0]
    n = sz[1]

    logging.info('Gamma removal: %d pixels substituted. %f percent of total pixels.', ntot,ntot*100.0/(m*n))
    logging.info('M3: %f=%f percent  M5: %f=%f percent  M7: %f=%f percent of total pixels',n3,n3*100.0/(m*n),n5,n5*100.0/(m*n),n7,n7*100.0/(m*n))

    return img_adp


    






