import cv2
import numpy as np
from sklearn import random_projection
import pywt


img = cv2.imread('V1_tamp.jpg')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(500)

Mr = 256

img = cv2.resize(img, dsize=(Mr,Mr))
cv2.imshow('img',img)
cv2.waitKey(500)

img0 = cv2.GaussianBlur(img, (5,5),sigmaX=1)
cv2.imshow('img',img0)
cv2.waitKey(500)

rgb_img = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)

M = np.array([[65.481, 128.553, 24.966],[-37.797, -74.203, 112.000],[112.000, -93.786, -18.214]])
Ycbcr = np.array([16,128,128])+ np.dot(rgb_img, M.T)
Ycbcr1 = Ycbcr/255.0

cv2.imshow('img',Ycbcr1)
cv2.waitKey(500)

Ycbcr1_mean = np.mean(Ycbcr1,axis=(0,1))
print('Ycbcr1_mean:',Ycbcr1_mean)

global_feats = Ycbcr1_mean

sift = cv2.SIFT_create()
gray= cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(gray, None, )
print('len kp:',len(kp))

imgkp=cv2.drawKeypoints(gray,kp,img0.copy())
cv2.imshow('img',imgkp)
cv2.waitKey(30)



print('des.shape:',des.shape)
transformer = random_projection.GaussianRandomProjection(eps=0.078, n_components=32)
des1 = transformer.fit_transform(des)
print('des1.shape:',des1.shape)

coeffs = pywt.dwt2(gray, 'haar')
cA, (cH, cV, cD) = coeffs

cv2.imshow('img',cA/255.0)
cv2.waitKey(30)

coeffs = pywt.wavedec2(cA,wavelet='db1')
dwt_low = coeffs[0]

# Substract median and compute hash
med = np.median(dwt_low)
diff_hash = dwt_low > med

print('diff_hash:',diff_hash)
import pdb;pdb.set_trace()