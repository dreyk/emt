import cv2
import numpy as np

back = cv2.imread('./testdata/default.png')

front = cv2.imread('/Users/agunin/Downloads/Alpha/People/hairs-3225896_1920.png',cv2.IMREAD_UNCHANGED)


small_front = cv2.resize(front,(256,256))

smal_back=  cv2.resize(back,(256,256))

a = small_front[:,:,3:].astype(np.float32)/255
res = small_front[:,:,0:3].astype(np.float32)*a + smal_back.astype(np.float32)*(1-a)
res = res.astype(np.uint8)

cv2.imwrite('res.png',res)




#key = cv2.waitKey()
