from imageio import imread
from sss_semantic_soft_segmentation.semantic_soft_segmentation import semantic_soft_segmentation
from scipy.io import loadmat

if __name__ == '__main__':
    #img = imread('COCO_train2014_000000362884.jpg', mode='RGB')
    image = imread('./SIGGRAPH18SSS/samples/docia.png')
    ori = loadmat('./SIGGRAPH18SSS/Feat/docia.mat')
    features = ori['embedmap']
    print(features.shape)

    #features = img[:, img.shape[1] // 2 + 1:, :]
    #image = img[:, :img.shape[1] // 2, :]

    sss = semantic_soft_segmentation(image, features)
    print(sss)