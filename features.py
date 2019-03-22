import math
from math import cos,sin
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
from scipy.linalg.misc import norm

import transformations
#import benchmark


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])  #存放强度
        orientationImage = np.zeros(srcImage.shape[:2])   #存放方向
        #用Sobel算子计算一阶梯度
        Ix = ndimage.sobel(srcImage,axis = 0,mode = 'reflect')
        Iy = ndimage.sobel(srcImage,axis = 1,mode = 'reflect')
        
        A = Ix * Ix
        B = Ix * Iy
        C = Iy * Iy

        x = np.size(srcImage,0)
        y = np.size(srcImage,1)

        a1 = ndimage.gaussian_filter(A,sigma = 0.5,mode = 'reflect')   #每个点有a1 b1 c1
        b1 = ndimage.gaussian_filter(B,sigma = 0.5,mode = 'reflect')
        c1 = ndimage.gaussian_filter(C,sigma = 0.5,mode = 'reflect')

        for i in range(0,x):
            for j in range(0,y):
                det = a1[i][j] * c1[i][j] - b1[i][j]**2
                tra = a1[i][j] + c1[i][j]
                harrisImage[i][j] = det - 0.1 * tra**2
        
        orientationImage = np.arctan2(Ix,Iy) * 180 / np.pi   #这里注意坐标的要求



        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN

        x = np.size(harrisImage,0)
        y = np.size(harrisImage,1)
        '''
        img = np.pad(harrisImage,(3,3),'constant',constant_values = (0,0))
        for i in range(3,x+3):
            for j in range(3,y+3):
                flag = 0
                for k in range(i-3,i+4):
                    for g in range(j-3,j+4):
                        if(img[i][j]<img[k][g]):
                            flag= 1
                            break
                if(flag == 1):
                    destImage[i-3][j-3] = False
                else:
                    destImage[i-3][j-3] = True
        '''
        #改进：

        img = ndimage.filters.maximum_filter(harrisImage,size = 7) #最大滤波

        for i in range(x):
            for j in range(y):
                if(img[i][j] == harrisImage[i][j]):
                    destImage[i][j] = True
                else:
                    destImage[i][j] = False


        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        count = 0
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()   #关键点检测函数    存关键点

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10    #特征点领域直径
                f.pt = (x,y)   #位置坐标   （列，行）
                f.angle = orientationImage[y][x]  #特征点方向
                f.response = harrisImage[y][x]
                count+=1


                features.append(f)
               # raise Exception("TODO in features.py not implemented")
                # TODO-BLOCK-END

              

        #return features
        #自适应最大响应
        '''
        features.sort(key=lambda KeyPoint:KeyPoint.response, reverse=True)#特征点响应排序
        features1=[]
        N=count/4     #特征点数的1/4
        R=9      #半径
        countn=0
        sta=1
        while(countn<N or sta==1):     #此区域的特征点数超过N 则退出
            sta=0
            p = R   #半径
            for i,f in enumerate(features):
                y,x=f.pt
                y,x=int(y),int(x)
                res=f.response
                img=np.pad(harrisImage,((p,p),(p,p)),'constant',constant_values = (0,0))
                #其实可以直接用上面的滤波函数

                flag=0
                for k in range(x-p,x+p+1):
                    for l in range(y-p,y+p+1):
                        if(res<img[k][l]):
                            flag=1
                            break
                    if flag==1:
                        break    
                if(flag!=1):
                    features1.append(f)
                    del features[i]
                    countn+=1
            R-=1     
            
        #return features
        '''
        return features

class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):  #i为关键点个数 f为关键点的枚举
            x, y = f.pt
            x, y = int(x), int(y)          #y是行

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.              图像外的点像素为0
            # TODO-BLOCK-BEGIN

            a = np.size(grayImage,0)
            b = np.size(grayImage,1)
            for m in range(y-2,y+3):
                for n in range(x-2,x+3):
                    if(m<0 or n<0 or m>=a or n>=b):
                        desc[i][(m-y+2)*5 + n-x+2] = 0
                    else:
                        desc[i][(m-y+2)*5 + n-x+2] = grayImage[m][n]

            i+=1      #改了
    
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))  #[特征点标号][每个点的区域]
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            '''MOPS'''
            x, y = f.pt
            x, y = int(x), int(y)    #y是行
            #a = np.size(grayImage,0)
            #b = np.size(grayImage,1)
            theta = f.angle * np.pi / 180
            '''
            translate1 = np.array([[1,0,-x],[0,1,-y],[0,0,1]]) #平移大图使特征点与坐标原点重合
            rotate = np.array([[cos(-theta),-sin(-theta),0],[sin(-theta),cos(-theta),0],[0,0,1]])#旋转使特征点方向与x轴方向重合
            scale = np.array([[0.2,0,0],[0,0.2,0],[0,0,1]]) #放缩五分之一
            translate2 = np.array([[1,0,4],[0,1,4],[0,0,1]]) #大图平移使特征点与坐标原点重合
            '''

            translate1 = transformations.get_trans_mx(np.array([-x, -y]))
            translate1 = translate1[0:3,0:3]
            rotate = transformations.get_rot_mx(0,0,-theta)
            rotate = rotate[0:3,0:3]
            scale = transformations.get_scale_mx(0.2, 0.2,1)
            scale = scale[0:3,0:3]
            translate2 = transformations.get_trans_mx(np.array([4, 4]))
            translate2 = translate2[0:3,0:3]

            t1 = np.dot(translate2,scale)
            t2 = np.dot(t1,rotate)
            t3 = np.dot(t2,translate1)
            transMx = t3[0:2]


            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)  #8*8图像 既是变化后的大图

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN

            std = np.std(destImage)  #标准差
            mean = np.mean(destImage) #均值

            if std <1e-5:
                desc[i] = 0
            else:
                desc[i] = ((destImage-mean) / std).reshape(-1,64)
            i+=1
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2    #断言图1 是2维的
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.    #这里说了 你要找最优的
        # TODO-BLOCK-BEGIN

        for i in range(desc1.shape[0]):
            m = cv2.DMatch()
            tmp1 = np.array(desc1[i]).reshape((1,-1))
            tmp2 = scipy.spatial.distance.cdist(tmp1,desc2,metric='euclidean')
            min1 = int(np.argmin(tmp2))
            m.queryIdx = i
            m.trainIdx = min1
            m.distance = tmp2[0][min1]
            matches.append(m)

        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN

        for i in range(desc1.shape[0]):
            m = cv2.DMatch()
            tmp1 = np.array(desc1[i]).reshape((1,-1)) #存图一每个特征点的描述符 不是二维要报错
            tmp2 = scipy.spatial.distance.cdist(tmp1,desc2,metric='euclidean') #图一点与图二所有点特征描述符进行匹配
            
            min1 = int(np.argmin(tmp2))
            m.queryIdx = i
            m.trainIdx = min1
            tmp2.sort()
            firstMin = tmp2[0][0]
            secondMin = tmp2[0][1]
            ratio = firstMin / secondMin
            m.distance = ratio

            matches.append(m)

        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

