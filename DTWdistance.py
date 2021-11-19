import numpy as np
import fastdtw as fd
import argparse
from scipy.spatial.distance import euclidean, sqeuclidean
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
import glob, os

#varthresh as Min. variance to use dimension in DTW
#sigma as Sigma for Gaussian smoothing

def distance_using_dtw(model, seq1, seq2="all", varthresh=0.05, sigma=0.5):
    distance_val = {}
    directory = os.getcwd()
    coords = "/home/itsc/project/coords/"

    if(model == "body"):
        coordSequences1 = [[] for _ in range(50)] # two dim. list with list for every coordinate
        coordSequences2 = [[] for _ in range(50)]
        normCoordSequences1 = [[] for _ in range(50)]
        normCoordSequences2 = [[] for _ in range(50)]
        # read npy file of user video
        x = np.load(seq1)
        for i in range(50): # 25 key points -> 50 coordinates
            coordSequences1[i]=x[i]

    elif(model == "coco"):
        coordSequences1 = [[] for _ in range(36)]
        coordSequences2 = [[] for _ in range(36)]
        normCoordSequences1 = [[] for _ in range(36)]
        normCoordSequences2 = [[] for _ in range(36)]
        # read npy file of user video
        x = np.load(seq1)
        for i in range(36): # 18 key points -> 36 coordinates
            coordSequences1[i]=x[i]

    else:
        coordSequences1 = [[] for _ in range(30)]
        coordSequences2 = [[] for _ in range(30)]
        normCoordSequences1 = [[] for _ in range(30)]
        normCoordSequences2 = [[] for _ in range(30)]
        # read npy file of user video
        x = np.load(seq1)
        for i in range(30): # 15 key points -> 30 coordinates
            coordSequences1[i]=x[i]   

    if(seq2!="all"): 
        y = np.load(seq2)
        if(model == "body"):
            for i in range(50): 
                coordSequences2[i]=y[i]   
        elif(model == "coco"):  
            for i in range(36): 
                coordSequences2[i]=y[i]   
        else:
            for i in range(30): 
                coordSequences2[i]=y[i]              
        
        compareDimensions = set() # add all dimensions to set that have a variance > threshold in either sequence

        if float(sigma) > 0.1:
            for i,s in enumerate(coordSequences1):
                s = medfilt(s, 5)
                coordSequences1[i] = s

            for i,s in enumerate(coordSequences2):
                s = medfilt(s, 5)
                coordSequences2[i] = s
        for i,kps in enumerate(coordSequences1):
            varSeq = np.var(kps)
            # mean = np.mean(kps)
            # std = np.std(kps)
            # nmkps = [x-mean for x in kps]
            # nmstdkps = [x/std for x in nmkps] 
            # normCoordSequences1[i] = nmstdkps
            norm = np.sqrt(np.sum(np.square(kps)))
            normCoordSequences1[i] = [x/norm for x in kps]
            # print('S1: Variance Signal ', i, ': ' , varSeq)
            if varSeq > float(varthresh):
                compareDimensions.add(i)

        for i,kps in enumerate(coordSequences2):
            varSeq = np.var(kps)
            # mean = np.mean(kps)
            # std = np.std(kps)
            # nmkps = [x-mean for x in kps]
            # nmstdkps = [x/std for x in nmkps]
            # normCoordSequences2[i] = nmstdkps
            norm = np.sqrt(np.sum(np.square(kps)))
            normCoordSequences1[i] = [x/norm for x in kps]
            # print('S2: Variance Signal ', i, ': ' , varSeq)
            if varSeq > float(varthresh):
                compareDimensions.add(i)

        # print(compareDimensions)

        '''
        For some reason, the commumative property is violated if the dimensions
        are in a different order... possibly rounding errors. Therefore they have
        to be sorted before performing the DTW.
        '''
        compareDimensions = sorted(compareDimensions)

        # print("USING SELECTED DIMENSIONS ")
        distSum = 0.0

        for dim in compareDimensions:
            distance, path = fd.fastdtw(normCoordSequences1[dim], normCoordSequences2[dim], dist=euclidean)
            distSum += distance
            # print(distance)
        # print("NORMALIZED DISTANCE: "+ str(distSum))
        distance_val[seq2]=distSum
        return distance_val

    else:
        os.chdir(coords)
        for file in glob.glob("*_"+model+".npy"):
            y = np.load(file)
            if(model == "body"):
                for i in range(50): 
                    coordSequences2[i]=y[i]   
            elif(model == "coco"):  
                for i in range(36): 
                    coordSequences2[i]=y[i]   
            else:
                for i in range(30): 
                    coordSequences2[i]=y[i]    

            os.chdir(directory)
            compareDimensions = set() # add all dimensions to set that have a variance > threshold in either sequence

            if float(sigma) > 0.1:
                for i,s in enumerate(coordSequences1):
                    s = medfilt(s, 5)
                    coordSequences1[i] = s

                for i,s in enumerate(coordSequences2):
                    s = medfilt(s, 5)
                    coordSequences2[i] = s

            for i,kps in enumerate(coordSequences1):
                varSeq = np.var(kps)
                # mean = np.mean(kps)
                # std = np.std(kps)
                # nmkps = [x-mean for x in kps]
                # nmstdkps = [x/std for x in nmkps]
                # normCoordSequences1[i] = nmstdkps
                # norm = np.sum(kps)
                norm = np.sqrt(np.sum(np.square(kps)))
                normCoordSequences1[i] = [x/norm for x in kps]
                # print('S1: Variance Signal ', i, ': ' , varSeq)
                if varSeq > float(varthresh):
                    compareDimensions.add(i)

            for i,kps in enumerate(coordSequences2):
                varSeq = np.var(kps)
                # mean = np.mean(kps)
                # std = np.std(kps)
                # nmkps = [x-mean for x in kps]
                # nmstdkps = [x/std for x in nmkps] 
                # normCoordSequences2[i] = nmstdkps
                # norm = np.sum(kps)
                norm = np.sqrt(np.sum(np.square(kps)))
                normCoordSequences2[i] = [x/norm for x in kps]
                # print('S2: Variance Signal ', i, ': ' , varSeq)
                if varSeq > float(varthresh):
                    compareDimensions.add(i)

            # print(compareDimensions)

            compareDimensions = sorted(compareDimensions)

            # print("USING SELECTED DIMENSIONS ")
            distSum = 0.0

            for dim in compareDimensions:
                distance, path = fd.fastdtw(normCoordSequences1[dim], normCoordSequences2[dim], dist=sqeuclidean)
                distSum += distance
                # print(distance)
            
            # print("NORMALIZED DISTANCE: "+ str(distSum)+'\n')
            distance_val[file]=distSum
            os.chdir(coords)
        return distance_val         