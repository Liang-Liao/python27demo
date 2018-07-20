#coding=utf-8

from numpy import *

#导入数据
def loadDataSet(fileName, delime='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delime) for line in fr.readlines()]
    dataArr = [map(float, line) for line in stringArr]
    return mat(dataArr)

#PCA降维方法
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meadRemoved = dataMat - meanVals
    covMat = cov(meadRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat + 1):-1]
    redEigVects = eigVects = eigVects[:, eigValInd]
    lowDDatMat = meadRemoved * redEigVects
    reconMat = (lowDDatMat * redEigVects.T) + meanVals
    return lowDDatMat, reconMat

#将NaN数据替换为平均值
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0], i])
        datMat[nonzero(isnan(datMat[:,i].A))[0], i] = meanVal
    return datMat

if __name__ == '__main__':
    replaceNanWithMean()
    a = 5