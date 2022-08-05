from sys import exit
from argparse import ArgumentParser
from utils import getFlattenedMNISTImages
import numpy as np
from multiprocessing import Pool, cpu_count
from readNNet import readNNet
from json import dump
import time
import math


def evaluateModel(prev, weights, biases):
    if len(weights) == 0:
        return []

    output = np.array(
        [np.dot(w, prev) + b for w, b in zip(weights[0], biases[0])])

    if len(weights) != 1:

        def relu(x):
            return max(x, 0)

        output = np.array([relu(x) for x in output])

    return [output.tolist()] + evaluateModel(output, weights[1:], biases[1:])


def getIntegerPart(t):
    minBound, maxBound = t[0], t[1]
    if abs(minBound) < 1 and abs(maxBound) < 1:
        return 1

    return (math.ceil(math.log2(max(abs(minBound), abs(maxBound)) + 1)) + 1)


def writeTypes(typeData):
    fileName = 'types.json'
    if args.timestamp:
        fileName = 'types' + str(time.time()) + '.json'

    with open(fileName, 'w') as f:
        dump(typeData, f)

    print(fileName)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('network', help='NNet file containing network')
    parser.add_argument('int', help='Maximum integer part', type=int)
    parser.add_argument('frac', help='Maximum fractional', type=int)
    parser.add_argument('--datasize',
                        help='Maximum number of images to try',
                        type=int)
    parser.add_argument('--noreduce',
                        help='Just create types without decreasing the int part',
                        action='store_true')
    parser.add_argument(
        '--timestamp', help='Attach the timestamp to end of types JSON file', action='store_true')
    args = parser.parse_args()

    weights, biases, inputMins, _, _, _ = readNNet(args.network, withNorm=True)

    maxBounds = [[(0, 0) for i in range(len(inputMins))]] + [[(0, 0)
                                                              for n in l]
                                                             for l in biases]

    if args.noreduce:
        typeData = [[[args.int, args.frac] for t in l]
                    for l in maxBounds]
        writeTypes(typeData)
        exit(0)

    (xTrain, _), (xTest, _) = getFlattenedMNISTImages()
    datasize = len(xTrain) + len(xTest)
    if args.datasize is not None:
        datasize = args.datasize
    x = np.concatenate((xTrain, xTest))[:datasize]

    with Pool(processes=cpu_count()) as p:
        results = p.starmap(evaluateModel, [(im, weights, biases) for im in x])

    maxIntValue = float('-inf')
    minIntValue = float('inf')
    for r, im in zip(results, x):
        resWithImage = [im] + r
        for l in range(len(resWithImage)):
            for n in range(len(resWithImage[l])):
                maxBounds[l][n] = (min(maxBounds[l][n][0], resWithImage[l][n]),
                                   max(maxBounds[l][n][1], resWithImage[l][n]))
                maxIntValue = max(maxIntValue, max(maxBounds[l][n]))
                minIntValue = min(minIntValue, min(maxBounds[l][n]))

    with open('bounds.json', 'w') as f:
        dump(maxBounds, f)

    typeData = [[[min(getIntegerPart(t), args.int), args.frac] for t in l]
                for l in maxBounds]

    writeTypes(typeData)
