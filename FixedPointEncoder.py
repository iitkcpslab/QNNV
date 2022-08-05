# Imports

from FixedPointData import FixedPointData
from readNNet import readNNet
from tensorflow.keras.datasets import mnist, fashion_mnist
from argparse import ArgumentParser
import json
import gurobipy as gp
import re
from gurobipy import GRB
import multiprocessing
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# FixedPointEncoder
#    A class for holding data about encoding process


class FixedPointEncoder:
    def __init__(self,
                 network,
                 weights,
                 biases,
                 typeData,
                 relu7,
                 lateshift,
                 unsat_core,
                 verbose=False,
                 modelName='FixedPointVerification',
                 maxint=500000):
        self.network = network
        self.weights = weights
        self.biases = biases
        self.verbose = verbose
        self.typeData = typeData
        self.numLayers = len(biases)
        self.relu7 = relu7
        self.lateshift = lateshift
        self.unsat_core = unsat_core
        self.inputVars = []
        self.outputVars = []
        self.allVars = []
        self.model = gp.Model(modelName)
        self.offset = 0.9999
        self.maxint = maxint


# Aux method for creating variables


def createVar(encoder: FixedPointEncoder,
              name,
              lb=-GRB.INFINITY,
              ub=GRB.INFINITY,
              type=GRB.INTEGER):
    var = encoder.model.addVar(lb=lb, ub=ub, vtype=type, name=name)
    encoder.allVars.append(var)
    return var


def addMinConstraint(encoder: FixedPointEncoder, ub, var, name):
    bUb = createVar(encoder,
                    "b_min_ub_{}".format(name),
                    lb=0,
                    ub=1,
                    type=GRB.BINARY)
    bVar = createVar(encoder,
                     "b_min_var_{}".format(name),
                     lb=0,
                     ub=1,
                     type=GRB.BINARY)
    result = createVar(encoder, "ub_{}".format(name))
    encoder.model.addConstr(bUb + bVar == 1)
    encoder.model.addConstr(result <= ub)
    encoder.model.addConstr(var - result - encoder.maxint * bUb <= 0)
    encoder.model.addConstr(var - result + encoder.maxint * bUb >= 0)
    encoder.model.addConstr(result + encoder.maxint * bVar >= ub)
    encoder.model.addConstr(var + encoder.maxint * bVar >= ub)
    return result


def addMaxConstraint(encoder: FixedPointEncoder, lb, var, name):
    bLb = createVar(encoder,
                    "b_max_lb_{}".format(name),
                    lb=0,
                    ub=1,
                    type=GRB.BINARY)
    bVar = createVar(encoder,
                     "b_max_var_{}".format(name),
                     lb=0,
                     ub=1,
                     type=GRB.BINARY)
    result = createVar(encoder, "lb_{}".format(name))
    encoder.model.addConstr(bLb + bVar == 1)
    encoder.model.addConstr(result >= lb)
    encoder.model.addConstr(var - result - encoder.maxint * bLb <= 0)
    encoder.model.addConstr(var - result + encoder.maxint * bLb >= 0)
    encoder.model.addConstr(result - encoder.maxint * bVar <= lb)
    encoder.model.addConstr(var - encoder.maxint * bVar <= lb)
    return result


# Clip a Variable Between Given Values
#    + The constraints encode the formula ~min(upperBoundY, max(lowerBoundY, y))~, i.e., ~clip(lowerBoundY, upperBoundY, y)~.


def clip(encoder: FixedPointEncoder, lb, ub, var, name):
    # Uncomment following snippet to use Gurobi's Max-Min constraints
    # lowerBoundVar = createVar(encoder, "lb_{}".format(name))
    # encoder.model.addGenConstrMax(lowerBoundVar, [lb, var])
    # upperBoundVar = createVar(encoder, "ub_{}".format(name))
    # encoder.model.addGenConstrMin(upperBoundVar, [ub, lowerBoundVar])
    # return upperBoundVar
    return addMinConstraint(encoder, ub,
                            addMaxConstraint(encoder, lb, var, name), name)


def convertToDifferentFP(encoder, var, varName, intBits, prevFloatBits,
                         nextFloatBits):
    floatVar = createVar(encoder, "float_" + varName, type=GRB.CONTINUOUS)
    model = encoder.model
    model.addConstr(floatVar == var / 2**prevFloatBits)
    fpVar = createVar(encoder, "fp_" + varName)
    model.addConstr(floatVar * 2**nextFloatBits - encoder.offset <= fpVar)
    model.addConstr(fpVar <= floatVar * 2**nextFloatBits)
    return fpVar


def encodeNeuronLateShift(encoder: FixedPointEncoder, inputVars, layerNo,
                          neuronNo):
    y = createVar(encoder, "pre_{}_{}".format(layerNo, neuronNo))

    sumFloatBits = encoder.typeData[layerNo + 1][neuronNo][1]
    multiplications = []

    weights = encoder.weights[layerNo][neuronNo]
    bias = encoder.biases[layerNo][neuronNo]

    prevLayerFloatBits = [t[1] for t in encoder.typeData[layerNo]]
    prevLayerIntBits = [t[0] for t in encoder.typeData[layerNo]]

    maxFloat = max(prevLayerFloatBits)

    model = encoder.model

    family = FixedPointData(sum(encoder.typeData[layerNo + 1][neuronNo]))

    lb = -(2**(family.totalBits - 1))
    ub = 2**(family.totalBits - 1) - 1

    if len(np.unique(prevLayerFloatBits)) != 1:
        # Will have to align the fractional-parts
        # will do that by shifting all of them left
        inputVars = [
            convertToDifferentFP(ip, ib, pfb, maxFloat) for ip, ib, pfb in zip(
                inputVars, prevLayerIntBits, prevLayerFloatBits)
        ]
        prevLayerFloatBits = [maxFloat] * len(prevLayerFloatBits)

    for (i, var), weight, prevFloatBits in zip(enumerate(inputVars), weights,
                                               prevLayerFloatBits):
        # prevFloatBits will be the same for all the multiplications
        # prevFloatBits == maxFloat
        family = FixedPointData(prevLayerIntBits[i] + prevLayerFloatBits[i])
        curLb = -(2**(family.totalBits - 1))
        curUb = 2**(family.totalBits - 1) - 1
        scaledWeight = family.convert(weight, prevFloatBits)

        res = createVar(encoder, "res_{}_{}_{}".format(layerNo, neuronNo, i))
        model.addConstr(res == var * scaledWeight)

        multiplications.append(res)

    family = FixedPointData(2 * (max(prevLayerIntBits) + maxFloat))
    scaledBias = family.convert(bias, 2 * maxFloat)

    sumVar = createVar(encoder, "sum_{}_{}".format(layerNo, neuronNo))

    multiplicationSum = gp.quicksum(multiplications + [scaledBias])
    model.addConstr(sumVar == multiplicationSum)
    # I need sumFloatBits number of bits in the fractional part for
    # next neuron
    divisor = 2**(2 * maxFloat - sumFloatBits)

    model.addConstr(multiplicationSum / divisor - encoder.offset <= y)
    model.addConstr(y <= multiplicationSum / divisor)

    if encoder.numLayers - 1 == layerNo:
        return clip(encoder, lb, ub, y, "pre_{}_{}".format(layerNo, neuronNo))

    lb = 0

    if encoder.relu7:
        ub = min(7, ub)

    return clip(encoder, lb, ub, y, "post_{}_{}".format(layerNo, neuronNo))


def encodeNeuronEarlyShift(encoder: FixedPointEncoder, inputVars, layerNo,
                           neuronNo):
    y = createVar(encoder, "pre_{}_{}".format(layerNo, neuronNo))

    sumFloatBits = encoder.typeData[layerNo + 1][neuronNo][1]
    multiplications = []

    weights = encoder.weights[layerNo][neuronNo]
    bias = encoder.biases[layerNo][neuronNo]

    prevLayerFloatBits = [t[1] for t in encoder.typeData[layerNo]]

    model = encoder.model

    family = FixedPointData(sum(encoder.typeData[layerNo + 1][neuronNo]))

    lb = -(2**(family.totalBits - 1))
    ub = 2**(family.totalBits - 1) - 1

    for (i, var), weight, prevFloatBits in zip(enumerate(inputVars), weights,
                                               prevLayerFloatBits):
        scaledWeight = family.convert(weight, sumFloatBits)
        curWeightFloatBits = sumFloatBits

        if (1 / 2**(prevFloatBits + curWeightFloatBits - sumFloatBits)) == 0:
            raise Exception("The division turns out to be zero")

        res = createVar(encoder, "res_{}_{}_{}".format(layerNo, neuronNo, i))
        multiply = var * scaledWeight
        divisor = 2**(prevFloatBits + curWeightFloatBits - sumFloatBits)

        model.addConstr(multiply / divisor - encoder.offset <= res)
        model.addConstr(res <= multiply / divisor)

        multiplications.append(
            clip(encoder, lb, ub, res,
                 "res_{}_{}_{}".format(layerNo, neuronNo, i)))

    scaledBias = family.convert(bias, sumFloatBits)

    model.addConstr(y == gp.quicksum(multiplications + [scaledBias]))

    if encoder.numLayers - 1 == layerNo:
        return clip(encoder, lb, ub, y, "pre_{}_{}".format(layerNo, neuronNo))

    lb = 0

    if encoder.relu7:
        ub = min(7, ub)

    return clip(encoder, lb, ub, y, "post_{}_{}".format(layerNo, neuronNo))


def encodeLayer(encoder: FixedPointEncoder, inputVars, weights, biases,
                layerNo):
    outputVars = []

    for neuron in range(len(biases)):
        if encoder.lateshift:
            outVar = encodeNeuronLateShift(encoder, inputVars, layerNo, neuron)
        else:
            outVar = encodeNeuronEarlyShift(encoder, inputVars, layerNo,
                                            neuron)

        outputVars.append(outVar)

    return outputVars


def encodeNetwork(encoder):
    model = encoder.model
    numInputs = len(encoder.weights[0][0])
    inputVars = [
        createVar(encoder, "inp{}".format(i)) for i in range(numInputs)
    ]

    lowerBounds = [-(2**(i + f - 1)) for [i, f] in encoder.typeData[0]]
    upperBounds = [2**(i + f - 1) - 1 for [i, f] in encoder.typeData[0]]
    clippedInputVars = [
        clip(encoder, lb, ub, v, "inp{}".format(i))
        for (i,
             v), lb, ub in zip(enumerate(inputVars), lowerBounds, upperBounds)
    ]

    encoder.inputVars += inputVars

    currentVars = clippedInputVars
    numLayers = encoder.numLayers

    for i in range(numLayers):
        currentVars = encodeLayer(encoder, currentVars, encoder.weights[i],
                                  encoder.biases[i], i)

    encoder.outputVars += currentVars
    return encoder


# Utility Methods For Manipulating Image
#    Flatten a 2D list into a 1D list.
def flatten(list2D):
    return [e for e1 in list2D for e in e1]


# Convert an image from floating-point types to fixed-point types.
def imageToFixedPoint(image, typeData):
    return [
        FixedPointData(ip + fp).convert(px, fp)
        for px, [ip, fp] in zip(image, typeData)
    ]


# Convert values of image pixels from 0-255 range to 0-1 range, i.e., normalize the values.
def normalizeImage(image):
    return [x / 256.0 for x in image]


# Add Infinity Norm constraints
#    Every input variable can deviate between the range [ip - e, ip + e].
def addInfNormInp(encoder: FixedPointEncoder, image, eps):
    for (i, ip), px, e in zip(enumerate(encoder.inputVars), image, eps):
        encoder.model.addConstr(ip - px >= -e)
        encoder.model.addConstr(ip - px <= e)


def addInfNormOut(encoder: FixedPointEncoder, result):
    model = encoder.model
    outputVars = encoder.outputVars
    # Uncomment the following code to use Gurobi's AND and OR constraints
    # binVars = [
    #     createVar(encoder, "bin_out_{}".format(i), type=GRB.BINARY, lb=0, ub=1)
    #     for i in range(len(encoder.outputVars))
    # ]

    # for (i, bv), ov, [_, bits] in zip(enumerate(binVars), outputVars,
    #                                   encoder.typeData[-1]):
    #     model.addConstr(
    #         (bv == 1) >>
    #         (ov / (1 << bits) >= outputVars[result] /
    #          (1 << encoder.typeData[-1][result][1]) +
    #          (1 / 2**bits)))  # np.nextafter(np.double(0), np.double(1))))
    #     model.addConstr((bv == 0) >> (ov / (1 << bits) <= outputVars[result] /
    #                                   (1 << encoder.typeData[-1][result][1])))

    # res = createVar(encoder, "res", lb=0, ub=1, type=GRB.BINARY)
    # model.addGenConstrOr(res, binVars)
    # model.addConstr(res == 1)
    # return res

    lessBinVars = [
        createVar(encoder,
                  "bin_out_less_{}".format(i),
                  type=GRB.BINARY,
                  lb=0,
                  ub=1) for i in range(len(encoder.outputVars))
    ]
    greaterBinVars = [
        createVar(encoder,
                  "bin_out_greater_{}".format(i),
                  type=GRB.BINARY,
                  lb=0,
                  ub=1) for i in range(len(encoder.outputVars))
    ]

    for (i, bl), bg, ov, [_,
                          bits] in zip(enumerate(lessBinVars), greaterBinVars,
                                       outputVars, encoder.typeData[-1]):
        if i == result:
            continue
        model.addConstr(bl + bg == 1)
        model.addConstr((ov / (1 << bits)) >= outputVars[result] /
                        (1 << encoder.typeData[-1][result][1]) +
                        (1 / 2**bits) - encoder.maxint * bl)
        model.addConstr(
            (ov / (1 << bits)) <= outputVars[result] /
            (1 << encoder.typeData[-1][result][1]) + encoder.maxint * bg)

    res = createVar(encoder, "res")
    model.addConstr(res == gp.quicksum(greaterBinVars[:result] +
                                       greaterBinVars[result + 1:]))
    model.addConstr(res >= 1)
    return res


# Get Image and Its Label From TensorFlow Dataset
def getImage(inf, fashion=False):
    if fashion:
        (_, _), (xTest, yTest) = fashion_mnist.load_data()
        print("using fashion")
    else:
        (_, _), (xTest, yTest) = mnist.load_data()
    print("\n\nTHIS IMAGE IS {}\n\n".format(yTest[inf]))
    return xTest[inf], yTest[inf]


def setupModel(model: gp.Model, timeout, logFile, debug, single_thread):
    model.Params.SolutionLimit = 1
    model.Params.MIPFocus = 1
    model.Params.IntFeasTol = 1e-9
    model.Params.TimeLimit = timeout
    if single_thread:
        model.Params.Threads = 1
        model.Params.ConcurrentMIP = 1
    else:
        model.Params.Threads = multiprocessing.cpu_count()
        model.Params.ConcurrentMIP = multiprocessing.cpu_count()
    if logFile is not None:
        model.Params.LogFile = logFile
        model.Params.LogToConsole = 0
    if debug:
        model.Params.DualReductions = 0
        model.Params.InfUnbdInfo = 1


def parse_vnncomp_property(encoder, property_file):
    target = -1
    comment_re = re.compile("; Mnist property with label: (.*?)\\.")
    assert_re = re.compile(r"\(assert \((.*?) (.*?) (.*?)\)\)")

    inputTypes = encoder.typeData[0]

    with open(property_file, 'r') as f:
        for line in f.readlines():
            comment_results = comment_re.findall(line)
            if len(comment_results) != 0:
                target = int(comment_results[0])
                continue

            assert_results = assert_re.findall(line)
            if len(assert_results) == 0:
                continue

            assert_results = assert_results[0]
            sign = assert_results[0].strip()
            input = int(assert_results[1].strip().split("_")[1])
            bound = float(assert_results[2].strip())

            integerPart, floatPart = inputTypes[input]

            if sign == '<=':
                encoder.model.addConstr(
                    encoder.inputVars[input] <= FixedPointData(
                        integerPart + floatPart).convert(bound, floatPart))
            elif sign == '>=':
                encoder.model.addConstr(
                    encoder.inputVars[input] >= FixedPointData(
                        integerPart + floatPart).convert(bound, floatPart))

    if target == -1:
        raise Exception("could not find the original image type")

    addInfNormOut(encoder, target)


def runVerification(network,
                    types,
                    timeout,
                    no_log,
                    inf,
                    eps,
                    relu7,
                    debug,
                    lateshift,
                    unsat_core,
                    single_thread,
                    vnn,
                    fashion=False):

    weights, biases, inputMins, _, _, _ = readNNet(network, withNorm=True)

    with open(types, 'r') as f:
        typeData = json.load(f)

    encoder = FixedPointEncoder(network, weights, biases, typeData, relu7,
                                lateshift, unsat_core)
    encoder = encodeNetwork(encoder)

    if vnn:
        # inf is a file name here
        parse_vnncomp_property(encoder, inf)
    else:
        # inf is the number of image in mnist dataset
        inf = int(inf)
        x, y = getImage(inf, fashion)
        x = imageToFixedPoint(normalizeImage(flatten(x)), typeData[0])

        if eps == int(eps) and 'exp' not in network:
            eps = [eps] * len(inputMins)
        else:
            inputTypes = encoder.typeData[0]
            eps = [
                FixedPointData(integerBits + floatBits).convert(
                    eps, floatBits) for [integerBits, floatBits] in inputTypes
            ]

        addInfNormInp(encoder, x, eps)
        res = addInfNormOut(encoder, y)

    encoder.model.setObjective(0, GRB.MAXIMIZE)

    logFile = None
    if no_log:
        logFile = "problem.log"

    setupModel(encoder.model, timeout, logFile, debug, single_thread)

    encoder.model.write('model.lp')
    encoder.model.write('model.mps')

    print("*** Total Vars = {} ***".format(len(encoder.model.getVars())))


def processModelOutput(encoder: FixedPointData):
    status = encoder.model.status

    if status == GRB.SOLUTION_LIMIT or status == GRB.OPTIMAL:
        inputValues = [
            round(v.X) / (1 << fp)
            for v, [_, fp] in zip(encoder.inputVars, encoder.typeData[0])
        ]

        with open('input_{}.json'.format(encoder.network), 'w') as f:
            json.dump(inputValues, f)

        vars = None
        if encoder.lateshift:
            maxFloat = max([fp for l in encoder.typeData for [ip, fp] in l])
            vars = [
                "{} = {}".format(
                    v,
                    round(v.X) / 2**maxFloat if "res" not in str(v)
                    and "sum" not in str(v) else v.X / 2**(2 * maxFloat))
                for v in encoder.model.getVars()
            ]
        else:
            vars = [str(v) for v in encoder.model.getVars()]

        with open('vars_{}.json'.format(encoder.network), 'w') as f:
            json.dump(vars, f)

        finalValues = [
            round(fv.X) / (1 << b)
            for fv, [_, b] in zip(encoder.outputVars, encoder.typeData[-1])
        ]

        with open('out_{}.json'.format(encoder.network), 'w') as f:
            json.dump(finalValues, f)

        print('SAT')
    elif status == GRB.INFEASIBLE:
        if encoder.unsat_core:
            encoder.model.computeIIS()
            encoder.model.write('conf.ilp')
        print('UNSAT')
    elif status == GRB.TIME_LIMIT:
        print('TIME_LIMIT')

    print("time:", encoder.model.Runtime)


# Driver Code

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('network', help='Neural Network to test, NNet File')
    parser.add_argument(
        'types',
        help='Types for each node, could be a constant or a JSON file')
    parser.add_argument('--timeout',
                        help='Maximum time budget (default is 3600s)',
                        type=int)
    parser.add_argument('--no_log',
                        help='Print solver log to a file named "problem.log"',
                        action='store_true')
    parser.add_argument('inf', help='Number of test image to verify')
    parser.add_argument('eps',
                        help='Amount of deviation in fixed-point world',
                        type=float)
    parser.add_argument('--relu7',
                        help='Use ReLU7 activation function',
                        action='store_true')
    parser.add_argument('--debug',
                        help='Add debugging parameters to the solver',
                        action='store_true')
    parser.add_argument('--lateshift',
                        help='Use the Google\'s late shift encoding',
                        action='store_true')
    parser.add_argument('--unsat_core',
                        help='Generate UNSAT cores',
                        action='store_true')
    parser.add_argument('--single_thread',
                        help='Run the solver single threaded',
                        action='store_true')
    parser.add_argument('--fashion',
                        help='Use fashion MNIST instead of MNIST',
                        action='store_true')
    parser.add_argument('--vnn',
                        help='Incoming inf is a vnn-conf testing file',
                        action='store_true')

    args = parser.parse_args()

    if args.timeout is None:
        args.timeout = 3600

    if args.timeout < 0:
        args.timeout = 3600 * 100

    runVerification(args.network, args.types, args.timeout, args.no_log,
                    args.inf, args.eps, args.relu7, args.debug, args.lateshift,
                    args.unsat_core, args.single_thread, args.vnn,
                    args.fashion)
