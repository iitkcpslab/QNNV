from argparse import ArgumentParser
from FixedPointEncoder import FixedPointEncoder, encodeNetwork, setupModel, convertToDifferentFP
from FixedPointEncoder import processModelOutput
from FixedPointData import FixedPointData
from readNNet import readNNet
from operator import le, ge, neg
import json
from gurobipy import GRB


def id(x):
    return x


def normalize(v, m, std):
    return (v - m) / std


def arrayToFixedPoint(arr, typeData):
    return [
        FixedPointData(ip + fp).convert(inp, fp)
        for inp, [ip, fp] in zip(arr, typeData)
    ]


def getVar(encoder: FixedPointEncoder, part):
    if part.startswith('x'):
        return encoder.inputVars[int(part[1:])]
    elif part.startswith('y'):
        return encoder.outputVars[int(part[1:])]
    else:
        return float(part)


def getType(encoder: FixedPointEncoder, part):
    if part.startswith('x'):
        return encoder.typeData[0][int(part[1:])]
    elif part.startswith('y'):
        return encoder.typeData[-1][int(part[1:])]
    else:
        raise Exception("Asking for type of a constant")


def addThreeElementConstraint(encoder: FixedPointEncoder, prop):
    parts = prop.split(" ")
    # assuming that these can never be a constant
    var = getVar(encoder, parts[0])
    typeData = getType(encoder, parts[0])
    constr = le if parts[1] == '<=' else ge
    # assuming that these can never be a variable
    otherVar = getVar(encoder, parts[2])
    family = FixedPointData(sum(typeData))
    otherVar = family.convert(otherVar, typeData[1])

    encoder.model.addConstr(constr(var, otherVar))


def addFourElementConstraint(encoder: FixedPointEncoder, prop):
    parts = prop.split(" ")
    fSign = id if parts[0].startswith('+') else neg
    sSign = id if parts[1].startswith('+') else neg
    parts[0] = parts[0][1:]
    parts[1] = parts[1][1:]

    # assuming that these can never be a constant
    leftVar = getVar(encoder, parts[0])
    leftType = getType(encoder, parts[0])
    # assuming that these can never be a constant
    rightVar = getVar(encoder, parts[1])
    rightType = getType(encoder, parts[1])

    maxFloat = max(leftType[1], rightType[1])

    if leftType[1] != rightType[1]:
        leftVar = convertToDifferentFP(encoder, leftVar, leftVar.VarName,
                                       leftType[0], leftType[1], maxFloat)
        rightVar = convertToDifferentFP(encoder, rightVar, rightVar.VarName,
                                        rightType[0], rightType[1], maxFloat)

    constr = le if parts[2] == '<=' else ge
    # assuming that these can never be a variable
    resVar = getVar(encoder, parts[3])
    family = FixedPointData(min(leftType[0], rightType[0]) + maxFloat)
    resVar = family.convert(resVar, maxFloat)

    encoder.model.addConstr(constr(fSign(leftVar) + sSign(rightVar), resVar))


def addConstraints(encoder: FixedPointEncoder, props):
    for p in props:
        if len(p.split(" ")) == 3:
            addThreeElementConstraint(encoder, p)
        else:
            addFourElementConstraint(encoder, p)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('network', help='Neural network to test, NNet file')
    parser.add_argument('types', help='Types for each node, JSON file')
    parser.add_argument('--timeout',
                        help='Maximum time budget (default is 3600s)',
                        type=int)
    parser.add_argument('--debug',
                        help='Add debugging parameters to solver',
                        action='store_true')
    parser.add_argument('--lateshift',
                        help='Use Google\'s late shift encoding',
                        action='store_true')
    parser.add_argument('--unsat_core',
                        help='Generate UNSAT cores',
                        action='store_true')
    parser.add_argument(
        '--no_log',
        help='Don\'t print log information to standard out, print to file "problem.log"',
        action='store_true')
    parser.add_argument('--single_thread',
                        help='Run the solver with single thread',
                        action='store_true')
    parser.add_argument('--prop',
                        help='Property to verify (default: builtin property)')
    args = parser.parse_args()

    timeout = 3600
    if args.timeout is not None:
        timeout = args.timeout

    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(
        args.network, withNorm=True)

    with open(args.types, 'r') as f:
        typeData = json.load(f)

    encoder = FixedPointEncoder(args.network, weights, biases, typeData, False,
                                args.lateshift, args.unsat_core)
    encoder = encodeNetwork(encoder)

    normalizedMins = [
        normalize(v, m, std) for v, m, std in zip(inputMins, means, ranges)
    ]
    normalizedMaxes = [
        normalize(v, m, std) for v, m, std in zip(inputMaxes, means, ranges)
    ]

    fpMins = arrayToFixedPoint(normalizedMins, typeData[0])
    fpMaxes = arrayToFixedPoint(normalizedMaxes, typeData[0])

    for inp, mn, mx in zip(encoder.inputVars, fpMins, fpMaxes):
        encoder.model.addConstr(inp == [mn, mx])

    if args.prop is None:
        encoder.model.addConstr(encoder.outputVars[0] <= 0)
    else:
        with open(args.prop, 'r') as f:
            props = [x.strip() for x in f.readlines()]

        addConstraints(encoder, props)

    logFile = None
    if args.no_log:
        logFile = "problem.log"

    encoder.model.setObjective(encoder.outputVars[0], GRB.MAXIMIZE)

    setupModel(encoder.model, timeout, logFile, args.debug, args.single_thread)

    encoder.model.write('model.lp')
    encoder.model.write('model.mps')

    with open('model.lp', 'r') as f:
        content = f.read()

    print("Replacing", encoder.outputVars[0].VarName,
          "with ", "0 " + encoder.outputVars[0].VarName)
    content = content.replace(
        encoder.outputVars[0].VarName, "0 " + encoder.outputVars[0].VarName, 1)

    with open('model.lp', 'w') as f:
        f.write(content)

    print("*** Total Vars = {} ***".format(len(encoder.model.getVars())))
