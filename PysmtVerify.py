import sys
from tensorflow.keras.datasets import mnist, fashion_mnist
import json
import time
import re
from argparse import ArgumentParser
from relu7_model import Relu7Model
from pysmt.shortcuts import SFXPAdd, SFXPSub, SFXPMul, SFXPLT, SFXPGT, SFXPLE, SFXPGE, Equals, SFXP, ST, RU, BV, Equals, Symbol, get_model, is_sat, And, Implies, Or, Not, SFXPDiv, get_env, Solver, RD
from pysmt.rewritings import get_fp_bv_converter
from quantization_util import quantize, de_quantize
from pysmt.logics import QF_BV
from readNNet import readNNet
import numpy as np
from fxp_nn_encoding import FXPencoding
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


sys.setrecursionlimit(10000)


def id(env, x):
    return x


def neg(env, x):
    return SFXPMul(ST, RD, env.const(-1), x)


def normalize(v, m, s):
    return (v - m) / s


def getVar(env: FXPencoding, part):
    if part.startswith('x'):
        return env.input_symbols[int(part[1:])]
    elif part.startswith('y'):
        return env.output_symbols[int(part[1:])]
    else:
        return float(part)


def addThreeElementConstraint(env, prop):
    parts = prop.split(" ")
    # assuming that these can never be a constant
    var = getVar(env, parts[0])
    constr = SFXPLE if parts[1] == "<=" else SFXPGE
    # assuming that these can never be a variable
    otherVar = env.const(getVar(env, parts[2]))

    env.assertion(constr(var, otherVar))


def addFourElementConstraint(env, prop):
    parts = prop.split(" ")
    fSign = id if parts[0].startswith('+') else neg
    sSign = id if parts[1].startswith('+') else neg
    parts[0] = parts[0][1:]
    parts[1] = parts[1][1:]

    # assuming that these can never be a constant
    leftVar = getVar(env, parts[0])
    rightVar = getVar(env, parts[1])

    constr = SFXPLE if parts[2] == '<=' else SFXPGE
    # assuming that these can never be a variable
    resVar = env.const(getVar(env, parts[3]))

    env.assertion(
        constr(SFXPAdd(ST, fSign(env, leftVar), sSign(env, rightVar)), resVar))


def addConstraints(env: FXPencoding, props):
    for p in props:
        if len(p.split(" ")) == 3:
            addThreeElementConstraint(env, p)
        else:
            addFourElementConstraint(env, p)


parser = ArgumentParser()
parser.add_argument('network', help='Neural network to test, NNet format')
parser.add_argument('--inf', help='Number of test image to verify', type=int)
parser.add_argument('--eps',
                    help='Amount of deviation in floating-point world',
                    type=float)
parser.add_argument('--builtin',
                    help='Use the builtin property',
                    action='store_true')
parser.add_argument('--prop', help='Use the property file to add constraints')
parser.add_argument('--fashion', help='Use fashion MNIST', action='store_true')
args = parser.parse_args()

model = Relu7Model()

weights, biases, inputMins, inputMaxes, means, stds = readNNet(args.network,
                                                               withNorm=True)

model.weights = [np.transpose(w) for w in weights]
model.biases = biases

env = FXPencoding(model)
conjunctions = env.conjunctions

if args.builtin:
    normalizedMins = [
        normalize(v, m, std) for v, m, std in zip(inputMins, means, stds)
    ]
    normalizedMaxes = [
        normalize(v, m, std) for v, m, std in zip(inputMaxes, means, stds)
    ]
    fpMins = [env.const(v) for v in normalizedMins]
    fpMaxes = [env.const(v) for v in normalizedMaxes]
    for ip, mn, mx in zip(env.input_symbols, fpMins, fpMaxes):
        conjunctions.append(SFXPGE(ip, mn))
        conjunctions.append(SFXPLE(ip, mx))
    conjunctions.append(SFXPLE(env.output_symbols[0], env.const(0)))
elif args.prop is not None:
    if args.prop.endswith('.vnnlib'):
        target = -1
        comment_re = re.compile("; Mnist property with label: (.*?)\\.")
        assert_re = re.compile(r"\(assert \((.*?) (.*?) (.*?)\)\)")

        with open(args.prop, 'r') as f:
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

                integerPart, floatPart = 4, 4

                if sign == '<=':
                    conjunctions.append(
                        SFXPLE(env.input_symbols[input], env.const(bound)))
                elif sign == '>=':
                    conjunctions.append(
                        SFXPGE(env.input_symbols[input], env.const(bound)))

        if target == -1:
            raise Exception("could not find the original image type")

        outConstr = [
            SFXPGT(env.output_symbols[i], env.output_symbols[target])
            for i in range(10) if i != target
        ]
        conjunctions.append(Or(*outConstr))

    else:
        with open(args.prop, 'r') as f:
            props = [x.strip() for x in f.readlines()]

        addConstraints(env, props)
else:
    if args.inf is None or args.eps is None:
        raise Exception("Please provide --inf and --eps")

    if args.fashion:
        (_, _), (xTest, yTest) = fashion_mnist.load_data()
        print("using fashion")
    else:
        (_, _), (xTest, yTest) = mnist.load_data()

    for ip, px in zip(env.input_symbols, xTest[args.inf].flatten()):
        conjunctions.append(SFXPGE(ip, env.const(px / 256 - args.eps)))
        conjunctions.append(SFXPLE(ip, env.const(px / 256 + args.eps)))

    y = yTest[args.inf]

    outConstr = [
        SFXPGT(env.output_symbols[i], env.output_symbols[y]) for i in range(10)
        if i != y
    ]
    conjunctions.append(Or(*outConstr))

conv = get_fp_bv_converter()
bv_cons = conv.convert(And(conjunctions))

start = time.perf_counter()
res = get_model(bv_cons, solver_name="btor", logic=QF_BV)
end = time.perf_counter()

if args.prop is not None and args.prop.endswith('.vnnlib'):
    print("| {} | {} | {} |".format(args.prop, str(round(end - start, 2)),
                                    "SAT" if res else "UNSAT"))
    exit(0)

if res:
    print("SAT")
    inputs = []
    for i in range(len(env.clipped_symbols)):
        bv_value = res.get_value(conv.symbol_map[env.clipped_symbols[i]])
        us_value = int(bv_value.bv_signed_value())
        float_value = de_quantize(us_value, num_bits=env._fractional_digits)
        inputs.append(float_value)
    inputs = np.array(inputs)

    outputs = []
    for i in range(len(env.output_symbols)):
        bv_value = res.get_value(conv.symbol_map[env.output_symbols[i]])
        us_value = int(bv_value.bv_signed_value())
        float_value = de_quantize(us_value, num_bits=env._fractional_digits)
        outputs.append(float_value)
    outputs = np.array(outputs)

    allVars = []
    for i in range(len(env.all_symbols)):
        bv_value = res.get_value(conv.symbol_map[env.all_symbols[i]])
        us_value = int(bv_value.bv_signed_value())
        float_value = de_quantize(us_value, num_bits=env._fractional_digits)
        allVars.append("{} = {} bv = {} us = {}".format(
            env.all_symbols[i].symbol_name(), float_value, bv_value, us_value))

else:
    print('UNSAT')
