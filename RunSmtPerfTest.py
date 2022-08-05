from tensorflow.keras.datasets import mnist
from argparse import ArgumentParser
import subprocess
import time
import os
import json
from utils import getFlattenedMNISTImages
from readNNet import readNNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('start',
                        help='Number of image to start working from.',
                        type=int)
    parser.add_argument('nSamples',
                        help='Number of samples to check.',
                        type=int)
    parser.add_argument('types',
                        help='JSON file containing types for all operations')
    parser.add_argument('network', help='Network to test with, NNet file')
    parser.add_argument('eps', help='Epsilon value to use')
    parser.add_argument('timeout', help='Timeout value', type=int)
    parser.add_argument('--fashion',
                        help='Use fashion MNIST',
                        action='store_true')
    args = parser.parse_args()

    print("SMT Results", flush=True)
    print("| Image Number | Time | Epsilon | Verdict |", flush=True)
    print("|-", flush=True)

    timeout = args.timeout

    (_, _), (xTest, yTest) = getFlattenedMNISTImages(args.fashion)
    weights, biases = readNNet(args.network)

    for i in range(args.start, args.nSamples + args.start):
        start_time = time.perf_counter()
        command = [
            'python', 'PysmtVerify.py', args.network, '--inf',
            str(i), '--eps',
            str(args.eps)
        ]

        if args.fashion:
            command.append('--fashion')

        try:
            ret = subprocess.check_output(command,
                                          encoding='utf-8',
                                          timeout=timeout)
            end_time = time.perf_counter()
            verdict = 'UNSAT' if 'UNSAT' in ret else 'SAT'

            time_output = str(round(end_time - start_time, 2))
            print("| {} | {} | {} | {} |".format(i, time_output + 's',
                                                 args.eps, verdict), flush=True)
        except subprocess.TimeoutExpired:
            print("| {} | {} | {} | TIMEOUT |".format(i,
                                                      str(timeout) + 's',
                                                      args.eps), flush=True)
