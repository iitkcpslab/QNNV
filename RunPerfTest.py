from argparse import ArgumentParser
import subprocess
import time
import os
import json
from multiprocessing import cpu_count
from utils import getFlattenedMNISTImages
from readNNet import readNNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.datasets import mnist

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('start',
                        help='Number of image to start working from.',
                        type=int)
    parser.add_argument('nSamples',
                        help='Number of samples to check.',
                        type=int)
    parser.add_argument('network', help='Network to test with, NNet file')
    parser.add_argument('types',
                        help='JSON file containing types for all operations')
    parser.add_argument('eps', help='Epsilon value to use')
    parser.add_argument('timeout', help='Timeout value', type=int)
    parser.add_argument('--single_thread',
                        help='Run the solver single threaded',
                        action='store_true')
    parser.add_argument('--fashion',
                        help='Run fashion mnist',
                        action='store_true')
    parser.add_argument('--vnnfiles',
                        help='Comma separated list of VNN-COMP instances')
    parser.add_argument('--gurobi', help='Not Run gurobi', action='store_true')
    parser.add_argument('--gurobi_par', help='Not Run gurobi parallel', action='store_true')
    parser.add_argument('--cbc', help='Not Run cbc', action='store_true')
    parser.add_argument('--cbc_par', help='Not Run cbc parallel', action='store_true')
    parser.add_argument('--glpk', help='Not Run glpk', action='store_true')
    args = parser.parse_args()

    print(
        "| Image Number | Gurobi verdict | Gurobi parallel verdict | CBC verdict | CBC parallel verdict | GLPK verdict | Gurobi Time | Gurobi parallel time | CBC Time | CBC parallel time | GLPK time | Epsilon |"
    ,flush=True)
    print("|-", flush=True)

    timeout = args.timeout

    with open(args.types, 'r') as f:
        typeData = json.load(f)
    weights, biases = readNNet(args.network)

    images = []
    numImages = 0
    if args.vnnfiles is not None and args.vnnfiles != "":
        images = args.vnnfiles.split(",")
        numImages = len(images)
    else:
        numImages = args.nSamples

    for i in range(numImages):
        start_time = time.perf_counter()

        if len(images) > 0:
            command = [
                'python3', 'FixedPointEncoder.py', args.network,
                str(args.types), images[i],
                str(args.eps), '--no_log', '--lateshift', '--timeout',
                str(timeout), '--vnn'
            ]
        else:
            command = [
                'python3', 'FixedPointEncoder.py', args.network,
                str(args.types),
                str(i),
                str(args.eps), '--no_log', '--lateshift', '--timeout',
                str(timeout)
            ]

        if args.single_thread:
            command.append('--single_thread')

        if args.fashion:
            command.append('--fashion')

        ret = subprocess.check_output(command, encoding='utf-8')
        end_time = time.perf_counter()

        gurobi_time = -1
        gurobi_verdict = '-'
        if not args.gurobi:
            command = [
                'gurobi_cl', 'TimeLimit={}'.format(timeout), 'Threads=1',
                'ConcurrentMIP=1', 'model.lp'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()

            gurobi_time = str(round(end_time - start_time, 2))
            gurobi_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Time limit reached' in ret else 'UNSAT')

        gurobi_parallel_time = -1
        gurobi_parallel_verdict = '-'
        if not args.gurobi_par:
            command = [
                'gurobi_cl', 'TimeLimit={}'.format(timeout),
                'Threads={}'.format(cpu_count()), 'LogFile=gurobi_par.log',
                'ConcurrentMIP={}'.format(cpu_count()), 'model.lp'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()

            gurobi_parallel_time = str(round(end_time - start_time, 2))
            gurobi_parallel_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Time limit reached' in ret else 'UNSAT')

        cbc_time = -1
        cbc_verdict = '-'
        if not args.cbc:
            command = [
                'cbc', '-sec',
                str(timeout), '-import', 'model.lp', '-solve'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()
            cbc_time = str(round(end_time - start_time, 2))
            cbc_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Stopped on time limit' in ret else 'UNSAT')

        cbc_parallel_time = -1
        cbc_parallel_verdict = '-'
        if not args.cbc_par:
            command = [
                'cbc', '-sec',
                str(timeout * cpu_count()), '-import', 'model.lp', '-threads',
                str(cpu_count()), '-solve'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()
            cbc_parallel_time = str(round(end_time - start_time, 2))
            cbc_parallel_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Stopped on time limit' in ret else 'UNSAT')

        glpk_time = -1
        glpk_verdict = '-'
        if not args.glpk:
            command = ['glpsol', '--tmlim', str(timeout), '--lp', 'model.lp']
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()
            glpk_time = str(round(end_time - start_time, 2))
            glpk_verdict = 'UNSAT' if "PROBLEM HAS NO INTEGER FEASIBLE SOLUTION" in ret else (
                'TIMEOUT' if 'TIME LIMIT EXCEEDED' in ret else 'SAT')

        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".
              format(images[i] if len(images) > 0 else i, gurobi_verdict, gurobi_parallel_verdict, cbc_verdict,
                     cbc_parallel_verdict, glpk_verdict, gurobi_time,
                     gurobi_parallel_time, cbc_time, cbc_parallel_time,
                     glpk_time, str(args.eps)), flush=True)
