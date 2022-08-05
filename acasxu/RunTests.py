from argparse import ArgumentParser
from multiprocessing import cpu_count
import subprocess
import os
import json
import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('int', help='Number of bits in integer part', type=int)
    parser.add_argument('float',
                        help='Number of bits in fractional part',
                        type=int)
    args = parser.parse_args()

    timeout = 3600

    files = os.listdir()
    timedata = []
    print(
        "| File Name | Property | Gurobi verdict | Gurobi parallel verdict | GLPK verdict | Gurobi Time | Gurobi parallel time | GLPK Time |",
        flush=True)
    print("|-", flush=True)

    propertyFiles = sorted([f for f in files if f.endswith('.txt')])
    networkFiles = sorted([f for f in files if f.endswith('.nnet')])

    for f in networkFiles:
        if not f.endswith(".nnet"):
            continue

        os.chdir("../")
        command = [
            'python', 'ReduceIntegerPart.py', 'acasxu/' + f,
            str(args.int),
            str(args.float), '--noreduce', '--timestamp'
        ]

        typeFile = subprocess.check_output(command, encoding='utf-8').strip()
        os.system("mv {} /tmp".format(typeFile))
        os.system("cp {} .".format('acasxu/' + f))

        for pf in propertyFiles:
            command = [
                'python', 'FixedPointPropertyEncoder.py', f,
                '/tmp/' + typeFile, '--debug', '--lateshift', '--prop',
                'acasxu/' + pf
            ]
            start = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end = time.perf_counter()
            timedata.append(end - start)
            command = [
                'gurobi_cl', 'TimeLimit={}'.format(timeout), 'Threads=1',
                'ConcurrentMIP=1', 'model.lp'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()

            gurobi_time = str(round(end_time - start_time, 3))
            gurobi_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Time limit reached' in ret else 'UNSAT')

            command = [
                'gurobi_cl', 'TimeLimit={}'.format(timeout),
                'Threads={}'.format(cpu_count()),
                'ConcurrentMIP={}'.format(cpu_count()), 'model.lp'
            ]
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()

            gurobi_parallel_time = str(round(end_time - start_time, 2))
            gurobi_parallel_verdict = 'SAT' if "Optimal solution found" in ret else (
                'TIMEOUT' if 'Time limit reached' in ret else 'UNSAT')

            command = ['glpsol', '--tmlim', str(timeout), '--lp', 'model.lp']
            start_time = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end_time = time.perf_counter()
            glpk_time = str(round(end_time - start_time, 2))
            glpk_verdict = 'UNSAT' if "PROBLEM HAS NO INTEGER FEASIBLE SOLUTION" in ret else (
                'TIMEOUT' if 'TIME LIMIT EXCEEDED' in ret else 'SAT')
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, pf, gurobi_verdict, gurobi_parallel_verdict, glpk_verdict,
                gurobi_time, gurobi_parallel_time, glpk_time),
                flush=True)

        os.system("rm {}".format(f))
        os.chdir("acasxu/")

    with open('time.json', 'w') as f:
        json.dump(timedata, f)
