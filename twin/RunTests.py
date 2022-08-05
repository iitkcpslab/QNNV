from argparse import ArgumentParser
import subprocess
import os
import json
import time
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('int', help='Number of bits in integer part', type=int)
    parser.add_argument('float',
                        help='Number of bits in fractional part',
                        type=int)
    args = parser.parse_args()

    timeout = 7200

    files = os.listdir()
    timedata = []
    print(
        "| File Name | Expected Verditct | Gurobi verdict | Gurobi parallel verdict | GLPK verdict | Gurobi Time | Gurobi parallel time | GLPK Time |",
        flush=True)
    print("|-", flush=True)
    for f in files:
        if f.endswith(".nnet"):
            os.chdir("../")
            command = [
                'python', 'ReduceIntegerPart.py', 'twin/' + f,
                str(args.int),
                str(args.float), '--noreduce', '--timestamp'
            ]
            typeFile = subprocess.check_output(command,
                                               encoding='utf-8').strip()
            os.system("mv {} /tmp".format(typeFile))

            os.system("cp twin/{} .".format(f))

            command = [
                'python', 'FixedPointPropertyEncoder.py', f,
                '/tmp/' + typeFile, '--lateshift', '--timeout', '7200'
            ]
            start = time.perf_counter()
            ret = subprocess.check_output(command, encoding='utf-8')
            end = time.perf_counter()
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

            timedata.append(end - start)

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
            expectedVerdict = 'UNSAT' if 'UNSAT' in f else 'SAT'
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                f, expectedVerdict, gurobi_verdict, gurobi_parallel_verdict,
                glpk_verdict, gurobi_time, gurobi_parallel_time, glpk_time),
                flush=True)
            os.system("rm {}".format(f))
            os.chdir("twin/")

    with open('time.json', 'w') as f:
        json.dump(timedata, f)
