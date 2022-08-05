from argparse import ArgumentParser
import subprocess
import os
import json
import time

if __name__ == '__main__':
    files = os.listdir()
    timedata = []
    timeout = 3600
    print("| File Name | Property | Result | Time |", flush=True)
    print("|-", flush=True)

    propertyFiles = sorted([f for f in files if f.endswith('.txt')])
    networkFiles = sorted([f for f in files if f.endswith('.nnet')])

    for f in networkFiles:
        if not f.endswith(".nnet"):
            continue

        os.chdir("../")
        command = [
            'python', 'ReduceIntegerPart.py', 'acasxu/' + f,
            str(4),
            str(4), '--noreduce', '--timestamp'
        ]

        typeFile = subprocess.check_output(command, encoding='utf-8').strip()
        os.system("mv {} /tmp".format(typeFile))
        os.system("cp {} .".format('acasxu/' + f))

        for pf in propertyFiles:
            try:
                command = [
                    'python', 'PysmtVerify.py', f, '--prop', 'acasxu/' + pf
                ]
                start = time.perf_counter()
                ret = subprocess.check_output(command,
                                              encoding='utf-8',
                                              timeout=timeout)
                end = time.perf_counter()
                timedata.append(end - start)
                out = ""
                if "UNSAT" in ret:
                    out = "UNSAT"
                elif "SAT" in ret:
                    out = "SAT"
                elif "TIME_LIMIT" in ret:
                    out = "TIMEOUT"

                print("| {} | {} | {} | {} |".format(f, pf, out,
                                                     round(end - start, 2)), flush=True)
            except subprocess.TimeoutExpired:
                print("| {} | {} | TIMEOUT | {} |".format(f, pf, timeout), flush=True)

        os.system("rm {}".format(f))
        os.chdir("acasxu/")

    with open('time.json', 'w') as f:
        json.dump(timedata, f)
