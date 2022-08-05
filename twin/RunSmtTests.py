import subprocess
import os
import json
import time

if __name__ == '__main__':
    files = os.listdir()
    timedata = []
    timeout = 7200
    print("| File Name | Verdict | Time |", flush=True)
    print("|-", flush=True)
    for f in files:
        if f.endswith(".nnet"):
            os.chdir("../")
            command = [
                'python', 'ReduceIntegerPart.py', 'twin/' + f,
                str(4),
                str(4), '--noreduce', '--timestamp'
            ]
            typeFile = subprocess.check_output(command,
                                               encoding='utf-8').strip()
            os.system("mv {} /tmp".format(typeFile))

            os.system("cp twin/{} .".format(f))

            command = ['python', 'PysmtVerify.py', f, '--builtin']
            expectedVerdict = 'UNSAT' if 'UNSAT' in f else 'SAT'
            try:
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

                print("| {} | {} | {}s |".format(f, out, round(end - start,
                                                               2)), flush=True)
            except subprocess.TimeoutExpired:
                print("| {} | TIMEOUT | {}s |".format(f, timeout), flush=True)

            os.system("rm {}".format(f))
            os.chdir("twin/")

    with open('time.json', 'w') as f:
        json.dump(timedata, f)
