from glob import glob
import os
import re
import time
import psutil
import numpy as np
import random
import xlrd
from xlutils.copy import copy

# from check_solution import check_solution
# from efficacy_test import efficacy_test

# usability test suite
testSuite = ['./benchmarks/imp/network.txt']
seedSet = ['./benchmarks/imp/seeds.txt']
diff_set = ['IC', 'LT']

# Obtain all carp_solver scripts
solverScripts = dict()
noSolverPy = set()
dirList = os.listdir('./imp_extract/')
for stuId in dirList:
    fileList = glob('./imp_extract/%s/*.py' % stuId)
    if not fileList:
        noSolverPy.add(stuId)
        continue

    for file in fileList:
        if re.search(r'(I|i)(S|s)(E|e)', file):
            if stuId in solverScripts:
                if len(solverScripts[stuId]) > len(file):
                    solverScripts[stuId] = file
            else:
                solverScripts[stuId] = file
    if stuId not in solverScripts:
        noSolverPy.add(stuId)

outputDir = './imp_result/ISE_result/'
wrapper = './wrapper.py'
# Set test budget 10s
budget = 300
maxP = 10
par = 10
running_tasks = 0
subprocess = dict()
for stuId, script in solverScripts.items():
    for i, case in list(enumerate(testSuite)):
        for seed in seedSet:
            for diff_model in diff_set:
                while True:
                    if running_tasks >= par:
                        time.sleep(0.1)
                        finished = [pid for pid in subprocess if
                                    pid.poll() is not None]
                        for pid in finished:
                            subprocess.pop(pid)
                        running_tasks = len(subprocess)

                        outtime = [pid for pid in subprocess if time.time() -
                                   subprocess[pid] >= budget]
                        for pid in outtime:
                            while pid.poll() is None:
                                try:
                                    pid.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            subprocess.pop(pid)
                            running_tasks = len(subprocess)
                        continue

                    outputFile = '%s%s_case_%d_%s' % (outputDir, stuId, i, diff_model)

                    cmd = 'python2 %s -i %s -s %s -m %s -b %d -t %d -r %s > %s' % \
                          (script, case, seed, diff_model, 1, 300, 0, outputFile)
                    
#python ISE.py –i <social network> -s <seed set> -m <diffusionmodel> -b <termination type> -t <time budget> -r <random seed>
                    pid = psutil.Popen(cmd, shell=True)
                    subprocess[pid] = time.time()
                    running_tasks += 1

                    outtime = [pid for pid in subprocess if time.time() -
                               subprocess[pid] >= budget]
                    for pid in outtime:
                        while pid.poll() is None:
                            try:
                                pid.kill()
                            except psutil.NoSuchProcess:
                                pass
                        subprocess.pop(pid)
                        running_tasks = len(subprocess)
                    break
# waiting
while subprocess:
    time.sleep(0.1)
    finished = [pid for pid in subprocess if
                pid.poll() is not None]
    for pid in finished:
        subprocess.pop(pid)

    outtime = [pid for pid in subprocess if time.time() -
               subprocess[pid] >= budget]
    for pid in outtime:
        while pid.poll() is None:
            try:
                pid.kill()
            except psutil.NoSuchProcess:
                pass
            subprocess.pop(pid)

candidates = set()
option = 'ISE_result'
resultPath = './imp_result/%s/' % option
# result = np.zeros(0, dtype=[('id', 'U20'), ('u1', 'U20'),
#                             ('u2', 'U20'), ('u3', 'U20'),
#                             ('u4', 'U20'), ('u5', 'U20')])
# resultDict = {-1: 'Wrong Format', -2: 'Repeated tasks',
#               -3: 'Missing tasks', -4: 'capacity violated',
#               -5: 'quality incorrect'}

for stuId in solverScripts:
    q = np.zeros(1, dtype=[('id', 'U20'), ('u1', 'U20'),
                           ('u2', 'U20'), ('u3', 'U20'),
                           ('u4', 'U20'), ('u5', 'U20')])
    q['id'] = stuId
    for i, case in list(enumerate(testSuite)):
        for diff in diff_set:
            resultFile = '%s%s_case_%d_%s' % (resultPath, stuId, i, diff)
            with open(resultFile, 'r') as f:
                solution = False
                for line in f:
                    if line is not None:
                        solution = line.strip()

                wbk = xlrd.open_workbook('./imp_result/ISE_result.xls')
                wbk_new = copy(wbk)
                ws1 = wbk.sheets()[0]
                ws1_new = wbk_new.get_sheet(0)

                num_rows = ws1.nrows
                ws1_new.write(num_rows, 0, stuId)
                ws1_new.write(num_rows, 1, diff)
                if solution:
                    ws1_new.write(num_rows, 2, solution)
                else:
                    ws1_new.write(num_rows, 2, 'Failed to run')
                wbk_new.save(u'./imp_result/ISE_result.xls')
