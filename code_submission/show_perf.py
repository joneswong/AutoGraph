from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse

import numpy as np

parser = argparse.ArgumentParser("show performance")
parser.add_argument(
    '--num_exp', type=int, default=None, help='number of experiments')
parser.add_argument(
    '--res_path', type=str, default="./", help='the experimental results path')
args = parser.parse_args()


def main():
    exp_accs = list()
    exp_durs = list()
    exp_trials = list()
    acc_reg = re.compile("INFO score\.py: accuracy: (\d+\.\d*)")
    dur_reg = re.compile("INFO timing\.py: train_predict success, time spent (\d+\.\d*) sec")
    trial_reg = re.compile("INFO ensembler\.py: to select config\(s\) from (\d+) candidates")
    for i in range(args.num_exp):
        accs = list()
        durs = list()
        trials = list()
        with open("{}exp{}.out".format(args.res_path, i + 1), 'r') as ips:
            for line in ips:
                if "Timed out" in line:
                    break
                result = acc_reg.search(line)
                if result:
                    accs.append(float(result.group(1)))
                result = dur_reg.search(line)
                if result:
                    durs.append(float(result.group(1)))
                result = trial_reg.search(line)
                if result:
                    trials.append(float(result.group(1)))
        assert len(accs) == len(durs) and len(accs) == 10, "Invalid experiment results {} {} of {}-th exp!".format(
            len(accs), len(durs), i + 1)
        exp_accs.append(accs)
        exp_durs.append(durs)
        exp_trials.append(trials)

    mean_acc = np.mean(exp_accs, 0)
    mean_dur = np.mean(exp_durs, 0)
    std_acc = np.std(exp_accs, 0)
    std_dur = np.std(exp_durs, 0)
    mean_trials = np.mean(exp_trials, 0)
    std_trials = np.std(exp_trials, 0)

    print("====== accuracy ======")
    print("exp_idx\ta\tb\tc\td\te\tcomputers\tcora_full\togbn-arxiv\tphoto\tpubmed")
    for i in range(len(exp_accs)):
        print('\t'.join([str(i + 1)] + [str(v) for v in exp_accs[i]]))
    print('\t'.join(["mean"] + [str(v) for v in mean_acc]))
    print('\t'.join(["std"] + [str(v) for v in std_acc]))
    print()
    print("====== duration of train_predict() ======")
    print("exp_idx\ta\tb\tc\td\te\tcomputers\tcora_full\togbn-arxiv\tphoto\tpubmed")
    for i in range(len(exp_durs)):
        print('\t'.join([str(i + 1)] + [str(v) for v in exp_durs[i]]))
    print('\t'.join(["mean"] + [str(v) for v in mean_dur]))
    print('\t'.join(["std"] + [str(v) for v in std_dur]))
    print()
    print("====== trail times of train_predict() ======")
    print("exp_idx\ta\tb\tc\td\te\tcomputers\tcora_full\togbn-arxiv\tphoto\tpubmed")
    for i in range(len(exp_durs)):
        print('\t'.join([str(i + 1)] + [str(v) for v in exp_trials[i]]))
    print('\t'.join(["mean"] + [str(v) for v in mean_trials]))
    print('\t'.join(["std"] + [str(v) for v in std_trials]))


if __name__ == "__main__":
    main()
