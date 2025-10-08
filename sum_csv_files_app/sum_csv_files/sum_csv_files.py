#!/usr/bin/env python3
"""
2) would it be possible to allow users to cut small rectangular boxes (3 to 6)
out of a single csv file - edit those files, then add them up afterwards - these
would be matrix boxes for different ion charge states to which I'll need to
adjust for their PAPS&DV varying efficiencies. It would be very, very helpful
and useful - and save a lot of time - I would be happy to test it with spot
checks, hand calculation, etc. as much as needed in the process. 
"""

import os
import re
import sys

import pandas as pd
from pathlib import Path


def main(args):
    print(args[1])
    if len(args) < 2:
        args.extend(Path(os.getcwd()).parents[2].glob('*.csv'))
    if len(args) < 2:
        return
    with open(args[1], 'rb') as csvfile:
        l1 = csvfile.readline().decode()
        l2 = csvfile.readline().decode()
        l3 = csvfile.readline().decode()
        csv_sum = pd.read_csv(csvfile, delimiter=',', index_col=0)
        if re.findall(r' PHAs of [0-9,]+ total', l2):
            totsPHAs = int(re.findall(r'[0-9,]+', l2)[-1].replace(',', ''))
        else:
            totsPHAs = 0
    for csvarg in args[2:]:
        with open(csvarg, 'rb') as csvfile:
            _ = csvfile.readline().decode()
            title = csvfile.readline().decode()
            _ = csvfile.readline().decode()
            csv_sum += pd.read_csv(csvfile, delimiter=',', index_col=0)
            if re.findall(r' PHAs of [0-9,]+ total', title):
                totsPHAs += int(re.findall(r'[0-9,]+', title)[-1].replace(',', ''))
    with open(Path(args[1]).with_stem('sum'), 'wb') as outfile:
        outfile.write(l1.encode())
        if totsPHAs > 0:
            # update total PHAs with the sum of totalPHAs
            l2 = re.sub(r' PHAs of [0-9,]+ total', f' PHAs of {totsPHAs:,} total', l2)
        outfile.write(l2.encode())
        outfile.write(l3.encode())
        csv_sum.to_csv(outfile)
        # np.savetxt(outfile, csv_sum, fmt='%.4e', delimiter=',')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
