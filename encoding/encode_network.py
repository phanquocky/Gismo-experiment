# encoding: utf-8
"""
Copyright (C) 2022 Anna L.D. Latour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

@author: Anna L.D. Latour
@contact: latour@nus.edu.sg
@time: 15 Oct 2022
@file: encode_network.py
@desc: Experiment script for encoding networks into CNF or ILP for solving the
       Identifying Codes problem that is formulated on those networks.
"""

import argparse
from datetime import datetime
import os
import pathlib
import signal
import sys
from ilp_encoding import ILPEncoding
from gis_encoding import GISEncoding

PROJECT_DIR = os.getenv('PROJECT_DIR')

sys.path.insert(1, '{PROJECT_DIR}/scripts/helpers'.format(PROJECT_DIR=PROJECT_DIR))
from timer import WallclockTimer, ProcessTimer

parser = argparse.ArgumentParser()
required_args = parser.add_argument_group("Required arguments")
optional_args = parser.add_argument_group("Optional arguments")
required_args.add_argument("--network", "-n", type=str, required=True,
                           help="Path to network file.")
required_args.add_argument("--out_dir", type=str, required=True,
                           help="Path to output directory above k sub directory.")
required_args.add_argument("--out_file", type=str, required=True,
                           help="Basename of output file.")
required_args.add_argument("--encoding", type=str, required=True,
                           choices=['maxsat', 'gis', 'ilp', 'sat', 'pb'],
                           help='Specify the encoding')
optional_args.add_argument("-b", type=int, required=False, default=-1,
                           help="Budget (number of smoke detectors / injected colors).")
optional_args.add_argument("-k", type=int, default=1,
                           help="Max number of simultaneous events.")
optional_args.add_argument("--two_step", required=False,
                           default=False, action="store_true",
                           help="Request two_step approach.")
optional_args.add_argument("--remove_supersets", required=False,
                           default=False, action="store_true",
                           help="For ILP encoding only: remove redundant constraints.")
optional_args.add_argument("--check_2_neighbourhood", required=False,
                           default=False, action="store_true",
                           help="For ILP encoding only: avoid adding unnecessary constraints.")

args = parser.parse_args()

encoding_settings = dict()
if args.encoding == 'ilp':
    encoding_settings['remove_supersets'] = args.remove_supersets
    encoding_settings['check_2_neighbourhood'] = args.check_2_neighbourhood

def handler(signum, frame):
    print("Timed out!")
    raise Exception("Timed out!")


def log_message(message):
    print('{date}: {message}'.format(
        date=datetime.now().strftime("%Y-%m-%d, %Hh%Mm%Ss"), message=message))


# Build and encode problem
log_message("Processing {network}".format(network=args.network))
log_message("Initialising {encoding} instance".format(encoding=args.encoding))
ic_instance = None

if args.encoding == 'gis':
    ic_instance = GISEncoding()
elif args.encoding == 'ilp':
    ic_instance = ILPEncoding()

log_message("Building {encoding} instance.".format(encoding=args.encoding))
build_successful = True

sys.stdout.flush()

t_wallclock = WallclockTimer(text="Building took {0:.4f} wallclock seconds.")
t_process = ProcessTimer(text="Building took {0:.4f} CPU seconds.")
try:
    t_wallclock.start()
    t_process.start()
    ic_instance.build_from_file(args.network,
                                budget=args.b,
                                two_step=args.two_step)
    log_message(t_wallclock.stop())
    log_message(t_process.stop())
    log_message("Building completed!")
except Exception as exc:
    build_successful = False
    log_message("Building FAILED!")
    log_message(exc)
    log_message(t_wallclock.stop())
    log_message(t_process.stop())

sys.stdout.flush()

if build_successful:
    log_message("Encoding {encoding} instance.".format(encoding=args.encoding))
    k = args.k
    log_message("Encoding k = {k}".format(k=k))
    t_wallclock = WallclockTimer(text="Encoding took {0:.4f} wallclock seconds for k = " + str(k) + ".")
    t_process = ProcessTimer(text="Encoding took {0:.4f} CPU seconds for k = " + str(k) + ".")
    out_dir = '{out_dir}/k{k}/'.format(out_dir=args.out_dir, k=k)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        t_wallclock.start()
        t_process.start()
        ic_instance.encode(out_dir + args.out_file, k, **encoding_settings)
        log_message(t_wallclock.stop())
        log_message(t_process.stop())
        log_message("Encoding completed!")
    except Exception as exc:
        log_message("Encoding FAILED!")
        log_message(exc)
        log_message(t_wallclock.stop())
        log_message(t_process.stop())
else:
    log_message("Building failed. Aborting rest of the process")

sys.stdout.flush()

log_message("Done!")
