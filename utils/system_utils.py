#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import sys
import subprocess

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def do_system(arg, exit=True):
    # err = os.system(arg)
    # if err:
    #     print("FATAL: command failed")
    #     if exit:
    #         sys.exit(err)
    try:
        result = subprocess.run(arg, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"==== running: {arg}")
            print(f"FATAL: command failed with error code {result.returncode}")
            print(f"Output:\n{result.stdout}")
            print(f"Error:\n{result.stderr}")
    except subprocess.TimeoutExpired as e:
        print(f"==== running: {arg}")
        print("FATAL: command timed out after 60 seconds.")
        print(f"Output (before timeout):\n{e.stdout.decode('utf-8') if e.stdout else 'No output'}")
        print(f"Error (before timeout):\n{e.stderr.decode('utf-8') if e.stderr else 'No error'}")
