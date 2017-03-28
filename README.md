# Flowlets
Converts KITTI tracklets into vectors denoting flow, for object tracking

by [Alvin Wan](http://alvinwan.com)

Flowlets are vectors between frames, from the center of a bounding box at time t
to a bounding box at time t+1. Each flowlet is associated with an object, for
all frames the involved object is detected.

The repository is written in Python 3.

# Installation

First, clone the repository. (Optional) Create a virtual environment, and
launch the environment.

    virtualenv flowlet --python=python3
    source flowlet/bin/activate

Let `$FHOME$` denote the repository root. Install all Python dependencies,
from the repository root.

    cd $FHOME
    pip install -r requirements.txt
    
Finally, use `python flowlets.py` per the usage instructions below. You
may additionally use `python flowlets.py --help` to retrieve instructions.

# Usage

The script is contained in the repository root.

```
Usage:
    flowlets.py 3d <path> [options]
    flowlets.py 2d <path> <calib_dir> [options]
    flowlets.py 3d KITTI (drive|all) [options]
    flowlets.py 2d KITTI <calib_dir> (drive|all) [options]

Options:
    --drive=<dir>       Drive identification. [default: ./]
    --kitti=<dir>       Path to KITTI data. [default: ./]
    --out=<dir>         Directory containing outputted files. [default: ./out]
    --mode=(obj|frame)  Output one file per object or one file per frame. [default: frame]
```
