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
    
If you're only looking to convert tracklets to flowlets, run the
following download script. Otherwise, please see the KITTI website for
a full dataset download.

```
bash tracklet_calib_data_downloader.sh
```

Finally, use `python flowlets.py` per the usage instructions below. You
may additionally use `python flowlets.py --help` to retrieve instructions.

# Usage

The script is contained in the repository root. To, for example, convert
all tracklets into flowlets, run the following, where `/KITTI_raw` is
the directory containing our raw KITTI data, and `~/flowlets` should
contain all outputs.

```
python flowlets.py KITTI /KITII_raw --out=~/flowlets
```

```
Usage:
    flowlets.py <path> [options]
    flowlets.py KITTI <kitti_dir> [options]
    flowlets.py KITTI <kitti_dir> <date> <drive_id> [options]

Options:
    --d=<dims>          Number of dimensions [default: 2]
    --out=<dir>         Directory containing outputted files. [default: ./out]
    --mode=(obj|frame)  Output one file per object or one file per frame. [default: frame]
    --columns=<columns> Specify order of columns
```
