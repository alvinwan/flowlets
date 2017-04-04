"""Converts trackelets to flowlets.

Flowlets are vectors between frames, from the center of a bounding box at time t
to a bounding box at time t+1. Each flowlet is associated with an object, for
all frames the involved object is detected.

The columns argument below allows you to organize, include, or omit columns
as you wish. The following are available. Note that if the 2D option is
selected, only the (x, y) below apply:

    x - x coordinate of the bounding box center
    y - y coordinate of the bounding box center
    z - z coordinate of the bounding box center
    h - height of the bounding box
    w - width of the bounding box
    l - length of the bounding box
    dx - x coordinate of the delta vector, tracking movement of the bounding box
    dy - y coordinate of the delta vector, tracking movement of the bounding box
    dz - z coordinate of the delta vector, tracking movement of the bounding box
    class - class of the object

Usage:
    flowlets.py <path> [options]
    flowlets.py KITTI <kitti_dir> [options]
    flowlets.py KITTI <kitti_dir> <date> <drive_id> [options]

Options:
    --d=<dims>          Number of dimensions [default: 2]
    --out=<dir>         Directory containing outputted files. [default: ./out]
    --mode=(obj|frame)  Output one file per object or one file per frame. [default: frame]
    --columns=<columns> Specify order of columns, comma-separated values
"""

import os

from collections import defaultdict
from os.path import join
from os.path import isdir
from os.path import basename
from os.path import exists
from typing import Dict
from typing import List

import docopt
import numpy as np
import scipy.io

from thirdparty.calib import Calib
from thirdparty.parseTrackletXML import parseXML


DEFAULT_PATH_FORMAT = '{date}_{drive_id}_{frame_id}.npy'
DEFAULT_2D_COLUMNS = 'x,y,w,h,dx,dy,class'
DEFAULT_3D_COLUMNS = 'x,y,z,w,h,l,dx,dy,dz,class'
OBJECT_TYPES = ('Car', 'Cyclist', 'Pedestrian', 'Van', 'Tram',
                'Person (sitting)', 'Truck', 'Misc')


def main():
    """Run the main utility."""
    arguments = docopt.docopt(__doc__, version='Flowlets 1.0')
    dimensions = int(arguments['--d'])
    default_columns = DEFAULT_3D_COLUMNS if dimensions == 3 \
        else DEFAULT_2D_COLUMNS

    if arguments['<path>']:
        tbundles = [extract_tracklets_bundle(arguments['<path>'])]
    elif arguments['<drive_id>']:
        tbundle = extract_drive_tracklets_bundle(
            kitti=arguments['<kitti_dir>'],
            date=arguments['<date>'],
            drive_id=arguments['<drive_id>'])
        tbundles = [tbundle]
    else:
        tbundles = extract_all_tracklets_bundles(kitti=arguments['<kitti_dir>'])

    for tbundle in tbundles:
        print(' * [INFO] Converting...')
        fbundle = flowletize(
            tbundle,
            calib_dir=join(arguments['<kitti_dir>'], tbundle['date']),
            dimensions=dimensions)
        print(' * [INFO] Writing...')
        output(
            arguments['--out'],
            fbundle=fbundle,
            mode=arguments['--mode'],
            columns=default_columns)


def extract_all_tracklets_bundles(kitti: str) -> List[Dict]:
    """Extract all KITTI tracklets."""
    flowlets = []
    dates = [path for path in os.listdir(kitti) if isdir(join(kitti, path))]
    for date in dates:
        drive_dir_path = join(kitti, date)
        drive_dirs = [path for path in os.listdir(drive_dir_path) if
                      isdir(join(drive_dir_path, path))]
        for drive_dir in drive_dirs:
            drive_id = drive_dir.split('_')[-2]
            flowlets.append(extract_drive_tracklets_bundle(
                kitti=kitti, date=date, drive_id=drive_id))
    return flowlets


def extract_drive_tracklets_bundle(
        kitti: str,
        date: str,
        drive_id: str,
        drive_path: str = None) -> Dict:
    """Extract all KITTI tracklet for a drive."""
    if drive_path is None:
        drive_dir = '_'.join([date, 'drive', drive_id, 'sync'])
        drive_path = join(kitti, date, drive_dir)
    filepath = join(drive_path, 'tracklet_labels.xml')
    return extract_tracklets_bundle(filepath, date=date, drive_id=drive_id)


def extract_tracklets_bundle(path: str, date: str = None, drive_id: str = None) \
        -> Dict:
    """Extract tracklets and cartesian coordinates from KITTI tracklet file.

    This function was adapted from the example method in parseTrackletXML.py.
    (See file for credits.)

    If the date and drive_ids are not specified, this assumes that the date
    and drive_id are supplmented by the path, where the directory containing the
    XML file is formatted as {date}_{drive_id}_{frame_id}_sync/.
    """
    assert basename(path) == 'tracklet_labels.xml', 'Unexpected filename.'
    print(' * Extracting', path)

    if date is None or drive_id is None:
        drive_dir = path.split('/')
        drive_data = drive_dir[-2].split('_')
        date = '_'.join(drive_data[:3])
        drive_id = drive_data[-2]

    twoPi = 2. * np.pi  # read tracklets from file
    tracklets_raw = parseXML(path)
    tracklets = []

    # loop over tracklets
    for iTracklet, tracklet in enumerate(tracklets_raw):
        print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        # this part is inspired by kitti object development kit matlab code:
        # computeBox3D
        h, w, l = tracklet.size
        trackletBox = np.array([
            # in velodyne coordinates around zero point and without
            # orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
        boxes, xs, ys, zs, yawVisuals = [], [], [], [], []
        tracklets.append({
            'firstFrame': tracklet.firstFrame,
            'nFrames': tracklet.nFrames,
            'class': OBJECT_TYPES.index(tracklet.objectType),
            'h': h,
            'w': w,
            'l': l,
            'xs': xs,
            'ys': ys,
            'zs': zs,
            'yawVisuals': yawVisuals,
            'boxes': boxes
        })

        # loop over all frames in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, \
            amtBorders, absoluteFrameNumber in tracklet:
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are 0 in all xml files
            assert np.abs(
                rotation[:2]).sum() == 0, \
                'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + \
                              np.tile(translation, (8, 1)).T

            # calc yaw as seen from the camera (i.e. 0 degree = facing away from
            # cam), as opposed to car-centered yaw (i.e. 0 degree =
            # same orientation as car). makes quite a difference for objects in
            # periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = (yaw - np.arctan2(y, x)) % twoPi
            xs.append(x)
            ys.append(y)
            zs.append(z)
            yawVisuals.append(yawVisual)

            boxes.append(cornerPosInVelo)
    return {'date': date, 'drive_id': drive_id, 'tracklets': tracklets}


def flowletize(
        bundle: Dict,
        calib_dir: str=None,
        dimensions: int=3,
        cam_idx: int=2,) -> Dict:
    """Vectorizes the list of cartesian coordinates.

    Returns a list of tuples, one number denoting the first frame the object is
    detected and one matrix for each tracked object. The matrix
    is tx3 where t is the number of frames each object is tracked for.

    If the provided dimension is 2, it will project velodyne coordinates onto a
    camera view, where the camera is specified by `cam_idx`.
    """
    if dimensions == 2:
        assert calib_dir is not None and exists(calib_dir), \
            'No directory for calibration files found.'
        calib = Calib(calib_dir)
    for tracklet in bundle['tracklets']:
        centers = np.array(
            [np.mean(frame, axis=1) for frame in tracklet['boxes']])
        if dimensions == 2:
            projected_xs, projected_ys, projected_ws, projected_hs = [], [], [], []
            for box in tracklet['boxes']:
                xs, ys = calib.velo2img(box.T, cam_idx).T
                xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
                projected_xs.append(xmin)
                projected_ys.append(ymin)
                projected_ws.append(xmax - xmin)
                projected_hs.append(ymax - ymin)
            tracklet['xs'] = np.vstack(projected_xs)
            tracklet['ys'] = np.vstack(projected_ys)
            tracklet['ws'] = np.vstack(projected_ws)
            tracklet['hs'] = np.vstack(projected_hs)
            tracklet['xs'] += tracklet['ws'] / 2
            tracklet['ys'] += tracklet['hs'] / 2
            centers = np.hstack((tracklet['xs'], tracklet['ys']))
            del tracklet['zs']
        Xt1, Xt2 = centers[:-1], centers[1:]
        tracklet['vectors'] = Xt2 - Xt1
        tracklet['biases'] = Xt1
    bundle['flowlets'] = bundle.pop('tracklets')
    return bundle


def output(
        out: str,
        fbundle: Dict,
        mode: str='frame',
        path_format: str=DEFAULT_PATH_FORMAT,
        columns: str=DEFAULT_3D_COLUMNS
    ) -> None:
    """Saves output per provided mode."""
    if mode == 'object':
        filepath = join(out, 'bundle.mat')
        scipy.io.savemat(filepath, {'bundle': fbundle})
    else:
        os.makedirs(out, exist_ok=True)
        frames = defaultdict(lambda: None)
        for flowlet in fbundle['flowlets']:
            nFrames = flowlet['nFrames']
            firstFrame = flowlet['firstFrame']
            flowlet_data = {
                'h': flowlet['h'],
                'w': flowlet['w'],
                'class': flowlet['class']
            }
            if 'l' in flowlet:
                flowlet_data['l'] = flowlet['l']
            for dt in range(nFrames - 1):
                t = firstFrame + dt
                flowlet_data['x'] = flowlet['xs'][dt]
                flowlet_data['y'] = flowlet['ys'][dt]
                flowlet_data['dx'] = flowlet['vectors'][dt][0]
                flowlet_data['dy'] = flowlet['vectors'][dt][1]
                if flowlet['vectors'].shape[1] >= 3:
                    flowlet_data['z'] = flowlet['zs'][dt]
                    flowlet_data['dz'] = flowlet['vectors'][dt][2]
                entry = np.matrix([flowlet_data[c] for c in columns.split(',')])
                if frames[t] is None:
                    frames[t] = entry
                else:
                    frames[t] = np.vstack((frames[t], entry))

        for frame_id, frame in frames.items():
            path = path_format.format(
                date=fbundle['date'],
                drive_id=fbundle['drive_id'],
                frame_id=str(frame_id).zfill(10))
            np.save(join(out, path), frame)


if __name__ == '__main__':
    main()
