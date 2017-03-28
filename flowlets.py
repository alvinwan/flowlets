"""Converts trackelets to flowlets.

Flowlets are vectors between frames, from the center of a bounding box at time t
to a bounding box at time t+1. Each flowlet is associated with an object, for
all frames the involved object is detected.

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
"""

from collections import defaultdict
from os.path import join
from typing import Dict
from typing import List

import docopt
import numpy as np
import scipy.io

from thirdparty.calib import Calib
from thirdparty.parseTrackletXML import parseXML

DEFAULT_DRIVE = './'
DEFAULT_KITTI = './'


def main():
    """Run the main utility."""
    arguments = docopt.docopt(__doc__, version='Flowlets 1.0')

    if arguments['<path>']:
        tracklets = extractTracklet(arguments['<path>'])
    elif arguments['drive']:
        raise NotImplementedError('Not yet ready.')
        tracklets = extractDriveTracklet(
            kitti=arguments['--kitti'],
            drive=arguments['--drive'])
    else:
        raise NotImplementedError('Not yet ready.')
        tracklets = extractKITTITracklets(kitti=arguments['--kitti'])

    if arguments['3d']:
        output3d(
            arguments['--out'],
            flowlets=flowletize3d(tracklets),
            mode=arguments['--mode'])
    else:
        output2d(arguments['--out'],
            flowlets=flowletize2d(tracklets, arguments['<calib_dir>']),
            mode=arguments['--mode'])


def extractKITTITracklets(kitti: str=DEFAULT_KITTI) -> List[Dict]:
    """Extract all KITTI tracklets."""
    pass


def extractDriveTracklet(
        kitti: str=DEFAULT_KITTI, drive: str=DEFAULT_DRIVE) -> List[Dict]:
    """Extract all KITTI tracklets for a single drive."""
    pass


def extractTracklet(path: str) -> List[Dict]:
    """Extract tracklets and cartesian coordinates from KITTI tracklet file.

    This function was adapted from the example method in parseTrackletXML.py.
    (See file for credits.)
    """

    twoPi = 2. * np.pi  # read tracklets from file
    tracklets = parseXML(path)
    objects = []

    # loop over tracklets
    for iTracklet, tracklet in enumerate(tracklets):
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
        objects.append({
            'firstFrame': tracklet.firstFrame,
            'nFrames': tracklet.nFrames,
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
            yaw = rotation[2]  # other rotations are 0 in all xml files I checked
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
    return objects


def flowletize3d(tracklets: List[Dict]) -> List[Dict]:
    """Vectorizes the list of cartesian coordinates.

    Returns a list of tuples, one number denoting the first frame the object is
    detected and one matrix for each tracked object. The matrix
    is tx3 where t is the number of frames each object is tracked for.
    """
    for tracklet in tracklets:
        centers = np.array(
            [np.mean(frame, axis=1) for frame in tracklet['boxes']])
        Xt1, Xt2 = centers[:-1], centers[1:]
        tracklet['vectors'] = Xt2 - Xt1
        tracklet['biases'] = Xt1
    return tracklets


def flowletize2d(tracklets: List[Dict], calib_dir: str, cam_idx: int=2)\
        -> List[Dict]:
    """Project vectors onto camera view.

    We assume the camera view runs in the x-z direction, perpendicular to the
    y axis.
    """
    calib = Calib(calib_dir)
    for tracklet in tracklets:
        centers = np.array(
            [np.mean(frame, axis=1) for frame in tracklet['boxes']])
        projected_centers = calib.velo2img(centers, cam_idx)
        Xt1, Xt2 = projected_centers[:-1], projected_centers[1:]
        tracklet['vectors'] = Xt2 - Xt1
        tracklet['biases'] = Xt1
    return tracklets


def output3d(
        out: str,
        flowlets: List[Dict],
        mode: str='frame',
        path_format: str='flowlet-{t}.npy',
    ) -> None:
    """Saves output per provided mode."""
    if mode == 'object':
        filepath = join(out, 'vectors.mat')
        scipy.io.savemat(filepath, {'vectors': flowlets})
    else:
        frames = defaultdict(lambda: None)
        for flowlet in flowlets:
            nFrames = flowlet['nFrames']
            firstFrame = flowlet['firstFrame']
            for dt in range(nFrames - 1):
                t = firstFrame + dt
                entry = np.matrix([
                    flowlet['h'],
                    flowlet['w'],
                    flowlet['l'],
                    flowlet['xs'][dt],
                    flowlet['ys'][dt],
                    flowlet['zs'][dt],
                    flowlet['vectors'][dt][0],
                    flowlet['vectors'][dt][1],
                    flowlet['vectors'][dt][2]
                ])
                if frames[t] is None:
                    frames[t] = entry
                frames[t] = np.vstack((frames[t], entry))

        for t, frame in frames.items():
            path = path_format.format(t=t)
            np.save(join(out, path), frame)


# TODO(Alvin): Smoosh together output3d, output2d for a general function.
def output2d(
        out: str,
        flowlets: List[Dict],
        mode: str='frame',
        path_format: str='flowlet-{t}.npy',
    ) -> None:
    """Saves output per provided mode."""
    if mode == 'object':
        filepath = join(out, 'vectors.mat')
        scipy.io.savemat(filepath, {'vectors': flowlets})
    else:
        frames = defaultdict(lambda: None)
        for flowlet in flowlets:
            nFrames = flowlet['nFrames']
            firstFrame = flowlet['firstFrame']
            for dt in range(nFrames - 1):
                t = firstFrame + dt
                entry = np.array([
                    flowlet['h'],
                    flowlet['w'],
                    flowlet['xs'][dt],
                    flowlet['ys'][dt],
                    flowlet['vectors'][dt][0],
                    flowlet['vectors'][dt][1],
                ])
                if frames[t] is None:
                    frames[t] = entry
                frames[t] = np.vstack((frames[t], entry))

        for t, frame in frames.items():
            path = path_format.format(t=t)
            np.save(join(out, path), frame)

if __name__ == '__main__':
    main()
