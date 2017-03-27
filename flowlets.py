"""Converts trackelets to vectors."""

from os.path import join
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import scipy.io

from thirdparty.parseTrackletXML import TRUNC_IN_IMAGE
from thirdparty.parseTrackletXML import TRUNC_TRUNCATED
from thirdparty.parseTrackletXML import parseXML

DEFAULT_DRIVE = './'


def extract(drive: str=DEFAULT_DRIVE) -> List[Dict]:
    """Extract tracklets and cartesian coordinates from KITTI tracklet file.

    This function was adapted from the example method in parseTrackletXML.py.
    (See file for credits.)
    """

    twoPi = 2. * np.pi  # read tracklets from file
    myTrackletFile = join(drive, 'data/tracklet_labels.xml')
    tracklets = parseXML(myTrackletFile)
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
        boxes = []
        translations = []
        yawVisuals = []
        objects.append({
            'firstFrame': tracklet.firstFrame,
            'size': (h, w, l),
            'translations': translations,
            'yawVisuals': yawVisuals,
            'boxes': boxes
        })

        # loop over all frames in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, \
            amtBorders, absoluteFrameNumber in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                continue

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
            translations.append((x, y, z))
            yawVisuals.append(yawVisual)

            boxes.append(cornerPosInVelo)
    return objects


def vectorize(tracklets: List[Dict]) -> List[Dict]:
    """Vectorizes the list of cartesian coordinates.

    Returns a list of tuples, one number denoting the first frame the object is
    detected and one matrix for each tracked object. The matrix
    is tx3 where t is the number of frames each object is tracked for.
    """
    vectors = []
    for tracklet in tracklets:
        centers = np.array(
            [np.mean(frame, axis=1) for frame in tracklet['boxes']])
        Xt1, Xt2 = centers[:-1], centers[1:]
        vectors.append({
            'firstFrame': tracklet['firstFrame'],
            'size': tracklet['size'],
            'translations': tracklet['translations'],
            'yawVisuals': tracklet['yawVisuals'],
            'vectors': Xt2 - Xt1,
            'biases': Xt1
        })
    return vectors


def project(vectors: List[Dict], focal_length: float,
        normal=np.matrix([0, 1, 0]).T) -> List[Tuple[int, np.array]]:
    """Project vectors onto camera view.

    We assume the camera view runs in the x-z direction, perpendicular to the
    y axis.
    """
    assert normal.shape == (3, 1), 'Normal must be a column vector in R^3'
    normal = normal / np.linalg.norm(normal)
    bias = focal_length * normal
    projected_vectors = []
    for vector_data in vectors:
        _vectors, _biases = vector_data['vectors'], vector_data['biases']
        _vectors = _vectors - (_vectors.dot(normal) * normal.T)
        _biases = bias.T.dot(normal)/ _vectors.dot(normal) * bias.T
        projected_vectors.append({
            'firstFrame': vector_data['firstFrame'],
            'size': vector_data['size'],
            'translations': vector_data['translations'],
            'yawVisuals': vector_data['yawVisuals'],
            'vectors': _vectors,
            'biases': _biases
        })
    return projected_vectors


if __name__ == '__main__':
    tracklets = extract()
    vectors = vectorize(tracklets)
    scipy.io.savemat('out/vectors.mat', {'vectors': vectors})
    projected_vectors = project(vectors, 50)
    scipy.io.savemat('out/projected_vectors.mat', {'vectors': projected_vectors})
