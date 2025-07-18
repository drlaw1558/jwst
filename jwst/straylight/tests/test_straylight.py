"""
Unit tests for straylight correction
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft

from jwst import datamodels
from jwst.straylight.straylight import clean_showers, makemodel_ccode, makemodel_composite


def test_correct_mrs_xartifact():
    """Test Correct Straylight routine gives expected results for small region"""

    image = np.zeros((1024, 1032))
    image[:, 300] = 100
    istart, istop = 0, 516
    xvec = np.arange(1032)
    lorfwhm = np.zeros(1024) + 100
    lorscale = np.zeros(1024) + 0.001
    gauxoff = np.zeros(1024) + 10
    gaufwhm = np.zeros(1024) + 7
    gauscale1 = np.zeros(1024) + 0.001
    gauscale2 = np.zeros(1024) + 0.001

    # Test C version
    result_c = makemodel_ccode(
        image, xvec, istart, istop, lorfwhm, lorscale, gaufwhm, gauxoff, gauscale1, gauscale2
    )
    cutout_c = result_c[500, 270:330]

    # Test python version
    result_py = makemodel_composite(
        image, xvec, istart, istop, lorfwhm, lorscale, gaufwhm, gauxoff, gauscale1, gauscale2
    )
    cutout_py = result_py[500, 270:330]

    compare = np.array(
        [
            0.09783203,
            0.10662437,
            0.11656731,
            0.12742336,
            0.13880997,
            0.15021261,
            0.1610211,
            0.17058835,
            0.17830785,
            0.18370678,
            0.18655573,
            0.18699956,
            0.18570036,
            0.18393369,
            0.18349825,
            0.18625805,
            0.19326517,
            0.20376261,
            0.21473958,
            0.2216788,
            0.22045676,
            0.2094179,
            0.1902925,
            0.16733257,
            0.14527816,
            0.12747414,
            0.11511109,
            0.10763159,
            0.10367233,
            0.10188942,
            0.10139532,
            0.10188942,
            0.10367233,
            0.10763159,
            0.11511109,
            0.12747414,
            0.14527816,
            0.16733257,
            0.1902925,
            0.2094179,
            0.22045676,
            0.2216788,
            0.21473958,
            0.20376261,
            0.19326517,
            0.18625805,
            0.18349825,
            0.18393369,
            0.18570036,
            0.18699956,
            0.18655573,
            0.18370678,
            0.17830785,
            0.17058835,
            0.1610211,
            0.15021261,
            0.13880997,
            0.12742336,
            0.11656731,
            0.10662437,
        ]
    )

    assert np.allclose(compare, cutout_c, rtol=1e-6)
    assert np.allclose(compare, cutout_py, rtol=1e-6)


def test_clean_showers():
    """Test cosmic ray shower cleaning routine gives expected results for mock data"""

    # Make a mock input image with a single blob on the detector
    input_model = datamodels.IFUImageModel()
    point = np.zeros((1024, 1032))
    point[614, 604] = 1
    gauss = Gaussian2DKernel(x_stddev=18, y_stddev=18)
    input_model.data = convolve_fft(point, gauss)
    input_model.dq = np.zeros((1024, 1032), dtype=">i4")

    # Set parameters like real step, but without rejection since we have no noise
    shower_plane = 0
    shower_low_reject = 0
    shower_high_reject = 100
    shower_x_stddev = 18
    shower_y_stddev = 5

    # Set up a mock regions file with just one slice partially covering the blob
    mockregions = np.zeros((1, 1024, 1032))
    mockregions[0, :, 583:609] = 1

    # Run the clean_showers routine and see if the values it reconstructed for the blob
    # look as expected for a small box of pixels underneath the slice mask
    result, shower_model = clean_showers(
        input_model,
        mockregions,
        shower_plane,
        shower_x_stddev,
        shower_y_stddev,
        shower_low_reject,
        shower_high_reject,
    )

    cutout = result.data[613:617, 591:596]
    compare = np.array(
        [
            [0.00018815, 0.00019507, 0.0002014, 0.00020712, 0.00021221],
            [0.00018846, 0.00019539, 0.00020173, 0.00020746, 0.00021256],
            [0.00018815, 0.00019507, 0.0002014, 0.00020712, 0.00021221],
            [0.00018722, 0.0001941, 0.0002004, 0.00020609, 0.00021116],
        ]
    )

    assert np.allclose(cutout, compare, rtol=1e-6)
