# Heritage mathematica nb from Alex & Laurent
# Python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

import logging
import numpy as np
import scipy.special
from . import leastsqnrm
from . import utils
from . import hextransformee

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def jinc(x, y):
    """
    Compute 2d Jinc for given coordinates.

    Parameters
    ----------
    x, y : float, float
        Input coordinates

    Returns
    -------
    jinc_2d : float array
        2d jinc at the given coordinates, with NaNs replaced by pi/4.
    """
    r = (
        (jinc.d / jinc.lam)
        * jinc.pitch
        * np.sqrt((x - jinc.offx) * (x - jinc.offx) + (y - jinc.offy) * (y - jinc.offy))
    )

    jinc_2d = leastsqnrm.replacenan(scipy.special.jv(1, np.pi * r) / (2.0 * r))

    return jinc_2d


def ffc(kx, ky, **kwargs):
    """
    Calculate cosine terms of analytic model.

    Parameters
    ----------
    kx, ky : float, float
        X-component and y-component of image plane (spatial frequency) vector

    **kwargs : dict
        Keyword arguments
        pitch: float
            sampling pitch in radians in image plane

        baseline: 2D float array
            hole centers

        lam: float
            wavelength

        oversample: integer
            number of samples per detector pixel pitch

        affine2d: Affine2d object

    Returns
    -------
    cos_array : 2D float array
        Cosine terms of analytic model
    """
    ko = kwargs["c"]  # the PSF ctr
    baseline = kwargs["baseline"]  # hole centers' vector
    lam = kwargs["lam"]  # m
    pitch = kwargs["pitch"]  # pitch for calcn = detpixscale/oversample
    affine2d = kwargs["affine2d"]
    kxprime, kyprime = affine2d.distort_f_args(kx - ko[0], ky - ko[1])

    cos_array = 2 * np.cos(
        2 * np.pi * pitch * (kxprime * baseline[0] + kyprime * baseline[1]) / lam
    )

    return cos_array


def ffs(kx, ky, **kwargs):
    """
    Calculate sine terms of analytic model.

    Parameters
    ----------
    kx, ky : float, float
        X-component and y-component of image plane (spatial frequency) vector

    **kwargs : dict
        Keyword arguments
        pitch: float
            sampling pitch in radians in image plane

        baseline: 2D float array
            hole centers

        lam: float
            wavelength

        oversample: integer
            number of samples per detector pixel pitch

        affine2d: Affine2d object

    Returns
    -------
    sin_array : 2D float array
        Sine terms of analytic model
    """
    ko = kwargs["c"]  # the PSF ctr
    baseline = kwargs["baseline"]  # hole centers' vector
    lam = kwargs["lam"]  # m
    pitch = kwargs["pitch"]  # pitch for calcn = detpixscale/oversample
    affine2d = kwargs["affine2d"]
    kxprime, kyprime = affine2d.distort_f_args(kx - ko[0], ky - ko[1])

    sin_array = 2 * np.sin(
        2 * np.pi * pitch * (kxprime * baseline[0] + kyprime * baseline[1]) / lam
    )

    return sin_array


def harmonicfringes(**kwargs):
    """
    Calculate the sine and cosine fringes.

    This is in image space and, for later
    versions, this works in the oversampled space that is each slice of the model.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments
        fov: integer, default=None
            number of detector pixels on a side

        pitch: float
            sampling pitch in radians in image plane

        psf_offset: 2D float array
            offset from image center in detector pixels

        baseline: 2D float array
            hole centers

        lam: float
            wavelength

        oversample: integer
            number of samples per detector pixel pitch

        affine2d = kwargs['affine2d']

    Returns
    -------
    (cosine_fringes, sine_fringes) : tuple
        Sine and cosine fringes: float arrays
    """
    fov = kwargs["fov"]  # in detpix
    pitch = kwargs["pitch"]  # detpixscale
    psf_offset = kwargs["psf_offset"]  # the PSF ctr, detpix
    baseline = kwargs["baseline"]  # hole centers' vector, m
    lam = kwargs["lam"]  # m
    oversample = kwargs["oversample"]
    affine2d = kwargs["affine2d"]

    cpitch = pitch / oversample
    im_ctr = image_center(fov, oversample, psf_offset)

    return (
        np.fromfunction(
            ffc,
            (fov * oversample, fov * oversample),
            c=im_ctr,
            baseline=baseline,
            lam=lam,
            pitch=cpitch,
            affine2d=affine2d,
        ),
        np.fromfunction(
            ffs,
            (fov * oversample, fov * oversample),
            c=im_ctr,
            baseline=baseline,
            lam=lam,
            pitch=cpitch,
            affine2d=affine2d,
        ),
    )


def phasor(kx, ky, hx, hy, lam, phi_m, pitch, affine2d):
    """
    Calculate the wavefront for a single hole.

    This routine returns the complex
    amplitude array of fringes phi to units of meters, which is more physical for
    broadband simulations.

    Parameters
    ----------
    kx, ky : float
        Image plane coords in units of sampling pitch (oversampled, or not)

    hx, hy : float
        Hole centers in meters

    lam : float
        Wavelength

    phi_m : float
        Distance of fringe from hole center in units of meters

    pitch : float
        Sampling pitch in radians in image plane

    affine2d : Affine2d object
        The affine2d object

    Returns
    -------
    phasor : complex
        Calculate wavefront for a single hole
    """
    kxprime, kyprime = affine2d.distort_f_args(kx, ky)
    return np.exp(
        -2 * np.pi * 1j * ((pitch * hx * kxprime + pitch * hy * kyprime) / lam + phi_m / lam)
    ) * affine2d.distortphase(kx, ky)


def image_center(fov, oversample, psf_offset):
    """
    Calculate the Image center location in oversampled pixels.

    Parameters
    ----------
    fov : int
        Number of detector pixels of field of view

    oversample : int
        Number of samples per detector pixel pitch

    psf_offset : 2D int array
        Offset from image center in detector pixels

    Returns
    -------
    offsets_from_center : 2D int array
        Offset of the psf center from the array center.
    """
    offsets_from_center = (
        np.array(utils.centerpoint((oversample * fov, oversample * fov)))
        + np.array((psf_offset[1], psf_offset[0])) * oversample
    )

    return offsets_from_center


def interf(kx, ky, **kwargs):
    """
    Calculate the complex amplitudes for all holes.

    Parameters
    ----------
    kx, ky : float, float radians
        X-component and y-component of image plane (spatial frequency) vector

    **kwargs : dict
        Keyword arguments
        psfctr : 2D float array
            center of PSF, in simulation pixels (i.e. oversampled)

        ctrs : 2D float array
            centers of holes

        phi : float
            distance of fringe from hole center in units of waves

        lam : float
            wavelength

        pitch : float
            sampling pitch in radians in image plane

        affine2d : Affine2d object

    Returns
    -------
    fringe_complexamp : 2D complex array
        Interference for all holes
    """
    psfctr = kwargs["c"]
    ctrs = kwargs["ctrs"]  # hole centers
    phi = kwargs["phi"]
    lam = kwargs["lam"]
    pitch = kwargs["pitch"]  # detpixscale/oversample
    affine2d = kwargs["affine2d"]

    fringe_complexamp = 0j
    for hole, ctr in enumerate(ctrs):
        fringe_complexamp += phasor(
            (kx - psfctr[0]),
            (ky - psfctr[1]),
            ctr[0],
            ctr[1],
            lam,
            phi[hole],
            pitch,
            affine2d,
        )

    return fringe_complexamp


def model_array(
    ctrs,
    lam,
    oversample,
    pitch,
    fov,
    d,
    psf_offset=(0, 0),
    phi=None,
    shape="circ",
    affine2d=None,
):
    """
    Create a model using the specified wavelength.

    Parameters
    ----------
    ctrs : 2D float array
        Centers of holes

    lam : float
        Wavelength in the bandpass for this particular model

    oversample : int
        Oversampling factor

    pitch : float
        Sampling pitch in radians in image plane

    fov : int
        Number of detector pixels on a side.

    d : float
        Hole diameter for 'circ'; flat to flat distance for 'hex

    psf_offset : 2D int array
        Offset from image center in detector pixels

    phi : float
        Distance of fringe from hole center in units of waves

    shape : str
        Shape of hole; possible values are 'circ', 'hex', and 'fringe'

    affine2d : Affine2d object
        The affine2d object

    Returns
    -------
    primary_beam : float 2D array
        Array of primary beam,

    ffmodel : list of fringe arrays
        List of fringe arrays
    """
    nholes = ctrs.shape[0]
    if phi is None:
        np.zeros((nholes,))  # no phase errors in the model slices...
    modelshape = (
        fov * oversample,
        fov * oversample,
    )  # spatial extent of image model - the oversampled array

    # calculate primary beam envelope (non-negative real)
    if shape == "circ":
        asf_pb = asf(pitch, fov, oversample, d, lam, psf_offset, affine2d)
    elif shape == "hex":
        asf_pb = asf_hex(pitch, fov, oversample, d, lam, psf_offset, affine2d)
    else:
        raise KeyError(
            "Must provide a valid hole shape. Current supported shapes are 'circ' and 'hex'."
        )

    primary_beam = (asf_pb * asf_pb.conj()).real

    alist = []
    for i in range(nholes - 1):
        for j in range(nholes - 1):
            if j + i + 1 < nholes:
                alist = np.append(alist, i)
                alist = np.append(alist, j + i + 1)
    alist = alist.reshape((len(alist) // 2, 2))

    ffmodel = []
    ffmodel.append(nholes * np.ones(modelshape))
    for basepair in alist:
        baseline = ctrs[int(basepair[0])] - ctrs[int(basepair[1])]
        cosfringe, sinfringe = harmonicfringes(
            fov=fov,
            pitch=pitch,
            psf_offset=psf_offset,
            baseline=baseline,
            oversample=oversample,
            lam=lam,
            affine2d=affine2d,
        )
        ffmodel.append(cosfringe)
        ffmodel.append(sinfringe)

    return primary_beam, ffmodel


def asf(detpixel, fov, oversample, d, lam, psf_offset, affine2d):
    """
    Calculate the Amplitude Spread Function for a circular aperture.

    Amplitude Spread Function is also know as image plane complex amplitude.

    Parameters
    ----------
    detpixel : float
        Pixel scale

    fov : int
        Number of detector pixels on a side

    oversample : int
        Oversampling factor

    d : float
        Hole diameter

    lam : float
        Wavelength

    psf_offset : 2D float array
        Offset from image center in detector pixels

    affine2d : Affine2d object
        The affine2d object

    Returns
    -------
    asf : 2D real array
        Amplitude Spread Function (a.k.a. image plane complex amplitude) for
        a circular aperture
    """
    pitch = detpixel / float(oversample)
    im_ctr = image_center(fov, oversample, psf_offset)

    return np.fromfunction(
        jinc,
        (oversample * fov, oversample * fov),
        c=im_ctr,
        D=d,
        lam=lam,
        pitch=pitch,
        affine2d=affine2d,
    )


def asffringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d):
    """
    Amplitude Spread Function (a.k.a. image plane complex amplitude) for a fringe.

    Parameters
    ----------
    detpixel : float
        Pixel scale

    fov : int
        Number of detector pixels on a side

    oversample : int
        Oversampling factor

    ctrs : 2D float array
        Centers of holes

    lam : float
        Wavelength

    phi : float
        Distance of fringe from hole center in units of waves

    psf_offset : 2D float array
        Offset from image center in detector pixels

    affine2d : Affine2d object
        The affine2d object

    Returns
    -------
    fringing : 2D complex array
        Amplitude Spread Function (a.k.a. image plane complex amplitude) for
        a fringe
    """
    pitch = detpixel / float(oversample)
    im_ctr = image_center(fov, oversample, psf_offset)

    return np.fromfunction(
        interf,
        (oversample * fov, oversample * fov),
        c=im_ctr,
        ctrs=ctrs,
        phi=phi,
        lam=lam,
        pitch=pitch,
        affine2d=affine2d,
    )


def asf_hex(detpixel, fov, oversample, d, lam, psf_offset, affine2d):
    """
    Amplitude Spread Function (a.k.a. image plane complex amplitude) for a hexagonal aperture.

    Parameters
    ----------
    detpixel : float
        Pixel scale

    fov : int
        Number of detector pixels on a side

    oversample : int
        Oversampling factor

    d : float
        Flat-to-flat distance across hexagon

    lam : float
        Wavelength

    psf_offset : 2D float array
        Offset from image center in detector pixels

    affine2d : Affine2d object
        The affine2d object

    Returns
    -------
    asf : 2D complex array
        Amplitude Spread Function (a.k.a. image plane complex amplitude) for
        a hexagonal aperture
    """
    pitch = detpixel / float(oversample)

    im_ctr = image_center(fov, oversample, psf_offset)

    return hextransformee.hextransform(
        s=(oversample * fov, oversample * fov),
        c=im_ctr,
        d=d,
        lam=lam,
        pitch=pitch,
        affine2d=affine2d,
    )


def psf(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, affine2d, shape="circ"):
    """
    Calculate the PSF for the requested shape.

    Parameters
    ----------
    detpixel : float
        Pixel scale

    fov : int
        Number of detector pixels on a side

    oversample : int
        Oversampling factor

    ctrs : 2D float array
        Centers of holes

    d : float
        Hole diameter for 'circ'; flat-to-flat distance across for 'hex'

    lam : float
        Wavelength

    phi : float
        Distance of fringe from hole center in units of waves

    psf_offset : 2D float array
        Offset from image center in detector pixels

    affine2d : Affine2d object
        The affine2d object

    shape : str
        Shape of hole; possible values are 'circ', 'hex', and 'fringe'

    Returns
    -------
    PSF : 2D float array
        The point-spread function
    """
    # Now deal with primary beam shapes...
    if shape == "circ":
        asf_fringe = asffringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
        asf_2d = asf(detpixel, fov, oversample, d, lam, psf_offset, affine2d) * asf_fringe

    elif shape == "circonly":
        asf_2d = asf(detpixel, fov, oversample, d, lam, psf_offset, affine2d)

    elif shape == "hex":
        asf_fringe = asffringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
        asf_2d = asf_hex(detpixel, fov, oversample, d, lam, psf_offset, affine2d) * asf_fringe

    elif shape == "hexonly":
        asf_2d = asf_hex(detpixel, fov, oversample, d, lam, psf_offset, affine2d)

    elif shape == "fringeonly":
        asf_fringe = asffringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, affine2d)
    else:
        raise ValueError(
            f"pupil shape {shape} not supported - choices: "
            "'circonly', 'circ', 'hexonly', 'hex', 'fringeonly'"
        )

    return (asf_2d * asf_2d.conj()).real
