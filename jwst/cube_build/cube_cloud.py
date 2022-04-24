""" Map the detector pixels to the cube coordinate system.
This is where the weight functions are used.
"""
import numpy as np
import logging
from shapely.geometry import Polygon
import pdb
from Geometry3D import *
import time

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def match_det2cube_driz(naxis1, naxis2, naxis3,
                       cdelt1, cdelt2,
                       zcdelt3,
                       xcenters, ycenters, zcoord,
                       spaxel_flux,
                       spaxel_weight,
                       spaxel_iflux,
                       spaxel_var,
                       flux,
                       err,
                       coord1, coord2, ccoord, wave, dwave,
                       weighting_type,
                       rois_pixel, roiw_pixel, weight_pixel,
                       softrad_pixel, scalerad_pixel,
                       cube_debug, debug_file):
    """ Map the detector pixels to the cube spaxels using the drizzle parameters


    Match the Point Cloud members to the spaxel centers that fall in the drizzle
    droplet.

    Note that this routine does NOT build the cube by looping over spaxels and
    looking for pixels that contribute to those spaxels.  The runtime is significantly
    better to instead loop over pixels, and look for spaxels that they contribute to.
    This way we can just keep a running sum of the weighted fluxes in a 1-d representation
    of the output cube, which can be normalized by the weights at the end.

    Parameters
    ----------
    naxis1 : int
       size of the ifucube in 1st axis
    naxis2 : int
       size of the ifucube in 2nd axis
    naxis3 : int
       size of the ifucube in 3rd axis
    cdelt1 : float
       ifucube spaxel size in axis 1 dimension
    cdelt2 : float
       ifucube spaxel size in axis 2 dimension
    cdelt3_normal :float
       ifu spectral size at wavelength
    rois_pixel : float
       region of influence size in spatial dimension
    roiw_pixel : float
       region of influence size in spectral dimension
    weight_power : float
       msm weighting parameter
    xcenter : numpy.ndarray
       spaxel center locations 1st dimensions.
    ycenter : numpy.ndarray
       spaxel center locations 2nd dimensions.
    zcoord : numpy.ndarray
        spaxel center locations in 3rd dimensions
    spaxel_flux : numpy.ndarray
       contains the weighted summed detector fluxes that fall
       within the roi
    spaxel_weight : numpy.ndarray
       contains the summed weights assocated with the detector fluxes
    spaxel_iflux : numpy.ndarray
       number of detector pixels falling with roi of spaxel center
    spaxel_var: numpy.ndarray
       contains the weighted summed variance within the roi
    flux : numpy.ndarray
       array of detector fluxes associated with each position in
       coorr1, coord2, wave
    err: numpy.ndarray
       array of detector errors associated with each position in
       coorr1, coord2, wave
    coord1 : numpy.ndarray
       contains the spatial coordinate for 1st dimension for the mapped
       detector pixel
    coord2 : numpy.ndarray
       contains the spatial coordinate for 2nd dimension for the mapped
       detector pixel
    wave : numpy.ndarray
       contains the spectral coordinate  for the mapped detector pixel

    Returns
    -------
    spaxel_flux, spaxel_weight, spaxel_ifux, and spaxel_var updated with the information
    from the detector pixels that fall within the roi of the spaxel center.
    """
    nplane = naxis1 * naxis2

    cdelt3=zcdelt3[0]# Note this will fail if zcdelt3 is constant throughout cube!!
    
    # Corner coordinates
    xcc1, ycc1, zcc1 = ccoord[0], ccoord[1], ccoord[2]
    xcc2, ycc2, zcc2 = ccoord[3], ccoord[4], ccoord[5]
    xcc3, ycc3, zcc3 = ccoord[6], ccoord[7], ccoord[8]
    xcc4, ycc4, zcc4 = ccoord[9], ccoord[10], ccoord[11]
    xcc5, ycc5, zcc5 = ccoord[12], ccoord[13], ccoord[14]
    xcc6, ycc6, zcc6 = ccoord[15], ccoord[16], ccoord[17]
    xcc7, ycc7, zcc7 = ccoord[18], ccoord[19], ccoord[20]
    xcc8, ycc8, zcc8 = ccoord[21], ccoord[22], ccoord[23]

    # xcenters and ycenters are a 1d representation of a maxtrix of all x,y locations
    # in a given wavelength plane
    # Need to generalize to 3d
    x3dcenters=np.zeros(naxis3 * naxis2 * naxis1)
    y3dcenters=np.zeros(naxis3 * naxis2 * naxis1)
    z3dcenters=np.zeros(naxis3 * naxis2 * naxis1)
    for ii in range(0,naxis3):
        x3dcenters[ii*nplane:(ii+1)*nplane]=xcenters
        y3dcenters[ii*nplane:(ii+1)*nplane]=ycenters
        z3dcenters[ii*nplane:(ii+1)*nplane]=zcoord[ii]

    xleft=x3dcenters-cdelt1/2.
    xright=x3dcenters+cdelt1/2.
    ybot=y3dcenters-cdelt2/2.
    ytop=y3dcenters+cdelt2/2.
    zbot=z3dcenters-cdelt3/2.# Note this will fail if zcdelt3 is not constant throughout cube!!
    ztop=z3dcenters+cdelt3/2.
    
    #pdb.set_trace()

    # Test code by running only a tiny wavelength range
    # Test 5.20-5.22 microns
    trim=True
    if trim:
        indx=np.where((wave >= 5.20)&(wave < 5.22))
        flux=flux[indx]
        err=err[indx]
        coord1=coord1[indx]
        coord2=coord2[indx]
        wave=wave[indx]
        xcc1,ycc1,zcc1=xcc1[indx],ycc1[indx],zcc1[indx]
        xcc2,ycc2,zcc2=xcc2[indx],ycc2[indx],zcc2[indx]
        xcc3,ycc3,zcc3=xcc3[indx],ycc3[indx],zcc3[indx]
        xcc4,ycc4,zcc4=xcc4[indx],ycc4[indx],zcc4[indx]
        xcc5,ycc5,zcc5=xcc5[indx],ycc5[indx],zcc5[indx]
        xcc6,ycc6,zcc6=xcc6[indx],ycc6[indx],zcc6[indx]
        xcc7,ycc7,zcc7=xcc7[indx],ycc7[indx],zcc7[indx]
        xcc8,ycc8,zcc8=xcc8[indx],ycc8[indx],zcc8[indx]
    


    
    # now loop over the pixel values for this region and find the spaxels that fall
    # within the region of interest.

    nn = coord1.size
    #    print('looping over n points mapping to cloud',nn)
    # ________________________________________________________________________________

    time0=time.perf_counter()
    
    for ipt in range(0, nn - 1):
        pctdone=(100*ipt/nn)
        time1 = time.perf_counter()
        dtime=int(time1-time0)
        if (dtime % 30 == 0): # Progress every 30 seconds
            print('Elapsed: ',dtime,' seconds.  Percent done: ',pctdone,'%')

            
        # Compute extreme possible boundaries of the polyhedron volume
        polyxmin=np.min([xcc1[ipt],xcc2[ipt],xcc3[ipt],xcc4[ipt],xcc5[ipt],xcc6[ipt],xcc7[ipt],xcc8[ipt]])
        polyxmax=np.max([xcc1[ipt],xcc2[ipt],xcc3[ipt],xcc4[ipt],xcc5[ipt],xcc6[ipt],xcc7[ipt],xcc8[ipt]])
        polyymin=np.min([ycc1[ipt],ycc2[ipt],ycc3[ipt],ycc4[ipt],ycc5[ipt],ycc6[ipt],ycc7[ipt],ycc8[ipt]])
        polyymax=np.max([ycc1[ipt],ycc2[ipt],ycc3[ipt],ycc4[ipt],ycc5[ipt],ycc6[ipt],ycc7[ipt],ycc8[ipt]])
        polyzmin=np.min([zcc1[ipt],zcc2[ipt],zcc3[ipt],zcc4[ipt],zcc5[ipt],zcc6[ipt],zcc7[ipt],zcc8[ipt]])
        polyzmax=np.max([zcc1[ipt],zcc2[ipt],zcc3[ipt],zcc4[ipt],zcc5[ipt],zcc6[ipt],zcc7[ipt],zcc8[ipt]])
        

        # Compute 3d index for possible overlaps
        index3d = np.where((xleft < polyxmax) \
                          & (xright > polyxmin) \
                          & (ybot < polyymax) \
                           & (ytop > polyymin) \
                           & (zbot < polyzmax) \
                           & (ztop > polyzmin))[0]

        #if (flux[ipt] > 1):
        #    pdb.set_trace()
            
        Ax,Ay,Az=xcc6[ipt],ycc6[ipt],zcc6[ipt]
        Bx,By,Bz=xcc5[ipt],ycc5[ipt],zcc5[ipt]
        Cx,Cy,Cz=xcc8[ipt],ycc8[ipt],zcc8[ipt]
        Dx,Dy,Dz=xcc7[ipt],ycc7[ipt],zcc7[ipt]
        
        Ex,Ey,Ez=xcc2[ipt],ycc2[ipt],zcc2[ipt]
        Fx,Fy,Fz=xcc1[ipt],ycc1[ipt],zcc1[ipt]
        Gx,Gy,Gz=xcc4[ipt],ycc4[ipt],zcc4[ipt]
        Hx,Hy,Hz=xcc3[ipt],ycc3[ipt],zcc3[ipt]

        allx=[Ax,Bx,Cx,Dx,Ex,Fx,Gx,Hx]
        ally=[Ay,By,Cy,Dy,Ey,Fy,Gy,Hy]
        allz=[Az,Bz,Cz,Dz,Ez,Fz,Gz,Hz]

        # Because of floating point issues, can't just make convex
        # polygons directly- one point will often be slightly out of plane at part in 1e9
        # Therefore define planes from sets of 3 coordinates to refine 4th coordinate
        
        # Use the plane defined by ABC to refine the point for D
        # to ensure it is planar (using eqn of a plane through 3 points)
        aa=(By-Ay)*(Cz-Az) - (Cy-Ay)*(Bz-Az)
        bb=(Bz-Az)*(Cx-Ax) - (Cz-Az)*(Bx-Ax)
        cc=(Bx-Ax)*(Cy-Ay) - (Cx-Ax)*(By-Ay)
        dd=-(aa*Ax + bb*Ay + cc*Az)
        Dz =-(dd + aa*Dx + bb*Dy)/cc

        # A,B,C,D,E are now good to use for defining points
        Apt=Point(Ax,Ay,Az)
        Bpt=Point(Bx,By,Bz)
        Cpt=Point(Cx,Cy,Cz)
        Dpt=Point(Dx,Dy,Dz)
        Ept=Point(Ex,Ey,Ez)

        # Use plane defined by ADE to refine point H
        aa=(Dy-Ay)*(Ez-Az) - (Ey-Ay)*(Dz-Az)
        bb=(Dz-Az)*(Ex-Ax) - (Ez-Az)*(Dx-Ax)
        cc=(Dx-Ax)*(Ey-Ay) - (Ex-Ax)*(Dy-Ay)
        dd=-(aa*Ax + bb*Ay + cc*Az)
        Hx =-(dd + cc*Hz + bb*Hy)/aa
        Hpt=Point(Hx,Hy,Hz)

        # Use plane defined by BAE to refine point F
        aa=(Ay-By)*(Ez-Bz) - (Ey-By)*(Az-Bz)
        bb=(Az-Bz)*(Ex-Bx) - (Ez-Bz)*(Ax-Bx)
        cc=(Ax-Bx)*(Ey-By) - (Ex-Bx)*(Ay-By)
        dd=-(aa*Bx + bb*By + cc*Bz)
        Fy =-(dd + aa*Fx + cc*Fz)/bb
        Fpt=Point(Fx,Fy,Fz)
        
        # Refine point G as the intersection of planes CBF, CDH, EFH
        p1=Plane(Fpt,Ept,Hpt)
        p2=Plane(Hpt,Dpt,Cpt)
        p3=Plane(Cpt,Bpt,Fpt)
        temp=intersection(p1,p2)
        Gpt=intersection(p3,temp)

        cpg0 = ConvexPolygon((Apt,Dpt,Hpt,Ept))
        cpg1 = ConvexPolygon((Apt,Ept,Fpt,Bpt))
        cpg2 = ConvexPolygon((Cpt,Bpt,Fpt,Gpt))
        cpg3 = ConvexPolygon((Cpt,Gpt,Hpt,Dpt))
        cpg4 = ConvexPolygon((Apt,Bpt,Cpt,Dpt))
        cpg5 = ConvexPolygon((Ept,Hpt,Gpt,Fpt))

        cph_pix = ConvexPolyhedron((cpg0,cpg1,cpg2,cpg3,cpg4,cpg5))
        
        # Find the cube spectral planes that this input point will contribute to
        nmatch = len(index3d)
        #print('nmatch = ',nmatch)
        for ii in range(0,nmatch):
            # Spaxel coordinates
            x1_spax=x3dcenters[index3d[ii]]-cdelt1/2.
            x2_spax=x3dcenters[index3d[ii]]+cdelt1/2.
            y1_spax=y3dcenters[index3d[ii]]-cdelt2/2.
            y2_spax=y3dcenters[index3d[ii]]+cdelt2/2.
            z1_spax=z3dcenters[index3d[ii]]-cdelt3/2.
            z2_spax=z3dcenters[index3d[ii]]+cdelt3/2.
            a=Point(x2_spax,y2_spax,z2_spax)
            b=Point(x1_spax,y2_spax,z2_spax)
            c=Point(x1_spax,y1_spax,z2_spax)
            d=Point(x2_spax,y1_spax,z2_spax)
            e=Point(x2_spax,y2_spax,z1_spax)
            f=Point(x1_spax,y2_spax,z1_spax)
            g=Point(x1_spax,y1_spax,z1_spax)
            h=Point(x2_spax,y1_spax,z1_spax)
            cpg0 = ConvexPolygon((a,d,h,e))
            cpg1 = ConvexPolygon((a,e,f,b))
            cpg2 = ConvexPolygon((c,b,f,g))
            cpg3 = ConvexPolygon((c,g,h,d))
            cpg4 = ConvexPolygon((a,b,c,d))
            cpg5 = ConvexPolygon((e,h,g,f))
            cph_spax = ConvexPolyhedron((cpg0,cpg1,cpg2,cpg3,cpg4,cpg5))

            # Run overlap computation in a try-except loop, since intersection code
            # crashes when non-zero, or if overlap is a single point
            try:
                overlap=intersection(cph_spax,cph_pix)
                volume=overlap.volume()
                #print('volume= ',volume)

                # Confirmed that most-angled pixels actually do LOOK angled in these plots
                #r = Renderer()
                #r.add((cph_pix,'g',1))
                #r.add((cph_spax,'b',1))
                #r.add((overlap,'r',1))
                #r.show()
                #blah=input('enter: ')
                #pdb.set_trace()
                
            except:
                volume=0.
                #print('No overlap!')
                #pdb.set_trace()

            weight=volume
            weighted_flux = weight * flux[ipt]
            weighted_var = (weight * err[ipt]) * (weight * err[ipt])

            spaxel_flux[index3d[ii]] += weighted_flux
            spaxel_weight[index3d[ii]] += weight
            spaxel_iflux[index3d[ii]] += 1
            spaxel_var[index3d[ii]] += weighted_var



# _______________________________________________________________________

def match_det2cube_msm(naxis1, naxis2, naxis3,
                       cdelt1, cdelt2,
                       zcdelt3,
                       xcenters, ycenters, zcoord,
                       spaxel_flux,
                       spaxel_weight,
                       spaxel_iflux,
                       spaxel_var,
                       flux,
                       err,
                       coord1, coord2, wave,
                       weighting_type,
                       rois_pixel, roiw_pixel, weight_pixel,
                       softrad_pixel, scalerad_pixel,
                       cube_debug, debug_file):
    """ Map the detector pixels to the cube spaxels using the MSM parameters


    Match the Point Cloud members to the spaxel centers that fall in the ROI.
    For each spaxel the coord1,coord1 and wave point cloud members are weighed
    according to modified shepard method of inverse weighting based on the
    distance between the point cloud member and the spaxel center.

    Note that this routine does NOT build the cube by looping over spaxels and
    looking for pixels that contribute to those spaxels.  The runtime is significantly
    better to instead loop over pixels, and look for spaxels that they contribute to.
    This way we can just keep a running sum of the weighted fluxes in a 1-d representation
    of the output cube, which can be normalized by the weights at the end.

    Parameters
    ----------
    naxis1 : int
       size of the ifucube in 1st axis
    naxis2 : int
       size of the ifucube in 2nd axis
    naxis3 : int
       size of the ifucube in 3rd axis
    cdelt1 : float
       ifucube spaxel size in axis 1 dimension
    cdelt2 : float
       ifucube spaxel size in axis 2 dimension
    cdelt3_normal :float
       ifu spectral size at wavelength
    rois_pixel : float
       region of influence size in spatial dimension
    roiw_pixel : float
       region of influence size in spectral dimension
    weight_power : float
       msm weighting parameter
    xcenter : numpy.ndarray
       spaxel center locations 1st dimensions.
    ycenter : numpy.ndarray
       spaxel center locations 2nd dimensions.
    zcoord : numpy.ndarray
        spaxel center locations in 3rd dimensions
    spaxel_flux : numpy.ndarray
       contains the weighted summed detector fluxes that fall
       within the roi
    spaxel_weight : numpy.ndarray
       contains the summed weights assocated with the detector fluxes
    spaxel_iflux : numpy.ndarray
       number of detector pixels falling with roi of spaxel center
    spaxel_var: numpy.ndarray
       contains the weighted summed variance within the roi
    flux : numpy.ndarray
       array of detector fluxes associated with each position in
       coorr1, coord2, wave
    err: numpy.ndarray
       array of detector errors associated with each position in
       coorr1, coord2, wave
    coord1 : numpy.ndarray
       contains the spatial coordinate for 1st dimension for the mapped
       detector pixel
    coord2 : numpy.ndarray
       contains the spatial coordinate for 2nd dimension for the mapped
       detector pixel
    wave : numpy.ndarray
       contains the spectral coordinate  for the mapped detector pixel

    Returns
    -------
    spaxel_flux, spaxel_weight, spaxel_ifux, and spaxel_var updated with the information
    from the detector pixels that fall within the roi of the spaxel center.
    """
    nplane = naxis1 * naxis2

    # now loop over the pixel values for this region and find the spaxels that fall
    # within the region of interest.
    nn = coord1.size
    #    print('looping over n points mapping to cloud',nn)
    # ________________________________________________________________________________
    for ipt in range(0, nn - 1):
        # xcenters, ycenters is a flattened 1-D array of the 2 X 2 xy plane
        # cube coordinates.
        # find the spaxels that fall withing ROI of point cloud defined  by
        # coord1,coord2,wave
        lower_limit = softrad_pixel[ipt]
        xdistance = (xcenters - coord1[ipt])
        ydistance = (ycenters - coord2[ipt])
        radius = np.sqrt(xdistance * xdistance + ydistance * ydistance)

        indexr = np.where(radius <= rois_pixel[ipt])
        indexz = np.where(abs(zcoord - wave[ipt]) <= roiw_pixel[ipt])

        # Find the cube spectral planes that this input point will contribute to
        if len(indexz[0]) > 0:
            d1 = np.array(coord1[ipt] - xcenters[indexr]) / cdelt1
            d2 = np.array(coord2[ipt] - ycenters[indexr]) / cdelt2
            d3 = np.array(wave[ipt] - zcoord[indexz]) / zcdelt3[indexz]

            dxy = (d1 * d1) + (d2 * d2)

            # shape of dxy is #indexr or number of overlaps in spatial plane
            # shape of d3 is #indexz or number of overlaps in spectral plane
            # shape of dxy_matrix & d3_matrix  (#indexr, #indexz)
            # rows = number of overlaps in spatial plane
            # cols = number of overlaps in spectral plane
            dxy_matrix = np.tile(dxy[np.newaxis].T, [1, d3.shape[0]])
            d3_matrix = np.tile(d3 * d3, [dxy_matrix.shape[0], 1])

            # wdistance is now the spatial distance squared plus the spectral distance squared
            wdistance = dxy_matrix + d3_matrix
            if weighting_type == 'msm':
                weight_distance = np.power(np.sqrt(wdistance), weight_pixel[ipt])
                weight_distance[weight_distance < lower_limit] = lower_limit
                weight_distance = 1.0 / weight_distance
            elif weighting_type == 'emsm':
                weight_distance = np.exp(-wdistance / (scalerad_pixel[ipt] / cdelt1))

            # DRL hack
            temp1=dxy_matrix.flatten('F')
            temp2=d3_matrix.flatten('F')
                
            weight_distance = weight_distance.flatten('F')
            weighted_flux = weight_distance * flux[ipt]
            weighted_var = (weight_distance * err[ipt]) * (weight_distance * err[ipt])

            # Identify all of the cube spaxels (ordered in a 1d vector) that this input point contributes to
            icube_index = [iz * nplane + ir for iz in indexz[0] for ir in indexr[0]]

            if cube_debug in icube_index:
                #pdb.set_trace()
                #log.info('cube_debug %i %d %d', ipt, flux[ipt], weight_distance[icube_index.index(cube_debug)])
                print('cube_debug %i %d %d %d %d', ipt, flux[ipt], temp1[icube_index.index(cube_debug)], temp2[icube_index.index(cube_debug)], weight_distance[icube_index.index(cube_debug)])

            # Add the weighted flux and variance to running 1d cubes, along with the weights
            # (for later normalization), and point count (for information)
            spaxel_flux[icube_index] = spaxel_flux[icube_index] + weighted_flux
            spaxel_weight[icube_index] = spaxel_weight[icube_index] + weight_distance
            spaxel_iflux[icube_index] = spaxel_iflux[icube_index] + 1
            spaxel_var[icube_index] = spaxel_var[icube_index] + weighted_var

# _______________________________________________________________________


def match_det2cube_miripsf(alpha_resol, beta_resol, wave_resol,
                           naxis1, naxis2, naxis3,
                           xcenters, ycenters, zcoord,
                           spaxel_flux,
                           spaxel_weight,
                           spaxel_iflux,
                           spaxel_var,
                           spaxel_alpha, spaxel_beta, spaxel_wave,
                           flux,
                           err,
                           coord1, coord2, wave, alpha_det, beta_det,
                           weighting_type,
                           rois_pixel, roiw_pixel,
                           weight_pixel,
                           softrad_pixel,
                           scalerad_pixel):
    """ Map the detector pixels to the cube spaxels using miri PSF weighting

    Map coordinates coord1,coord2, and wave of the point cloud to which
    spaxels they overlap with in the ifucube. For each spaxel the coord1,coord2
    and wave point cloud members are weighting according to the miri psf and lsf.
    The weighting function is based on the distance the point cloud member
    and spaxel center in the alph-beta coordinate system. The alpha and beta
    value of each point cloud member  and spaxel center is passed to this
    routine.

    Parameters
    ----------
    alpha_resol : numpy.ndarray
      alpha resolution table
    beta_resol : numpy.ndarray
      beta resolution table
    wave_resol : numpy.ndarray
      wavelength resolution table
    naxis1 : int
       size of the ifucube in 1st axis
    naxis2 : int
       size of the ifucube in 2nd axis
    naxis3 : int
       size of the ifucube in 3rd axis
    xcenter : numpy.ndarray
       spaxel center locations 1st dimensions.
    ycenter : numpy.ndarray
       spaxel center locations 2nd dimensions.
    zcoord : numpy.ndarray
        spaxel center locations in 3rd dimensions
    spaxel_flux : numpy.ndarray
       contains the weighted summed detector fluxes that fall withi the roi
    spaxel_weight : numpy.ndarray
       contains the summed weights assocated with the detector fluxes
    spaxel_iflux : numpy.ndarray
       number of detector pixels falling with roi of spaxel center
    spaxel_var: numpy.ndarray
       contains the weighted summed variance within the roi
    flux : numpy.ndarray
       array of detector fluxes associated with each position in
       coorr1, coord2, wave
    err: numpy.ndarray
       array of detector errors associated with each position in
       coorr1, coord2, wave
    spaxel_alpha : numpy.ndarray
       alpha value of spaxel centers
    spaxel_beta : numpy.ndarray
       beta value of spaxel centers
    spaxel_wave : numpy.ndarray
       alpha value of spaxel centers
    coord1 : numpy.ndarray
       contains the spatial coordinate for 1st dimension for the mapped
       detector pixel
    coord2 : numpy.ndarray
       contains the spatial coordinate for 2nd dimension for the mapped
       detector pixel
    wave : numpy.ndarray
       contains the spectral coordinate  for the mapped detector pixel
    alpha_det : numpy.ndarray
       alpha coordinate of mapped detector pixels
    beta_det :  numpy.ndarray
       beta coordinate of mapped detector pixels
    rois_pixel : float
       region of influence size in spatial dimension
    roiw_pixel : float
       region of influence size in spectral dimension
    weight_power : float
       msm weighting parameter
    softrad_pxiel :float
       weighting paramter

    Returns
    -------
    spaxel_flux, spaxel_weight, spaxel_ifux, spaxel_var updated with the information
    from the detector pixels that fall within the roi if the spaxel center.

    """

    nplane = naxis1 * naxis2
    # now loop over the pixel values for this region and find the spaxels
    # that fall within the region of interest.
    nn = coord1.size
    #    print('looping over n points mapping to cloud',nn)
    # _______________________________________________________________________
    for ipt in range(0, nn - 1):
        lower_limit = softrad_pixel[ipt]

        # ________________________________________________________
        # if weight is miripsf -distances determined in alpha-beta
        # coordinate system

        weights = FindNormalizationWeights(wave[ipt],
                                           wave_resol,
                                           alpha_resol,
                                           beta_resol)
        # ___________________________________________________________________
        # xcenters, ycenters is a flattened 1-D array of the 2 X 2 xy plane
        # cube coordinates.
        # find the spaxels that fall withing ROI of point cloud defined by
        # coord1,coord2,wave
        xdistance = (xcenters - coord1[ipt])
        ydistance = (ycenters - coord2[ipt])
        radius = np.sqrt(xdistance * xdistance + ydistance * ydistance)

        indexr = np.where(radius <= rois_pixel[ipt])
        indexz = np.where(abs(zcoord - wave[ipt]) <= roiw_pixel[ipt])

        # _______________________________________________________________
        # TODO if this method is used replace two for loops with quicker
        # list comprehension
        # loop over the points in the ROI
        for iz, zz in enumerate(indexz[0]):
            istart = zz * nplane
            for ir, rr in enumerate(indexr[0]):
                cube_index = istart + rr

                alpha_distance = alpha_det[ipt] - spaxel_alpha[cube_index]
                beta_distance = beta_det[ipt] - spaxel_beta[cube_index]
                wave_distance = abs(wave[ipt] - spaxel_wave[cube_index])

                xn = alpha_distance / weights[0]
                yn = beta_distance / weights[1]
                wn = wave_distance / weights[2]

                # only included the spatial dimensions
                wdistance = (xn * xn + yn * yn + wn * wn)
# ________________________________________________________________________________
                # MSM weighting based on 1/r**power
                if weighting_type == 'msm':
                    weight_distance = np.power(np.sqrt(wdistance), weight_pixel[ipt])
                    if weight_distance < lower_limit:
                        weight_distance = lower_limit
                        weight_distance = 1.0 / weight_distance
                elif weighting_type == 'emsm':
                    weight_distance = scalerad_pixel[ipt] * np.exp(1.0 / wdistance)

                weighted_flux = weight_distance * flux[ipt]
                weighted_var = (weight_distance * err[ipt]) * (weight_distance * err[ipt])

                spaxel_flux[cube_index] = spaxel_flux[cube_index] + weighted_flux
                spaxel_weight[cube_index] = spaxel_weight[cube_index] + weight_distance
                spaxel_iflux[cube_index] = spaxel_iflux[cube_index] + 1
                spaxel_var[cube_index] = spaxel_var[cube_index] + weighted_var
# _______________________________________________________________________


def FindNormalizationWeights(wavelength,
                             wave_resol,
                             alpha_resol,
                             beta_resol):
    """ Routine used in MIRI PSF weighting to normalize data


    Normalize weighting to each point cloud that contributes to the
    ROI of a spaxel. The normalization of weighting is determined from
    width of PSF as wellas wavelength resolution

    Parameters
    ----------
    wave_resol : numpy.ndarray
      wavelength resolution array
    alpha_resol : numpy.ndarray
      alpha psf resolution array
    beta_resol : numpy.ndarray
      beta psf resolution array

    Returns
    -------
    normalized weighting for 3 dimension
    """
    alpha_weight = 1.0
    beta_weight = 1.0
    lambda_weight = 1.0

    # alpha psf weighting
    alpha_wave_cutoff = alpha_resol[0]
    alpha_a_short = alpha_resol[1]
    alpha_b_short = alpha_resol[2]
    alpha_a_long = alpha_resol[3]
    alpha_b_long = alpha_resol[4]
    if wavelength < alpha_wave_cutoff:
        alpha_weight = alpha_a_short + alpha_b_short * wavelength
    else:
        alpha_weight = alpha_a_long + alpha_b_long * wavelength

    # beta psf weighting
    beta_wave_cutoff = beta_resol[0]
    beta_a_short = beta_resol[1]
    beta_b_short = beta_resol[2]
    beta_a_long = beta_resol[3]
    beta_b_long = beta_resol[4]
    if wavelength < beta_wave_cutoff:
        beta_weight = beta_a_short + beta_b_short * wavelength
    else:
        beta_weight = beta_a_long + beta_b_long * wavelength

    # wavelength weighting
    wavecenter = wave_resol[0]
    a_ave = wave_resol[1]
    b_ave = wave_resol[2]
    c_ave = wave_resol[3]

    wave_diff = wavelength - wavecenter
    resolution = a_ave + b_ave * wave_diff + c_ave * wave_diff * wave_diff
    lambda_weight = wavelength / resolution
    weight = [alpha_weight, beta_weight, lambda_weight]
    return weight
