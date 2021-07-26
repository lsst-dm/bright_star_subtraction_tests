"""This file contains standalone objects to be imported in the main notebook
to produce some of the bright star diagnostics. More details on each of these
tests can be found in the notebooks in each subdirectory of this repo."""

import numpy as np
import matplotlib.pyplot as plt
import os

from lsst.afw import image as afwImage
import lsst.afw.display as afwDisplay

class YuanyuanTest(object):
    def __init__(self, apRads,
                 datasetName='src', skyObjColumnName='sky_source', photoCalibDatasetName='calexp',
                 removeFlagged=True, butler=None, butlerPath=None,
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None, gen2=False):
        self.apRads = apRads
        self.datasetName = datasetName
        self.skyObjColumnName = skyObjColumnName
        self.photoCalibDatasetName = photoCalibDatasetName
        self.removeFlagged = removeFlagged
        if butler is not None:
            self.butler = butler
        elif butlerPath is None:
            raise TypeError("Either a butler instance, or a path to a repo where one should be initialized,"
                            "must be passed at initialization.")
        self.fluxColumnStub = fluxColumnStub
        self._fluxes = [[] for _ in range(len(apRads))]
        if dataId is not None:
            self.addFluxes(dataId)
        self.areas = np.array([np.pi * (radius**2) for radius in apRads])
        self.gen2 = gen2
        # Initialize other class variables to None for memory allocation
        self.annFluxe = None
        self.annAreas = None
        self._normalizedFluxes = None
        self.median = None
        self.q10, self.q90 = None, None
        self.q25, self.q75 = None, None
        self.isAnnular = False
        self.annularQ = None
        self.allApsFit = None

    def addFluxes(self, dataId, collection=None):
        print(f'Extracting fluxes for sky objects in {dataId}')
        if self.gen2:
            filename = self.butler.get(self.datasetName+'_filename', dataId=dataId)
            if not os.path.exists(filename[0]):
                print(" > WARNING: you are probably not loading the dataset you think you are loading;"
                      " > Ignoring dataId.")
                return
            if collection is not None:
                print("A collection was provided, but this YuanyuanTest was ran with gen2=True. Ignoring it.")
            sources = self.butler.get(self.datasetName, dataId=dataId)
            photoCalib = self.butler.get(self.photoCalibDatasetName, dataId=dataId).getPhotoCalib()
        else:
            sources = self.butler.get(self.datasetName, dataId=dataId, collections=collection)
            photoCalib = self.butler.get(self.photoCalibDatasetName, dataId=dataId, collections=collection).getPhotoCalib()
        skyObj = sources[sources[self.skyObjColumnName]]
        for j, radius in enumerate(self.apRads):
            instFluxes = np.copy(skyObj[self.fluxColumnStub + f'{radius}_0_instFlux'])
            if self.removeFlagged:
                flag = ~skyObj[self.fluxColumnStub + f'{radius}_0_flag']
                # replace flagged objects with nan
                instFluxes[flag] = np.nan
            # convert to nJy
            self._fluxes[j].extend([photoCalib.instFluxToNanojansky(flux) for flux in instFluxes])

    @property
    def fluxes(self):
        return np.array(self._fluxes)

    @property
    def normalizedFluxes(self):
        if self._normalizedFluxes is None:
            self._normalizeFluxes
        return self._normalizedFluxes

    def _normalizeFluxes(self, annular=False):
        if annular:
            print("Normalizing fluxes within annular apertures...")
            annAreas = self.areas[1:] - self.areas[:-1]
            annFluxes = np.array(self._fluxes)[1:] - np.array(self._fluxes)[:-1]
            self.isAnnular = True
            self._normalizedFluxes = annFluxes/annAreas.reshape(-1, 1)
        else:
            print("Normalizing fluxes within circular apertures...")
            self.isAnnular = False
            self._normalizedFluxes = np.array(self._fluxes)/self.areas.reshape(-1, 1)

    def computeQuantiles(self, medianOnly=False, annular=False):
        if self._normalizedFluxes is None or (annular != self.isAnnular):
            self._normalizeFluxes(annular=annular)
        self.median = np.nanmedian(self._normalizedFluxes, axis=1)
        if not medianOnly:
            self.annularQ = annular
            self.q10 = np.nanquantile(self._normalizedFluxes, 0.1, axis=1)
            self.q90 = np.nanquantile(self._normalizedFluxes, 0.9, axis=1)
            self.q25 = np.nanquantile(self._normalizedFluxes, 0.25, axis=1)
            self.q75 = np.nanquantile(self._normalizedFluxes, 0.75, axis=1)

    def _getRadii(self, annular=False):
        if annular:
            rads = self.apRads[1:] + (self.apRads[1:] - self.apRads[:-1])/2
        else:
            rads = self.apRads
        return rads

    def getSlope(self, annular=False):
        if self.median is None:
            self.computeQuantiles(medianOnly=True, annular=annular)
        self.linearFit = np.polyfit(self._getRadii(annular), self.median, 1)
        return self.linearFit[0]

    def plot(self, annular=False, plotQ10=True, plotQ25=True, plotQ75=True, plotQ90=False,
             adjustPlotSize=True,
             title='', addProcessingStep=True, show=True, savePath='', **kwargs):
        if adjustPlotSize:
            oldSize = plt.rcParams['figure.figsize']
            plt.rcParams['figure.figsize'] = 14, 9
        rads = self._getRadii(annular)
        plotAnyQ = plotQ10 or plotQ25 or plotQ75 or plotQ90
        if self.median is None or annular != self.isAnnular or (plotAnyQ and annular != self.annularQ):
            if plotAnyQ:
                self.computeQuantiles(annular=annular)
            else:
                self.computeQuantiles(medianOnly=True, annular=annular)
        plt.plot(rads, self.median, label='Median', **kwargs)
        label1090 = '10th, 90th percentile'
        label2575 = '25th, 75th percentile'
        if plotQ10:
            plt.plot(rads, self.q10, ls="--", alpha=.6, label=label1090, **kwargs)
            label1090 = ''
        if plotQ25:
            plt.plot(rads, self.q25, ls="-.", alpha=.6, label=label1090, **kwargs)
        if plotQ75:
            plt.plot(rads, self.q75, ls="-.", alpha=.6, label=label2575, **kwargs)
            label2575 = ''
        if plotQ90:
            plt.plot(rads, self.q90, ls="--", alpha=.6, label=label2575, **kwargs)
        plt.axhline(0, c='k', ls=":")
        if addProcessingStep:
            if self.photoCalibDatasetName == 'calexp':
                title = 'Visit-level; ' + title
            elif self.photoCalibDatasetName == 'deepCoadd_calexp':
                title = 'Coadd-level; ' + title
        if annular:
            title = 'Annular apertures; ' + title
            plt.xlabel('Average annulus radius (pix)')
        else:
            title = 'Circular apertures; ' + title
            plt.xlabel('Aperture radius (pix)')
        plt.ylabel(r'Flux per pixel (nJy.pix$^{-1}$)')
        plt.title(title)
        plt.legend()
        if show:
            plt.show()
        if savePath:
            plt.savefig(savePath)
        if adjustPlotSize:
            plt.rcParams['figure.figsize'] = oldSize


class YuanyuanVisitTest(YuanyuanTest):
    def __init__(self, apRads, removeFlagged=True, butler=None, butlerPath=None,
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None, gen2=False):
        YuanyuanTest.__init__(self, apRads, 'src', 'sky_source', 'calexp',
                              removeFlagged, butler, butlerPath, fluxColumnStub, dataId, gen2)


class YuanyuanCoaddTest(YuanyuanTest):
    def __init__(self, apRads, removeFlagged=True, butler=None, butlerPath=None,
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None, gen2=False):
        YuanyuanTest.__init__(self, apRads, 'deepCoadd_meas', 'merge_peak_sky', 'deepCoadd_calexp',
                              removeFlagged, butler, butlerPath, fluxColumnStub, dataId, gen2=gen2)


def plot_func(im, cmap='gist_stern', plotMaskPlane=False, symmetrizeCmap=False,
              wind=None, innerRadius=None, outerRadius=None, title='',
              showTicks=False, plotOneArcsecLine=False, OALy=150, OALcolor='darkorchid'):
    """Utility function to plt.imshow arrays or afw Images."""
    if type(im) in [afwImage.exposure.ExposureF, afwImage.exposure.ExposureD,
                    afwImage.exposure.ExposureI, afwImage.maskedImage.MaskedImageF,
                    afwImage.maskedImage.MaskedImageD, afwImage.maskedImage.MaskedImageI]:
        if plotMaskPlane:
            imarr = im.getMask().array
        else:
            imarr = im.getImage().array
    elif type(im) in [afwImage.image.ImageF, afwImage.image.ImageD,
                      afwImage.image.ImageI]:
        imarr = im.array
    elif type(im) == np.ndarray:
        imarr = im
    else:
        raise ValueError("im should be a numpy array or afw ExposureF")
    if symmetrizeCmap:
        span = max(np.abs(np.nanmin(imarr)), np.abs(np.nanmax(imarr)))
        wind = -span, span
    elif wind is None:
        wind = None, None
    plt.imshow(imarr, cmap=cmap, origin='lower', interpolation='Nearest',
               vmin=wind[0], vmax=wind[1])
    plt.colorbar()
    if not showTicks:
        plt.xticks([])
        plt.yticks([])
    if plotOneArcsecLine:
        dims = imarr.shape
        center = dims[1]/2
        plt.plot([center - 352.9/2, center + 352.9/2], [OALy, OALy],
                 lw=3, c=OALcolor)
        plt.text(center, OALy + 20, "1'", horizontalalignment='center',
                 color=OALcolor, fontsize=21)
    plt.title(title)
    plt.show()
    plt.close()

def plotExposure(exp, bg=None, title='', dispScale=None, mt=100, figsize=(8,8),
                 pixCenters=[], pixText=None):
    """Display plot of an exposure, and eventual points overlay."""
    fig = plt.figure(figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
    display = afwDisplay.getDisplay(backend='matplotlib', frame=fig)
    if dispScale is None:
        display.scale("asinh", np.quantile(exp.image.array, 0.2), np.quantile(exp.image.array, 0.8))
    else:
        display.scale("asinh", dispScale[0], dispScale[1])
    display.setMaskTransparency(mt)
    if bg is None:
        display.mtv(exp)
    else:
        bgexp = exp.clone()
        calexpIm = bgexp.getMaskedImage()
        calexpIm -= bg.getImage()
        display.mtv(bgexp)
    if len(pixCenters) and pixText is None:
        pixText = list(range(len(pixCenters)))
    for j, cen in enumerate(pixCenters):
        plt.plot(cen[0], cen[1], '.', c='orange', ms=4, alpha=.7)
        plt.text(cen[0]+10, cen[1]+10, str(pixText[j]), color='orange')
    plt.title(title)
    plt.show()
    display.close()
    plt.close()

def replaceMaskedPixels(maskedIm, maskPlane, onMask=False, val=0, inPlace=False, verbose=False):
    """ Replace all pixels that were flagged with the mask plane `maskPlane` by
    `val`. If `onMask`, the values are changed on the mask itself; if not, on
    the image.
    """
    mask = maskedIm.mask
    mpd = mask.getMaskPlaneDict()
    mpValues = set(list(mask.array.flatten()))
    bitNb = mpd[maskPlane]
    badVals = []
    if verbose:
        print('MaskPlaneValue\tlen(bin)Binary')
    for mpv in mpValues:
        if verbose:
            if len(str(mpv)) > 2:
                print(f'MPV {mpv}:\t{len(bin(mpv)[2:])}\t{bin(mpv)[2:]}')
            else:
                print(f'MPV {mpv}:\t\t{len(bin(mpv)[2:])}\t{bin(mpv)[2:]}')
        binPv = bin(mpv)[2:]
        if len(binPv) >= bitNb + 1:
            if int(binPv[-(bitNb + 1)]):
                badVals += [mpv]
    if not badVals:
        if verbose:
            print(f'Mask plane {maskPlane} seems absent from image; returning it unchanged')
        return maskedIm
    elif verbose:
        print(f'Mask values to be changed: {badVals}')
    if inPlace:
        newIm = maskedIm
    else:
        newIm = maskedIm.clone()
    if onMask:
        arr = newIm.mask.array
    else:
        arr = newIm.image.array
    maskArr = maskedIm.mask.array
    for bv in badVals:
        arr[maskArr==bv] = val
    return newIm

