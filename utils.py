"""This file contains standalone objects to be imported in the main notebook
to produce some of the bright star diagnostics. More details on each of these
tests can be found in the notebooks in each subdirectory of this repo."""

import numpy as np
import matplotlib.pyplot as plt
import os


class YuanyuanTest(object):
    def __init__(self, apRads,
                 datasetName='src', skyObjColumnName='sky_source', photoCalibDatasetName='calexp',
                 removeFlagged=True, butler=None, butlerPath=None,
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None):
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

    def addFluxes(self, dataId):
        print(f'Extracting fluxes for sky objects in {dataId}')
        filename = self.butler.get(self.datasetName+'_filename', dataId=dataId)
        if not os.path.exists(filename[0]):
            print(" > WARNING: you are probably not loading the dataset you think you are loading;"
                  " > Ignoring dataId.")
            return
        sources = self.butler.get(self.datasetName, dataId=dataId)
        skyObj = sources[sources[self.skyObjColumnName]]
        photoCalib = self.butler.get(self.photoCalibDatasetName, dataId=dataId).getPhotoCalib()
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
            self.isAnnular=True
            self._normalizedFluxes = annFluxes/annAreas.reshape(-1,1)
        else:
            print("Normalizing fluxes within circular apertures...")
            self.isAnnular=False
            self._normalizedFluxes = np.array(self._fluxes)/self.areas.reshape(-1,1)

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

    def plot(self, annular=False, plotQ10=True, plotQ25=True, plotQ75=True, plotQ90=False, adjustPlotSize=True,
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
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None):
        YuanyuanTest.__init__(self, apRads, 'src', 'sky_source', 'calexp',
                              removeFlagged, butler, butlerPath, fluxColumnStub, dataId)


class YuanyuanCoaddTest(YuanyuanTest):
    def __init__(self, apRads, removeFlagged=True, butler=None, butlerPath=None,
                 fluxColumnStub='base_CircularApertureFlux_', dataId=None):
        YuanyuanTest.__init__(self, apRads, 'deepCoadd_meas', 'merge_peak_sky', 'deepCoadd_calexp',
                              removeFlagged, butler, butlerPath, fluxColumnStub, dataId)
