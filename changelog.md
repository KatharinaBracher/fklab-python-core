

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0]


## Added

- Support for windows packaging
- fklab.signal package : add compute_threshold_mad  contrast_frequency_bands, factorization and decimate methods
- Added new package fklab.statistics.distance metrics
- add combine_pvalues method to the fklab.statistics package
- add joint_event_correlation and event_correlation methods to fklab.events package

## Fixed

- segment.contains methods : speed improvement for large inputs

## Restructure

- fklab.plot package :
  + Functions for plotting segments are now part of the fklab.segments.plot module (see docs).
  + There are two functions: plot_segments_as_lines and plot_segments_as_patches. For example use, see the gallery.
  + The fklab.plot.plots.plot_segments function is deprecated.
  + The function for plotting spike rasters has been renamed to plot_event_raster and has a few new features. See the docs and gallery. The old plot_raster and plot_events functions are deprecated.
  + Plot functions are now part of the fklab.plot.plotfunctions module and are available in the top level fklab.plot namespace. So, you just have to do import fklab.plot to get access to the functions. fklab.plot.plots is deprecated.


## [1.8.1]

### Added

- Geometry module (move from internal (private) repo to core (public) repo)
- Cleaning: Add complete tests coverage and clean package Events and Segments

## Deprecation
- fklab.events.event_average deprecated since multiple versions has been removed.
  fklab.signals.event_triggered_average should be used in replacement.

- event.asarray and segment.asarray has been depracted in favor of the numpy api (np.asarray(obj))


## [1.8]

### Added

- Add compute binned statistics feature in fklab.statistics : see [notebook](https://kloostermannerflab.bitbucket.io/notebooks/statistics/binned_statistics.html)
- Add peri_event_density in fklab.events: see [example](https://kloostermannerflab.bitbucket.io/gallery)

## Fixed

- Change of behavior in fklab.segment.asIndex : third output return segment_contains
-
## [1.7]

Project become open-source !!

### Added
- time alignment feature for neuralynx
- speed up apply_filter by replacing filtfilt with convolve to perform filtering
- Extend bootstrapped condifence interval with monte-carlo p-value calculatio

### fixed
- median filter not at the right place in the ripple detection method

## [1.6]
### Added
 - fklab.plot.core.artists: added options to set color and width of scalebar created with AnchoredScaleBar (Fred)
 - fklab.signals.filter: add 'spikes' : [500., 5000.] to standard_frequency_bands. (JJ)
 - fklab.sinals.filter: add median filter option in compute envelope - also available as a standalone filter function (Lies)

### Removed
 - fklab.io : remove the channelmap generation module - transferred in its own repository

## [1.5]
### Added
 - fklab.signals.smooth.kernelsmoothing : add a wrapper method create_smoother
 - fklab.plot.plots.plots: plot_1d_maps and plot_2d_maps methods
 - fklab.codetools.profile : Update ExecutionTimer

### Fixed
 - fklab.statistics.circular.circular : circular difference for single array now properly computed
 - fklab.signals.filter.compute_envelope : fix TypeError ("float' object cannot be interpreted as integer)

## [1.4]
### Added
 - Neuralynx module : generation of data in .dat format + channelmap generation for tetrodes

### Fixed

 - kernelsmoothing : add method option for the scipy.signal.convolve method

## [1.3.3]
### Added
 - Set up file for pre-commit test
 - automatic release pipeline
