# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
