# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
