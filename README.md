# PyNLLRunner
Module for running NLLoc package automaticlly.

Summary:
- Using this module you can run Non-Linear Earthquake Location program (NLLOC) more easily and see the statistical and map plots of the final results using matplotlib package.

Requirements:
- NLLoc package (http://alomax.free.fr/nlloc/).
- Numpy.
- Matplotlib.

Usage:
- copy input files into "inp" directory, edit the parameter files in "par" directory and run PyNLLocRunner.py in root directory.

Directories:
- inp: includes velocity model (model.dat), station file (station.dat) and data file (in different formats supported by NLLOC).
- par: includes file for plotting (loc.dat) and parameters for NLLOC (nlloc.dat).
