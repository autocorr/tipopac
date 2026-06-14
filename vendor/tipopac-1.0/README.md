tipopac
=======

[CASA](http://casa.nrao.edu/) task to derive zenith opacity and Tcals from JVLA tip data.

Latest version: 1.0 ([download here](https://github.com/chrishales/tipopac/releases/latest))

Tested with: CASA Version 5.6.0 REL

tipopac is released under a BSD 3-Clause License (open source, commercially useable); refer to LICENSE for details.

(For the curious, I wrote tipopac to support [ngVLA Memo 63](https://library.nrao.edu/public/memos/ngvla/NGVLA_63.pdf).)

Installation
======

Download the latest version of the source files from [here](https://github.com/chrishales/tipopac/releases/latest).

Place the source files into a directory containing your measurement set. Without changing directories, open CASA and type
```
os.system('buildmytasks')
```
then exit CASA. A number of files should have been produced, including ```mytasks.py```. Reopen CASA and type
```
execfile('mytasks.py')
```
To see the parameter listing, type
```
inp tipopac
```
For extensive details on how tipopac works, type
```
help tipopac
```
Now set some parameters and press go!

For a more permanent installation, place the source files into a dedicated tipopac code directory and perform the steps above. Then go to the hidden directory ```.casa``` which resides in your home directory and create a file called ```init.py```. In this file, put the line
```
execfile('/<path_to_tipopac_directory>/mytasks.py')
```
tipopac will now be available when you open a fresh terminal and start CASA within any directory.

Acknowledging use of tipopac
======

tipopac is provided in the hope that it (or elements of its code) will be useful for your work. If you find that it is, I would appreciate your acknowledgement by citing [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3829385.svg)](https://doi.org/10.5281/zenodo.3829385) as follows (note the [AAS guidelines for citing software](http://journals.aas.org/policy/software.html)):
```
Hales, C. A. 2019, tipopac, v1.0, doi:10.5281/zenodo.3829385
```
