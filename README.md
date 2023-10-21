# MCPFM_for_Lumen_Fusion

![AppCoM](docs/Figures/test.gif)

<hr/>

[![Doc Status](https://readthedocs.org/projects/appcom/badge/?version=latest)](https://appcom.readthedocs.io/en/latest/)

The `MCPFM_for_Lumen_Fusion` library is an implementation of the multicellular phase-field model for lumen fusion events.

## Overview

### The multicellular phase-field model for lumen fusion


## Usage
```sh

# Code compile
nvcc -O3 -DSFMT_MEXP=19937 src/mcpf_2d_usc.cu src/SFMT.c -o run_simulation -std=c++11

# Runnning simulation from 8 cells: 
./run.sh test 200 0.18
# The first argument represents the parameter name, the second one is the cell growth time scale, and the third one is the rumen pressure.

```

## Authors

* Kana Fuji - The University of Tokyo

## Dependencies

- nvcc
- imagej

## License
"MCPFM_for_Lumen_Fusion" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
