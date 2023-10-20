# MCPFM_for_Lumen_Fusion

![AppCoM](docs/Figures/test.gif)

<hr/>

[![Doc Status](https://readthedocs.org/projects/appcom/badge/?version=latest)](https://appcom.readthedocs.io/en/latest/)

The `MCPFM_for_Lumen_Fusion` library is an implementation of the multicellular phase-field model to reprizanted lumen fusion events.

## Overview

### The multicellular phase-field model for lumen fusion


### Demo: running simulations and viewing 

```sh

# Conpaile of sorce code
nvcc -O3 -DSFMT_MEXP=19937 src/mcpf_2d_usc.cu src/SFMT.c -o run_simulation -std=c++11

# Load the initial condtions with 8 cells
./run.sh test 200 0.18


# View in ImageJ
plt.show()
```

### Authors

* Kana Fuji - The University of Tokyo

## Dependencies

- nvcc
- imagej

