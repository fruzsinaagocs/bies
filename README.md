# Aperiodic scattering from a periodic, corrugated surface in 2D

## Summary

This repository contains code for solving the 2D Helmholtz equation in the presence of a point source and a periodic "staircase" geometry subject to Neumann bondary conditions. It is not a standalone package (but may become one in the future).

## Methods

This code implements the method published in [this paper](https://doi.org/10.1016/j.jcp.2024.113383).

## The Well dataset - how to run

To generate simulations used in The Well dataset, please run

```bash
pip install matplotlib pandas
clone https://github.com/fruzsinaagocs/bies
cd bies/src
mkdir data
python generate-the-well.py
```

This will generate a large number of files in the `data/` directory. They are in the following format:
```
thewell-omega-<1>-x0-<2>-<3>.txt
```
where `<1>` denotes the value for the incident frequency, `<2>` is the source position, and `<3>` tells you what quantity is being stored. The value of `<3>` is to be interpreted as follows: "kappa-<...>" are intermediate results, "params" contains parameters for the PDE solver for the run, "u" contains the unknown field, "kappas" and "weights" are quadrature nodes and weights associated with the PDE solver, respectively.
Two more files, named `thewell-coords-x1.txt` and `thewell-coords-x2.txt` are generated that store the x and y coordinates associated with the unknown field values (and are needed for plotting).

The `plot()` function in `generate-the-well.py` may be used to visualize the output.

## Documentation

You can generate offline API-level documentation with `sphinx`. First, you'll need to install it:

```bash
pip install sphinx sphinx-rtd-theme numpydoc
```

then run

```
cd docs
make html
```

and finally open `docs/_build/index.html`.
