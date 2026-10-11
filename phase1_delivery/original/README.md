[![DOI](https://zenodo.org/badge/362146259.svg)](https://zenodo.org/badge/latestdoi/362146259)

# emergent-grid-cells

### Create and install Conda env from .yml file:
```console
$ conda env create -f environment.yml
```

### Activate conda environment:
```console
$ conda activate ml
```

### Add anaconda environment to ipykernel:
```console
$ python -m ipykernel install --user --name=ml
```
Source:
https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

### To update environment.yml packages / Save a Conda environment (cross-platform compatible, with python version):
```console
$ conda env export --from-history > environment.yml
```
