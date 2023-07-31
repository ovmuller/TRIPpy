[![DOI](https://zenodo.org/badge/663470111.svg)](https://zenodo.org/badge/latestdoi/663470111)


# TRIPpy

## Brief description

This repository serves as the public source code repository of the TRIPpy model. TRIPpy stands for Total Runoff Integrating Pathways in python. TRIPpy is a standalone implementation of the TRIP model, originally developed by [Oki and Sud (1999)](https://doi.org/10.1175/1087-3562(1998)002<0001:DOTRIP>2.3.CO;2) in Fortran. This implementation is developed and maintained by [Omar V. Müller](https://www.researchgate.net/profile/Omar-Mueller). For help and/or contributions contact him at [ovmuller@gmail.com](mailto:ovmuller@gmail.com).

TRIPpy is a river routing model that collects the runoff from each grid cell in a given domain and drives it through a prescribed river network, estimating the river storage and outflow of each grid cell. TRIPpy uses a simple advection method to route total runoff through the topography. Full description of the equations is provided in the appendix of [Müller et al. 2021](https://doi.org/10.1175/JHM-D-20-0290.1). 

The main feature of the model is its simplicity, allowing long-term simulations in short term. It requires just one forcing variable (total runoff) and few parameters (flow direction, flow sequence, flow velocity, and meandering ratio). Another important feature of the model is that TRIPpy does not consider any loss or gain of water, it just translates runoff values into river discharge. This feature allows (1) a comparison of simulated river discharge against observations, and thereby, a validation of runoff at catchment scale, and (2) offline simulations of rivers, conserving the water balance at the catchment scale when the forcing is generated by a GCM, a RCM, a LSM, or a reanalysis. 

## Download

This is the very first version of the model called TRIPpy v1.0. Download it from [releases](https://github.com/ovmuller/TRIPpy/releases).

## Documentation

The documentation of TRIPpy is provided in the [wiki](https://github.com/ovmuller/TRIPpy/wiki).

## Citation
Müller, Omar V., 2023. TRIPpy v1.0. Zenodo. DOI: [https://doi.org/10.5281/zenodo.8199913](https://doi.org/10.5281/zenodo.8199913).

## Publications

TRIPpy has been succesfully used in :
- Müller, O. V., P. L. Vidale, B. Vannière, R. Schiemann, and P. C. McGuire, 2021: Does the HadGEM3-GC3.1 GCM Overestimate Land Precipitation at High Resolution? A Constraint Based on Observed River Discharge. J. Hydrometeor., 22, 2131–2151, [https://doi.org/10.1175/JHM-D-20-0290.1](https://doi.org/10.1175/JHM-D-20-0290.1).

- Müller, O. V., P. C. McGuire, P. L. Vidale, E. Hawkins: River flow in the near future: a global perspective in the context of a high-emission climate change scenario. Under revision in HESS, [https://doi.org/10.5194/egusphere-2023-1281](https://doi.org/10.5194/egusphere-2023-1281).

## Acknowledgements

I would like to thank the following colleagues, whose comments and feedback have contributed to the better development of TRIPpy.

- Leandro Sgroi
- Patrick McGuire
- Pier Luigi Vidale
- Benoît Vannière
- Reinhard Schiemann
