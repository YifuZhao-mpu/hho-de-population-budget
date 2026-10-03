# Data licence

The per-run results, analysis outputs and pre-registration files in `results/`
are licensed under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

You may share and adapt them for any purpose, provided you give appropriate
credit (cite the paper and this repository), provide a link to the licence, and
indicate if changes were made.

The code in this repository is licensed separately under the MIT licence
(`LICENSE`).

## Third-party material (not covered by either licence)

`cec_native/` contains the official C/C++ sources and input data of the CEC2014,
CEC2017 and CEC2022 benchmark suites, distributed by P. N. Suganthan and
colleagues (https://github.com/P-N-Suganthan). They are included unmodified
except for the documented `%Lf` -> `%lf` fix (see `cec_native/patch_cec14.py`
and the README) and remain under their authors' terms. The published jSO,
L-SHADE and SHADE result files used by the validation scripts are not
redistributed; the README gives their official sources and checksums.
