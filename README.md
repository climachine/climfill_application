# Gap-filled Multivariate Observations of Global Land- climate Interactions
## Application of CLIMFILL: A Framework for Intelligently Gap-filling Earth Observations

This repo contains all scripts necessary to reproduce the results of Bessenbacher, et al, 2023 (in review). The gap-filled dataset can be downloaded here: https://doi.org/10.5281/zenodo.7817826.

CLIMFILL fills gaps in gridded geoscientific observational data by taking into account spatial neighborhood, temporal context and multivariate dependencies. It takes a multivariate dataset with any number and pattern of missing values per variable and returns the dataset with all missing points replaced by estimates. Please see https://github.com/climachine/climfill for more details.

## License
The work is distributed under the Apache-2.0 License.

## References
- [0] Bessenbacher, V., Schumacher, D. L., Hirschi, M., Seneviratne, S. I. and Gudmundsson, L.: Gap-filled Multivariate Observations of Global Land- climate Interactions (in review at JGR Atmospheres)
- [1] Bessenbacher, V., Gudmundsson, L. and Seneviratne, S.I.: CLIMFILL: A Framework for Intelligently Gap-filling Earth Observations. Geoscientific Model Development, 2022. [10.5194/gmd-15-4569 -2022](https://gmd.copernicus.org/articles/15/4569/2022/)
and references therein, especially
- [2] Haylock, M. R, Hofstra, N., Klein Tank, A. M. G., Klok, E. J., Jones, P. D. and New, M. (2008): A European daily high-resolution gridded data set of surface temperature and precipitation for 1950–2006. Journal of Geophysical Research: Atmospheres, 113, D20. DOI:10.1029/2008JD010201
- [3] Das, S., Roy, S. and Sambasivan, R. (2018): Fast Gaussian Process Regression for Big Data. Big Data Research, 14. DOI:10.1016/j.bdr.2018.06.002
- [4] Stekhoven, D. J. and Buehlmann, P. (2012): MissForest -- non-parametric missing value imputation for mixed-type data. Bioinformatics, 28, 1, 112-118. DOI:10.1093/bioinformatics/btr597

