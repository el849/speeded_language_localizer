# A 3.5-minute-long reading-based fMRI localizer for the language network

This repository contains the script for running the reading-based fMRI localizer task as well as the accompanying code for analysis. Additional data files may be found in the [OSF repository](https://osf.io/2vskh/).

## Localizer Task ([evlab_langloc_speeded](evlab_langloc_speeded))

This folder contains all necessary files to run the [localizer task](evlab_langloc_speeded/evlab_langloc_speeded.m). This script was tested on MATLAB R2020a and requires [Psychtoolbox](http://psychtoolbox.org/).

[langloc_speeded.webm](https://github.com/el849/speeded_language_localizer/assets/58826739/1c5b7f07-41a9-420f-a343-61440a9be9db)

## Analysis ([analysis](analysis))

To run scripts for analysis and plotting, upload the data folder from the [OSF repository](https://osf.io/2vskh/). The directory structure should look like such:

```bash
speeded_language_localizer
├── analysis
│   ├── data
│   ├── dice_results
│   ├── plots
│   ├── scripts
│   └── stats
├── evlab_langloc_speeded
│   |
.   .
```
