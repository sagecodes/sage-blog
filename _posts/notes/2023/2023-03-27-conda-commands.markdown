---
layout: post
title:  "Conda commands for managing environments"
permalink: /conda-commands/
date:   2023-03-27
categories: note
tags: conda environment
author: Sage Elliott
published: true
---

** 
This post is a "note". I use notes to keep track of useful things for myself, but if you stumbled upon this page and find it useful, great!

Check out my blog posts for more in-depth content.
**

# Common Conda commands for managing environments

[Conda](https://docs.conda.io/en/latest/) is a package manager commonly used for python. It is used to create environments for different projects and to install packages in those environments. It is similar to virtualenv.

## Conda Creating & Using Environment Commands: Create, Activate, Deactivate, List
- `conda create -n envName python=3.8` create environment with python version
- `conda activate envName` activate environment (or `source activate envName` in some operating systems)
- `conda deactivate` deactivate environment
- `conda env list` list environments

## Conda Package Commands: Install, Uninstall, List
- `conda install packageName` install package in current environment
- `conda install -c conda-forge packageName` install package in current environment from conda-forge channel
- `conda list` list packages in current environment
- `conda uninstall packageName` uninstall package in current environment
- `conda 

## Exporting & Importing Environments
- `conda install -c conda-forge --file requirements.txt` install packages in current environment from conda-forge channel from requirements.txt file
- `conda env create -f environment.yml` create environment from yml file
- `conda env export > environment.yml` export environment to yml file
- `pip freeze > requirements.txt` export packages to requirements.txt file for pip installs


## Other Conda Commands: Export, Create, Remove
- `conda env remove -n envName` remove environment
- `conda install -n envName packageName` install package in environment
- `conda install -n envName -c conda-forge packageName` install package in environment from conda-forge channel
- `conda env export > environment.yml` export environment to yml file



