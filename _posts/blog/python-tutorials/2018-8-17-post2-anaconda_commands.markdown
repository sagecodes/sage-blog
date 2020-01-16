---
layout: post
title:  "Anaconda Commands - Quick Start"
permalink: /Anaconda Commands/
date:   2019-12-12
categories: post
tags: python anaconda tools
author: Sage Elliott
published: false
---

Anaconda is **

This is a guide on the features I use most, and sometimes forget! 

Full list and documentation from conda is [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

## Create New Conda Environment

`conda create --name myenv`

`conda create -n myenv python=3.4`

## View List of Conda Environments on your machine:

`conda env list`

## Activate environment 

windows: `activate myenv` 

Mac/linux: `source activate myenv`

## deactivate environment

windows: `deactivate`

mac/linux: `source deactivate`

## install libraries / packages

`conda install packagename`

replacing `packagename` with the package you would like to install

Example for installing numpy:

`conda install numpy`


## Update libraries / packages

conda update

## Creat environment from YAML file


## Anaconda does more

Again this is just a quick start guide for the most common things I use or have people ask please checkout the offcial anaconda site for more information! 

