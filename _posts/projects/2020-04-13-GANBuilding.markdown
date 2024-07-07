---
layout: project
title:  "GAN Architecture"
permalink: /gan-bulding-architecture/
date: 2019-12-18
categories: project
tags: computer-vision machine-learning
author: Sage Elliott
published: True
---
This project was in collaboration with the architect [Michael Hasey](http://www.michaelhasey.com/).

## Summary:

The goal of this research project was to explore the ability to generate new building designs based on a specific architecture style using a [Generative Adversarial Network(GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network).

During the traditional method of design it can take hours to create several drawings.
Once trained the GAN could generate several thousand in the same amount of time.    

We started testing with [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), but due to our original small dataset we switched to using a [Wasserstein Generative Adversarial Network (WGAN)](https://arxiv.org/abs/1701.07875) using [gradient penalty](https://arxiv.org/pdf/1704.00028.pdf) which in our case helped avoid mode collpase (the generator figures out how to convince the discriminator and starts producing the same images).

We used images of building designed by [Zaha Hadid](https://en.wikipedia.org/wiki/Zaha_Hadid) to see if the network could capture her design style.

> Example of generated images:
 ![Gan](../../img/gan-building/gan-build1.png)

### Fun Along the Way

### More Results

![Gan](../../img/gan-building/gan-build2.png)

### Building your own image generator 

Interested in getting started with your own image generator?
I've outlined the steps and resources to get you going. 

[Generative Deep Learning Book](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947
 https://github.com/davidADSP/GDL_code)

 [Good article about WGANs](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)

 more images:

  


  