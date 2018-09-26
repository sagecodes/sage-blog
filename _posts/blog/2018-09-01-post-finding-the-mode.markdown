---
layout: post
title:  "Finding the mode"
permalink: /find-mode-python/
date:   2018-09-01
categories: post
tags: math python
author: Sage Elliott
published: false
---

This post will cover how to find the mode (also referred to as the most popular number) from a set of numbers. We'll then implement a solution in the python programming language!

For this example we will find the mean for this set of numbers: 

`2,5,6,7,6,2,6,9,30,1` 


Step1: Sort your list of numbers in the order of least to greatest

`1,2,2,5,6,6,6,7,9,30`

*Note*: Sorting isn't a needed step when making a program do this, but having a sorted list to look at makes it easier for us humans to see repeating numbers.

Check to see if any numbers repeat. The ones that repeat the most are the mode.

`1,`**`2,2,`**`5,`**`6,6,6,`**`7,9,30`

We can see that `2`s repeat 2 times and `6` repeats 3 times. So in the example `6` is the Mode.

if all numbers are repeated same amount of times the list is considered multimodal. Any number would be considered a valid mode. 

`6,7,8,9` is a multimodal list. 






