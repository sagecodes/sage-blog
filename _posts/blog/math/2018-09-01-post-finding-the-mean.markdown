---
layout: post
title:  "Finding the mean"
permalink: /find-mean-python/
date:   2018-09-01
categories: post
tags: math python
author: Sage Elliott
published: true
---

### How to find the mean

This post will cover how to find the mean (also called average) from a set of numbers. We'll then implement a solution in the python programming language!

For this example we will find the mean for this set of numbers: 

`5,6,7,2,6,9,30,1` 

Step1: Add all the numbers together:

`5+6+7+2+6+9+30+1 = 66`

Step2: Count how many numbers we added together `5,6,7,2,6,9,30,1`. There is a total of 8 numbers. 

Step3: We then divide the sum of the numbers (`66`) by how many numbers there are (`8`).

`66 / 8 = 8.25` 

The mean or average for `5,6,7,2,6,9,30,1` is `8.25`

### Finding the mean with Python:

```
# Take a list of numbers and return the mean
def findMean(numbers):
    mean = sum(numbers) / len(numbers)
    print(mean)

#Run the function and provide list of numbers 
findMean([2,4,5,6,7])
```


