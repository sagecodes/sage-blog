---
layout: project
title:  "Titanic Survival"
permalink: /titanic-survival-logistic/
date: 2019-08-11
categories: project
tags: machine-learning case-study
author: Sage Elliott
published: true
github_url: https://github.com/sagecodes/titanic-survival-logistic-regression
---

# Titanic Survival

 

## About:

This project / case study is for phase 1 of my [100 days of machine learning code](https://sageelliott.com/100daysofmlcode/) challenge.

This is a homework solution to a section in [Machine Learning Classification Bootcamp in Python](https://www.udemy.com/machine-learning-classification). 

#### Problem Statement:

Predict Passenger Survival based on  feature measurments of the titanic dataset.



## Technology used:

#### Model(s): 
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)

#### Dataset(s):

- The famous [Titanic dataset](https://www.kaggle.com/c/titanic)

#### Libraries:

- [Scikit Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
<!--- [numpy](https://www.numpy.org/)-->
- [seaborn](https://seaborn.pydata.org/)

#### Resources:

- [Scikit Learn knn classification](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)

#### Contact:

If for any reason you would like to contact me please do so at the following:

- [sageelliott.com](https://sageelliott.com/)
- [hello@sageelliott.com](hello@sageelliott.com)
- [twitter](https://twitter.com/sagecodes)
- [linkedin](https://www.linkedin.com/in/sageelliott)




<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Logistic-regression">Logistic regression<a class="anchor-link" href="#Logistic-regression">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Import-Libraries">Import Libraries<a class="anchor-link" href="#Import-Libraries">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># import Libraries</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Import-Dataset">Import Dataset<a class="anchor-link" href="#Import-Dataset">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#import Data into Pandas DataFrame</span>
<span class="n">training_set</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/titanic/Train_Titanic.csv&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Verify Data imported</span>
<span class="n">training_set</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
<span class="c1"># training_set.tail(10)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Explore-Dataset">Explore Dataset<a class="anchor-link" href="#Explore-Dataset">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survived</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">[</span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">]</span>
<span class="n">no_survived</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">[</span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">0</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Total Passengers = &#39;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">training_set</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Number of Passengers who survived = &#39;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">survived</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Number of Passengers who died = &#39;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">no_survived</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;% Survived = &#39;</span><span class="p">,</span> <span class="mi">1</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">survived</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">training_set</span><span class="p">)</span> <span class="o">*</span> <span class="mi">100</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;% Died = &#39;</span><span class="p">,</span> <span class="mi">1</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">no_survived</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">training_set</span><span class="p">)</span> <span class="o">*</span> <span class="mi">100</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Total Passengers =  891
Number of Passengers who survived =  342
Number of Passengers who died =  549
% Survived =  38.38383838383838
% Died =  61.61616161616161
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger class numbers</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Pclass&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a0d99c3c8&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
rkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger survival by class numbers</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Pclass&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a0d7f4438&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
kqRW/x/zC6WRyZEB0AAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger siblings</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;SibSp&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a15eb57f0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
TnE72wAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger survival with siblings</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;SibSp&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a15fac400&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
zMws0/8HYoe4vUvyPH0AAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passengers with Parent / child</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Parch&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a1607d2b0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
TkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger survival with Parent / child</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Parch&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a161b2710&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
SUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passengers embarked</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Embarked&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[12]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a162c5b70&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYgAAAEKCAYAAAAIO8L1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAEk5JREFUeJzt3Xu0nXV95/H3ByKgtRiQAzJJnFBNL7RVpKdMLJ22iu0S
egm1YnW1Q8qwJjNrUacdOxemuqq92GVnxlovLZ1MqQZrVUpLyTgsW1aUdrRFe6gMCliTMkrOCiUH
RbwVO+B3/ti/U7bJj2QH82Tv5Lxfa+31PL/f83v2/iZ7hQ/P77nsVBWSJO3ruGkXIEmaTQaEJKnL
gJAkdRkQkqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV2rpl3A1+K0006r9evXT7sMSTqq3HrrrfdX
1dzBxh3VAbF+/XoWFhamXYYkHVWSfGqScU4xSZK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKk
LgNCktRlQEiSuo7qO6kPxXf8h2umXcKKcOt/vXTaJUg6TDyCkCR1GRCSpC4DQpLUZUBIkroMCElS
lwEhSeoyICRJXQaEJKnLgJAkdQ0aEElWJ7kuyceT3JXkuUlOTXJTkp1teUobmyRvSrIrye1Jzh2y
NknSgQ19BPFG4L1V9c3As4G7gCuBHVW1AdjR2gAXAhvaawtw1cC1SZIOYLCASHIy8D3A1QBV9Q9V
9VlgE7CtDdsGXNzWNwHX1MgtwOokZw5VnyTpwIY8gvgGYAl4a5KPJPmdJF8HnFFV9wK05elt/Bpg
99j+i61PkjQFQwbEKuBc4Kqqeg7wRR6dTupJp6/2G5RsSbKQZGFpaenwVCpJ2s+QAbEILFbVh1r7
OkaBcd/y1FFb7h0bv25s/7XAnn3ftKq2VtV8Vc3Pzc0NVrwkrXSDBURV/R2wO8k3ta4LgDuB7cDm
1rcZuKGtbwcubVczbQQeXJ6KkiQdeUP/YNDLgXckOQG4G7iMUShdm+Ry4B7gkjb2RuAiYBfwpTZW
kjQlgwZEVd0GzHc2XdAZW8AVQ9YjSZqcd1JLkroMCElSlwEhSeoyICRJXQaEJKnLgJAkdRkQkqQu
A0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKkLgNCktRlQEiSugwISVKXASFJ6jIg
JEldBoQkqcuAkCR1GRCSpK5BAyLJJ5N8NMltSRZa36lJbkqysy1Paf1J8qYku5LcnuTcIWuTJB3Y
kTiCeF5VnVNV8619JbCjqjYAO1ob4EJgQ3ttAa46ArVJkh7DNKaYNgHb2vo24OKx/mtq5BZgdZIz
p1CfJInhA6KAP01ya5Itre+MqroXoC1Pb/1rgN1j+y62PknSFKwa+P3Pr6o9SU4Hbkry8QOMTaev
9hs0CpotAE9/+tMPT5WSpP0MegRRVXvaci9wPXAecN/y1FFb7m3DF4F1Y7uvBfZ03nNrVc1X1fzc
3NyQ5UvSijZYQCT5uiRfv7wO/ADwMWA7sLkN2wzc0Na3A5e2q5k2Ag8uT0VJko68IaeYzgCuT7L8
Ob9fVe9N8lfAtUkuB+4BLmnjbwQuAnYBXwIuG7A2SdJBDBYQVXU38OxO/6eBCzr9BVwxVD2SpEPj
ndSSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaEJKnLgJAkdRkQkqQuA0KS1GVASJK6DAhJUpcB
IUnqMiAkSV0GhCSpy4CQJHUZEJKkLgNCktRlQEiSugwISVKXASFJ6jIgJEldBoQkqcuAkCR1DR4Q
SY5P8pEk72nts5J8KMnOJO9OckLrP7G1d7Xt64euTZL02I7EEcTPAHeNtX8NeENVbQAeAC5v/ZcD
D1TVM4E3tHGSpCkZNCCSrAV+EPid1g7wfOC6NmQbcHFb39TatO0XtPGSpCkY+gjiN4D/CHyltZ8K
fLaqHm7tRWBNW18D7AZo2x9s479Kki1JFpIsLC0tDVm7JK1ogwVEkh8C9lbVrePdnaE1wbZHO6q2
VtV8Vc3Pzc0dhkolST2rBnzv84EfSXIRcBJwMqMjitVJVrWjhLXAnjZ+EVgHLCZZBTwF+MyA9UmS
DmCwI4iq+s9Vtbaq1gMvBd5XVT8BvB94cRu2GbihrW9vbdr291XVfkcQkqQjYxr3Qfwn4BVJdjE6
x3B1678aeGrrfwVw5RRqkyQ1Q04x/aOquhm4ua3fDZzXGfMQcMmRqEeSdHDeSS1J6jIgJEldEwVE
kh2T9EmSjh0HPAeR5CTgScBpSU7h0XsVTgb+ycC1SZKm6GAnqf818LOMwuBWHg2IzwG/OWBdkqQp
O2BAVNUbgTcmeXlVvfkI1SRJmgETXeZaVW9O8l3A+vF9quqageqSJE3ZRAGR5O3AM4DbgEdadwEG
hCQdoya9UW4eONtHX0jSyjHpfRAfA542ZCGSpNky6RHEacCdST4MfHm5s6p+ZJCqJElTN2lAvGbI
IiRJs2fSq5j+bOhCJEmzZdKrmD7Po7/udgLwBOCLVXXyUIVJkqZr0iOIrx9vJ7mYziO7JUnHjsf1
NNeq+mPg+Ye5FknSDJl0iulFY83jGN0X4T0RknQMm/Qqph8eW38Y+CSw6bBXI0maGZOeg7hs6EIk
SbNl0h8MWpvk+iR7k9yX5A+TrB26OEnS9Ex6kvqtwHZGvwuxBvifrU+SdIyaNCDmquqtVfVwe70N
mBuwLknSlE0aEPcn+ckkx7fXTwKfHrIwSdJ0TRoQ/xJ4CfB3wL3Ai4EDnrhOclKSDyf5P0nuSPKL
rf+sJB9KsjPJu5Oc0PpPbO1dbfv6x/uHkiR97SYNiF8GNlfVXFWdzigwXnOQfb4MPL+qng2cA7ww
yUbg14A3VNUG4AHg8jb+cuCBqnom8IY2TpI0JZMGxLOq6oHlRlV9BnjOgXaokS+05hPaqxjdgX1d
698GXNzWN7U2bfsFSTJhfZKkw2zSgDguySnLjSSnMsE9FO18xW3AXuAm4G+Bz1bVw23IIqOromjL
3QBt+4PAUzvvuSXJQpKFpaWlCcuXJB2qSe+kfj3wF0muY3QU8BLgtQfbqaoeAc5Jshq4HviW3rC2
7B0t7Pc4j6raCmwFmJ+f93EfkjSQSe+kvibJAqPpoQAvqqo7J/2QqvpskpuBjcDqJKvaUcJaYE8b
tgisAxaTrAKeAnxm4j+JJOmwmvhprlV1Z1W9parePEk4JJlrRw4keSLwAuAu4P2MroIC2Azc0Na3
tzZt+/uqyiMESZqSSaeYHo8zgW1JjmcURNdW1XuS3Am8K8mvAB8Brm7jrwbenmQXoyOHlw5YmyTp
IAYLiKq6nc6VTlV1N50fG6qqh4BLhqpHknRoHtcPBkmSjn0GhCSpy4CQJHUZEJKkLgNCktRlQEiS
ugwISVKXASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaEJKnL
gJAkdRkQkqQuA0KS1GVASJK6DAhJUtdgAZFkXZL3J7kryR1Jfqb1n5rkpiQ72/KU1p8kb0qyK8nt
Sc4dqjZJ0sENeQTxMPBzVfUtwEbgiiRnA1cCO6pqA7CjtQEuBDa01xbgqgFrkyQdxGABUVX3VtVf
t/XPA3cBa4BNwLY2bBtwcVvfBFxTI7cAq5OcOVR9kqQDOyLnIJKsB54DfAg4o6ruhVGIAKe3YWuA
3WO7Lba+fd9rS5KFJAtLS0tDli1JK9rgAZHkycAfAj9bVZ870NBOX+3XUbW1quaran5ubu5wlSlJ
2segAZHkCYzC4R1V9Uet+77lqaO23Nv6F4F1Y7uvBfYMWZ8k6bENeRVTgKuBu6rq18c2bQc2t/XN
wA1j/Ze2q5k2Ag8uT0VJko68VQO+9/nAvwA+muS21vfzwOuAa5NcDtwDXNK23QhcBOwCvgRcNmBt
kqSDGCwgquoD9M8rAFzQGV/AFUPVI0k6NN5JLUnqMiAkSV0GhCSpy4CQJHUZEJKkriEvc5UOm3t+
6dunXcIx7+m/8NFpl6AZ4xGEJKnLgJAkdRkQkqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSp
y4CQJHUZEJKkLgNCktRlQEiSugwISVKXASFJ6jIgJEldBoQkqWuwgEjyu0n2JvnYWN+pSW5KsrMt
T2n9SfKmJLuS3J7k3KHqkiRNZsgjiLcBL9yn70pgR1VtAHa0NsCFwIb22gJcNWBdkqQJDBYQVfXn
wGf26d4EbGvr24CLx/qvqZFbgNVJzhyqNknSwR3pcxBnVNW9AG15eutfA+weG7fY+iRJUzIrJ6nT
6avuwGRLkoUkC0tLSwOXJUkr15EOiPuWp47acm/rXwTWjY1bC+zpvUFVba2q+aqan5ubG7RYSVrJ
jnRAbAc2t/XNwA1j/Ze2q5k2Ag8uT0VJkqZj1VBvnOSdwPcBpyVZBF4NvA64NsnlwD3AJW34jcBF
wC7gS8BlQ9UlSZrMYAFRVS97jE0XdMYWcMVQtUiSDt2snKSWJM0YA0KS1DXYFJMkAZz/5vOnXcKK
8MGXf/Cwv6dHEJKkLgNCktRlQEiSugwISVKXASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBI
kroMCElSlwEhSeoyICRJXQaEJKnLgJAkdRkQkqQuA0KS1GVASJK6DAhJUtdMBUSSFyb5myS7klw5
7XokaSWbmYBIcjzwm8CFwNnAy5KcPd2qJGnlmpmAAM4DdlXV3VX1D8C7gE1TrkmSVqxZCog1wO6x
9mLrkyRNwappFzAmnb7ab1CyBdjSml9I8jeDVjVdpwH3T7uIQ5H/tnnaJcyKo+6749W9f4Ir1lH3
/eXfHtL3908nGTRLAbEIrBtrrwX27DuoqrYCW49UUdOUZKGq5qddhw6d393Rze9vZJammP4K2JDk
rCQnAC8Ftk+5JklasWbmCKKqHk7y08CfAMcDv1tVd0y5LElasWYmIACq6kbgxmnXMUNWxFTaMcrv
7ujm9wekar/zwJIkzdQ5CEnSDDEgZlCSVya5I8ntSW5L8s+mXZMml+RpSd6V5G+T3JnkxiTfOO26
dHBJ1ia5IcnOJHcneUuSE6dd17QYEDMmyXOBHwLOrapnAS/gq28g1AxLEuB64OaqekZVnQ38PHDG
dCvTwbTv7o+AP66qDcAG4InAf5lqYVM0UyepBcCZwP1V9WWAqjqqbtYRzwP+X1X99nJHVd02xXo0
uecDD1XVWwGq6pEk/w74VJJXVtUXplvekecRxOz5U2Bdkk8k+a0k3zvtgnRIvg24ddpF6HH5Vvb5
7qrqc8AngWdOo6BpMyBmTPu/lO9g9DiRJeDdSX5qqkVJK0PoPN6H/mOAVgQDYgZV1SNVdXNVvRr4
aeDHpl2TJnYHo4DX0ecO4Kser5HkZEbnj47lZ749JgNixiT5piQbxrrOAT41rXp0yN4HnJjkXy13
JPlOpwqPCjuAJyW5FP7xN2peD7ylqv5+qpVNiQExe54MbGuXR97O6MeTXjPdkjSpGt15+qPA97fL
XO9g9P3t9+BJzZax7+7FSXYCnwa+UlWvnW5l0+Od1JLUkeS7gHcCL6qqFXnhgQEhSepyikmS1GVA
SJK6DAhJUpcBIUnqMiC0IiV5pD0pd/l15SHs+31J3vM1fv7NSR7Xbx4fjs+XJuHD+rRS/X1VnTON
D243YEkzzyMIaUySTyb51SR/mWQhyblJ/qTd9PZvxoaenOT6dkPjbyc5ru1/VdvvjiS/uM/7/kKS
DwCXjPUfl2Rbkl9p7R9on/3XSf4gyZNb/wuTfLzt/6Ij8pehFc+A0Er1xH2mmH58bNvuqnou8L+B
twEvBjYCvzQ25jzg54BvB57Bo//RfmVVzQPPAr43ybPG9nmoqr67qt7V2quAdwCfqKpXJTkNeBXw
gqo6F1gAXpHkJOB/AD8M/HPgaYfp70A6IKeYtFIdaIppe1t+FHhyVX0e+HySh5Ksbts+XFV3AyR5
J/DdwHXAS5JsYfRv60xGj0q5ve3z7n0+578D1449ymFjG//B0W/XcALwl8A3A/+3qna2z/s9Rk/7
lQZlQEj7+3JbfmVsfbm9/G9m30cQVJKzgH8PfGdVPZDkbcBJY2O+uM8+fwE8L8nrq+ohRo+Vvqmq
XjY+KMk5nc+TBucUk/T4nJfkrHbu4ceBDwAnMwqBB5OcAVx4kPe4GrgR+IMkq4BbgPOTPBMgyZPa
b1l/HDgryTPafi/rvpt0mHkEoZXqiUnGfwr0vVU18aWujKZ+XsfoHMSfA9dX1VeSfITR7wrcDXzw
YG9SVb+e5CnA24GfAH4KeGeSE9uQV1XVJ9q01f9Kcj+jMPq2Q6hVelx8WJ8kqcspJklSlwEhSeoy
ICRJXQaEJKnLgJAkdRkQkqQuA0KS1GVASJK6/j/WIJIuPR89VwAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passenger survival from Embarked</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Embarked&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a16347c88&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
gg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passengers Sex</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Sex&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a164200f0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYgAAAEKCAYAAAAIO8L1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAEchJREFUeJzt3XuwXWV9xvHvA0GteAlIoDShjZeMl9oqcETUtoPiqFAr
1BYvVYk007RTRB17kdpp7VStWm1VHIvNiBocq1LUEh2qpShaa7WeVOQiOqTokNNQORS5WEYc9Nc/
9ht7DG+SnUPW2cdzvp+ZPWutd7177V8mK+fJ+6691klVIUnSrg6YdAGSpMXJgJAkdRkQkqQuA0KS
1GVASJK6DAhJUpcBIUnqMiAkSV2DBkSSlUkuTPK1JNckeUKSQ5NckuTatjyk9U2Sc5JsS3JFkmOG
rE2StGcZ8k7qJJuBf6mqdyW5F3Bf4FXAzVX1hiRnA4dU1SuTnAycBZwMPB54W1U9fk/HP+yww2rt
2rWD1S9JS9HWrVtvqqpVe+s3WEAkeQDwFeAhNedDknwdOKGqbkhyJHBZVT08yd+29Q/s2m93nzE1
NVXT09OD1C9JS1WSrVU1tbd+Q04xPQSYBd6T5MtJ3pXkYOCInT/02/Lw1n81sH3O+2da249IsjHJ
dJLp2dnZAcuXpOVtyIBYARwDnFtVRwP/C5y9h/7ptN1teFNVm6pqqqqmVq3a6whJkjRPQwbEDDBT
VV9s2xcyCoxvtakl2vLGOf2PmvP+NcCOAeuTJO3BYAFRVf8NbE/y8NZ0IvBVYAuwvrWtBy5q61uA
09u3mY4Hbt3T9QdJ0rBWDHz8s4D3t28wXQecwSiULkiyAbgeOK31vZjRN5i2AXe0vpKkCRk0IKrq
cqB3pfzETt8CzhyyHknS+LyTWpLUZUBIkroMCElS19AXqRe9Y//g/EmXoEVo65tOn3QJ0sQ5gpAk
dRkQkqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKkLgNCktRlQEiSugwISVKX
ASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXYMGRJJvJrkyyeVJ
plvboUkuSXJtWx7S2pPknCTbklyR5Jgha5Mk7dlCjCCeXFWPraqptn02cGlVrQMubdsAJwHr2msj
cO4C1CZJ2o1JTDGdAmxu65uBU+e0n18jXwBWJjlyAvVJkhg+IAr4pyRbk2xsbUdU1Q0AbXl4a18N
bJ/z3pnW9iOSbEwynWR6dnZ2wNIlaXlbMfDxn1RVO5IcDlyS5Gt76JtOW92toWoTsAlgamrqbvsl
SfvHoCOIqtrRljcCHwWOA761c+qoLW9s3WeAo+a8fQ2wY8j6JEm7N1hAJDk4yf13rgNPA64CtgDr
W7f1wEVtfQtwevs20/HArTunoiRJC2/IKaYjgI8m2fk5f1dVn0jyJeCCJBuA64HTWv+LgZOBbcAd
wBkD1iZJ2ovBAqKqrgMe02n/H+DETnsBZw5VjyRp33gntSSpy4CQJHUZEJKkLgNCktRlQEiSugwI
SVKXASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaEJKnLgJAk
dRkQkqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKkrsEDIsmBSb6c5ONt+8FJ
vpjk2iQfSnKv1n7vtr2t7V87dG2SpN1biBHEy4Br5my/EXhLVa0Dvg1saO0bgG9X1cOAt7R+kqQJ
GTQgkqwBfhl4V9sO8BTgwtZlM3BqWz+lbdP2n9j6S5ImYOgRxFuBPwR+0LYfBNxSVXe17RlgdVtf
DWwHaPtvbf1/RJKNSaaTTM/Ozg5ZuyQta4MFRJJnAjdW1da5zZ2uNca+/2+o2lRVU1U1tWrVqv1Q
qSSpZ8WAx34S8KwkJwP3AR7AaESxMsmKNkpYA+xo/WeAo4CZJCuABwI3D1ifJGkPBhtBVNUfVdWa
qloLPA/4VFW9APg08Out23rgora+pW3T9n+qqu42gpAkLYxJ3AfxSuAVSbYxusZwXms/D3hQa38F
cPYEapMkNUNOMf1QVV0GXNbWrwOO6/T5LnDaQtQjSdo776SWJHUZEJKkLgNCktRlQEiSugwISVKX
ASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaEJKnLgJAkdY0V
EEkuHadNkrR07PFXjia5D3Bf4LAkhwBpux4A/NTAtUmSJmhvv5P6t4GXMwqDrfx/QNwGvGPAuiRJ
E7bHgKiqtwFvS3JWVb19gWqSJC0CextBAFBVb0/yRGDt3PdU1fkD1SVJmrCxAiLJ+4CHApcD32/N
BRgQkrREjRUQwBTwqKqqIYuRJC0e494HcRXwk0MWIklaXMYdQRwGfDXJvwN37mysqmcNUpUkrv/z
n5t0CVqEfvpPr1ywzxo3IP5sXw/c7qH4LHDv9jkXVtWrkzwY+CBwKPAfwIuq6ntJ7s3omsaxwP8A
z62qb+7r50qS9o9xv8X0mXkc+07gKVX1nSQHAZ9L8o/AK4C3VNUHk7wT2ACc25bfrqqHJXke8Ebg
ufP4XEnSfjDuozZuT3Jbe303yfeT3Lan99TId9rmQe1VwFOAC1v7ZuDUtn5K26btPzHJzhvzJEkL
bNwRxP3nbic5FThub+9LciCjO7AfxujO6/8Ebqmqu1qXGWB1W18NbG+fd1eSW4EHATeNU6Mkaf+a
19Ncq+ofGI0E9tbv+1X1WGANo0B5ZK9bW/ZGC3f7Wm2SjUmmk0zPzs7uQ9WSpH0x7o1yz56zeQCj
+yLGvieiqm5JchlwPLAyyYo2ilgD7GjdZoCjgJkkK4AHAjd3jrUJ2AQwNTXlfRmSNJBxRxC/Muf1
dOB2RtcMdivJqiQr2/pPAE8FrgE+Dfx667YeuKitb2nbtP2f8sY8SZqcca9BnDGPYx8JbG7XIQ4A
Lqiqjyf5KvDBJK8Fvgyc1/qfB7wvyTZGI4fnzeMzJUn7ybhTTGuAtwNPYjS19DngZVU1s7v3VNUV
wNGd9uvoXOCuqu8Cp41XtiRpaONOMb2H0RTQTzH6ttHHWpskaYkaNyBWVdV7ququ9novsGrAuiRJ
EzZuQNyU5IVJDmyvFzJ6HIYkaYkaNyB+E3gO8N/ADYy+ZTSfC9eSpB8T4z6s7zXA+qr6NkCSQ4E3
MwoOSdISNO4I4ud3hgNAVd1M5xtKkqSlY9yAOCDJITs32ghi3NGHJOnH0Lg/5P8K+HySCxndB/Ec
4HWDVSVJmrhx76Q+P8k0owf0BXh2VX110MokSRM19jRRCwRDQZKWiXk97luStPQZEJKkLgNCktRl
QEiSugwISVKXASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaE
JKnLgJAkdQ0WEEmOSvLpJNckuTrJy1r7oUkuSXJtWx7S2pPknCTbklyR5JihapMk7d2QI4i7gN+r
qkcCxwNnJnkUcDZwaVWtAy5t2wAnAevaayNw7oC1SZL2YrCAqKobquo/2vrtwDXAauAUYHPrthk4
ta2fApxfI18AViY5cqj6JEl7tiDXIJKsBY4GvggcUVU3wChEgMNbt9XA9jlvm2ltkqQJGDwgktwP
+DDw8qq6bU9dO23VOd7GJNNJpmdnZ/dXmZKkXQwaEEkOYhQO76+qj7Tmb+2cOmrLG1v7DHDUnLev
AXbsesyq2lRVU1U1tWrVquGKl6RlbshvMQU4D7imqv56zq4twPq2vh64aE776e3bTMcDt+6cipIk
LbwVAx77ScCLgCuTXN7aXgW8AbggyQbgeuC0tu9i4GRgG3AHcMaAtUmS9mKwgKiqz9G/rgBwYqd/
AWcOVY8kad94J7UkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElSlwEhSeoyICRJXQaEJKnLgJAkdRkQ
kqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKkLgNCktRlQEiSugwISVKXASFJ
6jIgJEldBoQkqcuAkCR1GRCSpK7BAiLJu5PcmOSqOW2HJrkkybVteUhrT5JzkmxLckWSY4aqS5I0
niFHEO8FnrFL29nApVW1Dri0bQOcBKxrr43AuQPWJUkaw2ABUVWfBW7epfkUYHNb3wycOqf9/Br5
ArAyyZFD1SZJ2ruFvgZxRFXdANCWh7f21cD2Of1mWpskaUIWy0XqdNqq2zHZmGQ6yfTs7OzAZUnS
8rXQAfGtnVNHbXlja58BjprTbw2wo3eAqtpUVVNVNbVq1apBi5Wk5WyhA2ILsL6trwcumtN+evs2
0/HArTunoiRJk7FiqAMn+QBwAnBYkhng1cAbgAuSbACuB05r3S8GTga2AXcAZwxVlyRpPIMFRFU9
fze7Tuz0LeDMoWqRJO27xXKRWpK0yBgQkqQuA0KS1GVASJK6DAhJUpcBIUnqMiAkSV0GhCSpy4CQ
JHUZEJKkLgNCktRlQEiSugwISVKXASFJ6jIgJEldBoQkqcuAkCR1GRCSpC4DQpLUZUBIkroMCElS
lwEhSeoyICRJXQaEJKnLgJAkdRkQkqQuA0KS1LWoAiLJM5J8Pcm2JGdPuh5JWs4WTUAkORB4B3AS
8Cjg+UkeNdmqJGn5WjQBARwHbKuq66rqe8AHgVMmXJMkLVuLKSBWA9vnbM+0NknSBKyYdAFzpNNW
d+uUbAQ2ts3vJPn6oFUtL4cBN026iMUgb14/6RL0ozw3d3p170flPvuZcTotpoCYAY6as70G2LFr
p6raBGxaqKKWkyTTVTU16TqkXXluTsZimmL6ErAuyYOT3At4HrBlwjVJ0rK1aEYQVXVXkpcAnwQO
BN5dVVdPuCxJWrYWTUAAVNXFwMWTrmMZc+pOi5Xn5gSk6m7XgSVJWlTXICRJi4gBoa4kJyT5+KTr
0NKQ5KVJrkny/oGO/2dJfn+IYy9ni+oahKQl63eBk6rqG5MuRONzBLGEJVmb5GtJ3pXkqiTvT/LU
JP+a5Nokx7XX55N8uS0f3jnOwUneneRLrZ+PQNHYkrwTeAiwJckf986lJC9O8g9JPpbkG0lekuQV
rc8Xkhza+v1We+9Xknw4yX07n/fQJJ9IsjXJvyR5xML+iZcOA2LpexjwNuDngUcAvwH8AvD7wKuA
rwG/VFVHA38K/EXnGH8MfKqqHgc8GXhTkoMXoHYtAVX1O4xuen0ycDC7P5cezej8PA54HXBHOy//
DTi99flIVT2uqh4DXANs6HzkJuCsqjqW0Xn+N8P8yZY+p5iWvm9U1ZUASa4GLq2qSnIlsBZ4ILA5
yTpGjzY5qHOMpwHPmjPHex/gpxn9A5X2xe7OJYBPV9XtwO1JbgU+1tqvZPQfHIBHJ3ktsBK4H6P7
pn4oyf2AJwJ/n/zwkRT3HuIPshwYEEvfnXPWfzBn+weM/v5fw+gf5q8mWQtc1jlGgF+rKp97pXuq
ey4leTx7P1cB3gucWlVfSfJi4IRdjn8AcEtVPXb/lr08OcWkBwL/1dZfvJs+nwTOSvsvWZKjF6Au
LU339Fy6P3BDkoOAF+y6s6puA76R5LR2/CR5zD2sedkyIPSXwOuT/CujR5z0vIbR1NMVSa5q29J8
3NNz6U+ALwKXMLp+1vMCYEOSrwBX4++VmTfvpJYkdTmCkCR1GRCSpC4DQpLUZUBIkroMCElSlwEh
zVN7rtDVSa5Icnm72UtaMryTWpqHJE8AngkcU1V3JjkMuNeEy5L2K0cQ0vwcCdxUVXcCVNVNVbUj
ybFJPtOeJPrJJEcmWdGeQHoCQJLXJ3ndJIuXxuGNctI8tIfCfQ64L/DPwIeAzwOfAU6pqtkkzwWe
XlW/meRngQuBlzK6e/3xVfW9yVQvjccpJmkequo7SY4FfpHRY6s/BLyW0SOrL2mPGjoQuKH1vzrJ
+xg9ofQJhoN+HBgQ0jxV1fcZPf32svb49DOBq6vqCbt5y88BtwBHLEyF0j3jNQhpHpI8vP0OjZ0e
y+j3Y6xqF7BJclCbWiLJs4EHAb8EnJNk5ULXLO0rr0FI89Cml97O6BfX3AVsAzYCa4BzGD1GfQXw
VuCjjK5PnFhV25O8FDi2qtZPonZpXAaEJKnLKSZJUpcBIUnqMiAkSV0GhCSpy4CQJHUZEJKkLgNC
ktRlQEiSuv4PoyuAlRWoZYIAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot Passengers Sex Survival</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Sex&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a161bd208&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
rkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot survival by Age</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span>  <span class="p">(</span><span class="mi">40</span><span class="p">,</span> <span class="mi">30</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="n">hue</span> <span class="o">=</span> <span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a165549b0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAACPUAAAaPCAYAAAD7G0m9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzs3V+olYWexvHfe9ymkf0htQnbnbYRHcrxD6MWEYYTgzUS
+2JIt15kYWFkgVdBXXSygSCyuYjqoqCDRaUWMVjNJHNVw1RUrrBOWSGNM+M2mdIYmQrJ7J2LKWfO
ZOaZ1nY9e/v53LjXWu9617OUfSNf3rdp27YAAAAAAAAAAIAcv+r1AAAAAAAAAAAA4A+JegAAAAAA
AAAAIIyoBwAAAAAAAAAAwoh6AAAAAAAAAAAgjKgHAAAAAAAAAADCiHoAAAAAAAAAACCMqAcAAAAA
AAAAAMKIegAAAAAAAAAAIIyoBwAAAAAAAAAAwvT1ekA3TZkypR0YGOj1DAAAAAAAAAAAOKJOp7O3
bdupP3fcmIp6BgYGauvWrb2eAQAAAAAAAAAAR9Q0zb8ey3FuvwUAAAAAAAAAAGFEPQAAAAAAAAAA
EEbUAwAAAAAAAAAAYfp6PQAAAACAP97BgwdreHi4Dhw40Ospv9jEiROrv7+/xo8f3+spAAAAADFE
PQAAAACj0PDwcJ166qk1MDBQTdP0es7/W9u2tW/fvhoeHq7p06f3eg4AAABADLffAgAAABiFDhw4
UJMnTx7VQU9VVdM0NXny5DFxxSEAAACAbhL1AAAAAIxSoz3o+cFY+R4AAAAA3STqAQAAAAAAAACA
MKIeAAAAgDHi3nvvrRkzZtSsWbNqzpw59eabb/7ic77wwgt13333dWFd1aRJk7pyHgAAAIATQV+v
BwAAAADwy73xxhv10ksv1TvvvFMTJkyovXv31jfffHNM7/3222+rr+/I/000ODhYg4OD3ZwKAAAA
wDFwpR4AAACAMWDPnj01ZcqUmjBhQlVVTZkypaZNm1YDAwO1d+/eqqraunVrLVy4sKqq1q5dW6tW
rapFixbVihUr6tJLL60PPvjg8PkWLlxYnU6n1q9fX7fddlvt37+/BgYG6rvvvquqqq+//rrOPffc
OnjwYH3yySd19dVX19y5c2vBggX10UcfVVXVzp0767LLLqv58+fXXXfddRz/NgAAAABGP1EPAAAA
wBiwaNGi2rVrV1144YW1evXqevXVV3/2PZ1OpzZv3lzPPPNMLVu2rJ599tmq+u9A6NNPP625c+ce
Pvb000+v2bNnHz7viy++WFdddVWNHz++Vq1aVQ899FB1Op164IEHavXq1VVVtWbNmrrlllvq7bff
rrPPPnsEvjUAAADA2CXqAQAAABgDJk2aVJ1Opx577LGaOnVqDQ0N1fr164/6nsHBwTr55JOrqmrp
0qX13HPPVVXVs88+W0uWLPnR8UNDQ7Vp06aqqtq4cWMNDQ3Vl19+Wa+//notWbKk5syZUzfffHPt
2bOnqqpee+21Wr58eVVVXXfddd36qgAAAAAnhCPfLB0AAACAUWfcuHG1cOHCWrhwYc2cObOeeOKJ
6uvrO3zLrAMHDvzB8aeccsrhn88555yaPHlyvffee7Vp06Z69NFHf3T+wcHBuvPOO+uLL76oTqdT
V155ZX311Vd1xhln1LZt2464qWmaLn5DAAAAgBOHK/UAAAAAjAEff/xx7dix4/Djbdu21XnnnVcD
AwPV6XSqqur5558/6jmWLVtW999/f+3fv79mzpz5o9cnTZpUl1xySa1Zs6auueaaGjduXJ122mk1
ffr0w1f5adu23n333aqquvzyy2vjxo1VVfX000935XsCAAAAnChEPQAAAABjwJdfflnXX399XXzx
xTVr1qzavn17rV27tu6+++5as2ZNLViwoMaNG3fUc1x77bW1cePGWrp06U8eMzQ0VE899VQNDQ0d
fu7pp5+uxx9/vGbPnl0zZsyozZs3V1XVgw8+WI888kjNnz+/9u/f350vCgAAAHCCaNq27fWGrpk3
b167devWXs8AAAAAGHEffvhhXXTRRb2e0TVj7fsAAAAA/JSmaTpt2877ueNcqQcAAAAAAAAAAMKI
egAAAAAAAAAAIIyoBwAAAAAAAAAAwoh6AAAAAAAAAAAgjKgHAAAAAAAAAADCiHoAAAAAAAAAACBM
X68HAAAAANB9c29/sqvn66xbcUzHbdmypdasWVOHDh2qm266qe64446u7gAAAAA4UbhSDwAAAABd
cejQobr11lvr5Zdfru3bt9eGDRtq+/btvZ4FAAAAMCqJegAAAADoirfeeqsuuOCCOv/88+ukk06q
ZcuW1ebNm3s9CwAAAGBUEvUAAAAA0BW7d++uc8899/Dj/v7+2r17dw8XAQAAAIxeoh4AAAAAuqJt
2x891zRND5YAAAAAjH6iHgAAAAC6or+/v3bt2nX48fDwcE2bNq2HiwAAAABGL1EPAAAAAF0xf/78
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
AAAAAAAAAABAGFEPAAAAAAAAAACEEfUAAAAAAAAAAEAYUQ8AAAAAAAAAAIQR9QAAAAAAAAAAQBhR
DwAAAAAAAAAAhBH1AAAAAAAAAABAGFEPAAAAAAAAAACEEfUAAAAAAAAAAEAYUQ8AAAAAAAAAAIQR
9QAAAAAAAAAAQBhRDwAAAAAAAAAAhBH1AAAAAAAAAABAGFEPAAAAAAAAAACEEfUAAAAAAAAAAEAY
UQ8AAAAAAAAAAIQR9QAAAAAAAAAAQBhRDwAAAAAAAAAAhBH1AAAAAAAAAABAGFEPAAAAAAAAAACE
EfUAAAAAAAAAAEAYUQ8AAAAAAAAAAIQR9QAAAAAAAAAAQBhRDwAAAAAAAAAAhBH1AAAAAAAAAABA
GFEPAAAAAAAAAACEEfUAAAAAAAAAAEAYUQ8AAAAAAAAAAIQR9QAAAAAAAAAAQBhRDwAAAAAAAAAA
hOnr9QAAAMaGubc/eUzHddatGOEl/MC/CQAAAAAAjF6u1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbU
AwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFE
PQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG
1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABh
RD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQ
RtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAA
YUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAA
EEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAA
AGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAA
ABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAA
AABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAA
AAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAA
AAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAAAAAAAGFEPQAAAAAA
AAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAAAAAAABBG1AMAAAAA
AAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAAAAAAAABhRD0AAAAA
AAAAABBG1AMAAAAAAAAAAGFEPQAAAAAAAAAAEEbUAwAAAAAAAAAAYUQ9AAAAAAAAAAAQRtQDAAAA
AAAAAABhRizqaZrmd03TfNY0zfv/67m1TdPsbv6LvfsLtey8yzj+/NJBipJipGNNmg4VKYXQajCH
CAoSlcY01AYVxCAk2Mr0wgrehFaURFIvhEGktEWJbUgjEr2KFoy2IaBBqLQZiE0U/5QS2iGlCUSI
ICqJrxfZxeN4ZmZ3MvvMM5nPBzaz17vftdbv4pzDXHxZe+bJzevWM5x7y8z808x8eWY+vKsZAQAA
AAAAAACg0S6f1PNAklsOWP/dtdb1m9cjp384M69L8okk705yXZLbZ+a6Hc4JAAAAAAAAAABVdhb1
rLUeT/LCeZx6Y5Ivr7W+stb6ryR/nOS2CzocAAAAAAAAAAAU2+WTes7kgzPzpc3Xc111wOdvTvK1
fcenNmsAAAAAAAAAAHBZOHLI9/u9JB9Jsjb//k6S9522Zw44b53pgjNzPMnxJDl27NiFmRIAALjs
3HDXg1vvffjKE1vtO3b3U+c7DtTa9nfl5Ik7djwJAAAAALy2HeqTetZa31hrvbzW+u8kf5BXvmrr
dKeSvGXf8bVJnj3LNe9ba+2ttfaOHj16YQcGAAAAAAAAAICL4FCjnpm5et/hTyd5+oBtX0zytpn5
3pn5tiQ/n+QzhzEfAAAAAAAAAAA02NnXb83MQ0luSvLGmTmV5J4kN83M9Xnl67SeSfKBzd5rknxy
rXXrWuulmflgks8meV2S+9daf7+rOQEAAAAAAAAAoM3Oop611u0HLH/qDHufTXLrvuNHkjyyo9EA
AAAAAAAAAKDaoX79FgAAAAAAAAAAcG6iHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAo
I+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACA
MqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAA
KCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAA
gDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAA
ACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAA
AIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAA
AAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAA
AACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAA
AAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAA
AAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAA
AAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAA
AAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAA
AAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAA
AAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAA
AAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEA
AAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4A
AAAAAACFu1aCAAAgAElEQVQAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIe
AAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPq
AQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKi
HgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj
6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAy
oh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAo
I+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACA
MqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAA
KCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAA
gDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAA
ACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAA
AIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAAAACAMqIeAAAAAAAA
AAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAAAAAAKCPqAQAAAAAA
AACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAAAAAAgDKiHgAAAAAA
AAAAKCPqAQAAAAAAAACAMqIeAAAAAAAAAAAoI+oBAAAAAAAAAIAyoh4AAAAAAAAAACgj6gEAAAAA
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
zr5kLJWzLxlL5TzWx3L5GOPSCy107mTrs5PPJXn2oevPSvLQmmU+V1Xnknxzki+terIxxtuS00Dk
zAAAHoZJREFUvO0kK1JVt48xrjrJY7vl7EvGUjn7krFUjrH0y1gqZ18ylsrZl4ylcoylX8ZSOfuS
sVTOvmQslWMs/TKWytmXjKVy9iVjqRxj6ZexVM6+ZCyVsy8ZS+UYS7+MpXL2JWOpnH3JWCrHWPpl
LJWzLxlL5exLxlI5xtIvY6mcfclYKmdfMpbKMZbtnMV/v/VbSZ5bVc+pqscnuS7J+48s8/4kr54u
vzLJfx1L/0ohAAAAAAAAAAA4I4v/pp4xxsNV9SNJfjXJ45K8Y4xxX1XdlOT2Mcb7k7w9ybur6oEc
/Iae65ZeTwAAAAAAAAAAOCtn8d9vZYzxgSQfOHLbDYcufzXJqxZYlRP9t11Nc/YlY6mcfclYKsdY
+mUslbMvGUvl7EvGUjnG0i9jqZx9yVgqZ18ylsoxln4ZS+XsS8ZSOfuSsVSOsfTLWCpnXzKWytmX
jKVyjKVfxlI5+5KxVM6+ZCyVYyz9MpbK2ZeMpXL2JWOpHGPpl7FUzr5kLJWzLxlL5RjLFsr/agUA
AAAAAAAAAL18w1mvAAAAAAAAAAAA8PX2ttRTVa+oqk9U1QNV9cYNy72yqkZVXTVd/5tVdUdV3TP9
/dJdMqrq+6rqt6vqvqr6uem276mquw79+WpV/d2TZFTVWw49zyer6o8P3fd/Dt33/vVba6ucy6rq
I1X1saq6u6qumW6/+lDGx6vq7+2QcXlV/dr0/B+tqmdNt7+wqv77tA3vrqp/sGksh57vHVX1haq6
d839VVX/flqfu6vqxds875HnePa0Xe6f1u/1p5TzhKr6H9M2vq+q/vWKZS6uqvdMObdV1RXHzZme
53HTfv6V08ioqgen19ddVXX7ivt33l7T8zy5qt5bVf9z2j/fNWdOVT3vyOv4y1X1Y3OPpar+6bTP
762qn6+qJxy5f679/vop476j4zjpWFa9BqvqqVX14ar61PT3U9Y89tXTMp+qqlefIOdV01j+b03z
+5rHbvtesSrj5un4uruq3ldVT94lY0POm6aMu6rqQ1X1jDWP3Wqbrco4dN+P18F74tPOMOM4712r
tteNVfX7h57jmjWPPfG+n25/3fT4+6rqJ+bOmF7X58fwYFXdtUvGhpwXVtWtU87tVXX1mseeeN9X
1XfUwfv4PVX1n6vqm3YZS615751zfpkhY6vjeEPObHPYhozZ5rANGXPPX7vmXHC/rMs4dP9c8+Su
OSceS804T24aR807T64by2xz5YaM2ebJGTJ2nVtmm49rzeeTqnpOHfxM+qlp/zx+zeP/5fT8n6iq
l8+dUVVXVNWfH9peb92wvdbl/Mi0jmtfi9Ny2+z7dRk/O22De+vgPfSik2bMlLPN3LIu4+3TbXfX
weewJ615/In3/aH7f6qqvrJhO1wwY9ecbY+xDdvrnVX1O4ce/8I1j9/l+Kqq+jd1cK7o/qr60ZNm
zJSzy/H1G4ce+1BV/fIpjeVlVXXnlPObVfWtax6/yxz20inj3qp6V1Wd22Us07Jfdx6nZpyLd8nY
9nVygZzZ5uINGbPOxWsyZpsjN+Ucun2WeXLNWGabv2bI2Pr8wZqcqhnnyTUZs85fG3Jmm782ZJzG
/PVgHTlPXDOfO1yTMfd5w1UZp3HecFXO3J+7H5Fx6L65Pg+vGseNNeP5vBly1m6HbZareT8PrxrH
aZw3XJUz93nDVRmznjecln3Ed0M1/9yyKmPuuWVVxmnMLaty5p5b1n5fV/PNLavGcRpzy8qx1Eyv
+1rz3eOcx/AMGduen1qXM+e5743f1c51fE3LPuL72jqFz15rjTH27k+SxyX5dJJvSfL4JB9PcuWK
5S5J8utJbk1y1XTbi5I8Y7r87Ul+/6QZSZ6b5GNJnjJd/0srnuepSb6U5IknHceh5V+X5B2Hrn9l
ru2Vg/8D7rXT5SuTPDhdfmKSc9Plpyf5wvnrJ8j4T0lePV1+aZJ3T5e/Lclzp8vPSPL5JE/eYlzf
neTFSe5dc/81ST6YpJJ8Z5LbTnCsPT3Jiw8dT59cMa45cirJk6bLFyW5Lcl3Hlnmh5O8dbp8XZL3
nPD188+S/FySX1lx384ZSR5M8rQN9++8vabneVeSH5ouP/7oMTNXzqHj+w+SXD5nRpJnJvmdJN84
Xf+PSf7hKeyTb09y7/nXc5L/cv41t8tYVr0Gk/xEkjdOl9+Y5M0rHvfUJJ+Z/n7KdPkpx8x5fpLn
Jflopvl9zX7bao5dk/G38v/nvzevGctx5/FVOd906PKPnt/fJ91mqzKm25+d5FeT/G5WvEaXyJiW
2eq9a8P2ujHJj1/gcbvu+++ZXicXT9dXvb/vlHHk/n+X5IZTOr4+lOR7p8vXJPno3Ps+yW8l+evT
5X+c5E07bq+V772ZcX7ZJeM4x/GGnNnmsA0Zs81hGzLmnr9OnLPtflmXMV2fc548cc6uY8mM8+SG
jLnnyW1+5t5prtwwltnmyV0ytt3vF8iZbT7Oms8nOfh59brp9rdm+gx55LFXTs97cZLnTHmPmznj
iqx5Xz1Gzoum53kw61+L2+77dRnXTPdVkp9fM5bjzC0nztn2GNuQcXgu/slM75dz7vvp+lVJ3r1u
XbfNmCFnq2Nsw/Z6Z5JXXuCxux5f/yjJLUm+Ybpv3TmpXY+vC+bsenwdWeYXk1x/SmP5ZJLnT7f/
cJJ3znwc/9Ukv5fk26bbb0ryg7uMZVr+687jZMa5eMeMK7LlXLwhZ7a5eEPGrHPxmozZ5shNOdNt
s82Ta8byzsw0f+2SMS239fmDNTmzzpPr9smh+3aevzaMZbb5a1VGDv5x+GnMXw/myGs7M587XJMx
93nDVRmncd5wVc7cn7sfkTHdPufn4VXjuDEzns/bJWfTdtgyY+7PwxvXJfOdN1w1lrnPG67KmPW8
4bT8I74byvxzy6qMueeWVRmnMbesypl7bln5fV3mnVtWjePGzD+3rMqZ9XV/5DF/kOTyuY/hk2ZM
9x3rZ7AVObO+VlZlnMLxtfL72pzCZ691f/b1N/VcneSBMcZnxhhfS/ILSa5dsdybcnCAfvX8DWOM
j40xHpqu3pfkCVV18Qkz/kmSnxlj/NH03F9Y8TyvTPLBMcaf7TCO874/Bx8+j2ubnJHkfEP2m5M8
lCRjjD8bYzw83f6EabmTZlyZ5Nemyx85f/8Y45NjjE9Nlx/KQXHo0gsNaozx6zkoTK1zbZJbxoFb
kzy5qp5+oec9kvH5Mcad0+U/TXJ/Dl7Yc+eMMcb5f2Fz0fTn6La+NgdvJkny3iQvq6o6Tk4d/Hak
v53kP6xZZOeMLey8veqgzf3dSd6eJGOMr40x/vjIYjvnHPKyJJ8eY/zuKWScS/KNdfCvX56Y6bV3
JGPXffL8JLceej3/tyRHf+vWscey5jV4eH3flWTVbyl7eZIPjzG+NM2fH07yiuPkjDHuH2N8YtP6
5Rhz7JqMDx2a/25N8qxdMjbkfPnQ1b+Q1fPs1ttsw9z4liT/Ys3zL5VxLFvM8+vstO+TvDbJvx1j
/O9pmVXv77tmJDn4l4NJvi+r3993Pr6y5v39iF33/fNyUKLO9Ni/v8tY/l97dx98W1XXcfy9rldQ
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">bins</span> <span class="o">=</span> <span class="mi">40</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a169f6f60&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">bins</span> <span class="o">=</span> <span class="mi">40</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a1915dd30&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
PytDREj4OkFTAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Cleaning-Data">Cleaning Data<a class="anchor-link" href="#Cleaning-Data">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[19]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0</td>
      <td>3</td>
      <td>Saundercock, Mr. William Henry</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5. 2151</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
      <td>Andersson, Mr. Anders Johan</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>347082</td>
      <td>31.2750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>Vestrom, Miss. Hulda Amanda Adolfina</td>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>350406</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>248706</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Planke, Mrs. Julius (Emelia Maria Vande...</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>345763</td>
      <td>18.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>0</td>
      <td>2</td>
      <td>Fynney, Mr. Joseph J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>239865</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>Beesley, Mr. Lawrence</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>248698</td>
      <td>13.0000</td>
      <td>D56</td>
      <td>S</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>1</td>
      <td>3</td>
      <td>McGowan, Miss. Anna "Annie"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>330923</td>
      <td>8.0292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>Sloper, Mr. William Thompson</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113788</td>
      <td>35.5000</td>
      <td>A6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Miss. Torborg Danira</td>
      <td>female</td>
      <td>8.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>1</td>
      <td>3</td>
      <td>Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>5</td>
      <td>347077</td>
      <td>31.3875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1</td>
      <td>3</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330959</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>0</td>
      <td>3</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>862</td>
      <td>0</td>
      <td>2</td>
      <td>Giles, Mr. Frederick Edward</td>
      <td>male</td>
      <td>21.0</td>
      <td>1</td>
      <td>0</td>
      <td>28134</td>
      <td>11.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>862</th>
      <td>863</td>
      <td>1</td>
      <td>1</td>
      <td>Swift, Mrs. Frederick Joel (Margaret Welles Ba...</td>
      <td>female</td>
      <td>48.0</td>
      <td>0</td>
      <td>0</td>
      <td>17466</td>
      <td>25.9292</td>
      <td>D17</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>864</th>
      <td>865</td>
      <td>0</td>
      <td>2</td>
      <td>Gill, Mr. John William</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>233866</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>865</th>
      <td>866</td>
      <td>1</td>
      <td>2</td>
      <td>Bystrom, Mrs. (Karolina)</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>236852</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>866</th>
      <td>867</td>
      <td>1</td>
      <td>2</td>
      <td>Duran y More, Miss. Asuncion</td>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>SC/PARIS 2149</td>
      <td>13.8583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>867</th>
      <td>868</td>
      <td>0</td>
      <td>1</td>
      <td>Roebling, Mr. Washington Augustus II</td>
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17590</td>
      <td>50.4958</td>
      <td>A24</td>
      <td>S</td>
    </tr>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>869</th>
      <td>870</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Master. Harold Theodor</td>
      <td>male</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>870</th>
      <td>871</td>
      <td>0</td>
      <td>3</td>
      <td>Balkic, Mr. Cerin</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>349248</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>871</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>873</th>
      <td>874</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Cruyssen, Mr. Victor</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>345765</td>
      <td>9.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>874</th>
      <td>875</td>
      <td>1</td>
      <td>2</td>
      <td>Abelson, Mrs. Samuel (Hannah Wizosky)</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>P/PP 3381</td>
      <td>24.0000</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>875</th>
      <td>876</td>
      <td>1</td>
      <td>3</td>
      <td>Najib, Miss. Adele Kiamie "Jane"</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>2667</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>876</th>
      <td>877</td>
      <td>0</td>
      <td>3</td>
      <td>Gustafsson, Mr. Alfred Ossian</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>7534</td>
      <td>9.8458</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>877</th>
      <td>878</td>
      <td>0</td>
      <td>3</td>
      <td>Petroff, Mr. Nedelio</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>349212</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>878</th>
      <td>879</td>
      <td>0</td>
      <td>3</td>
      <td>Laleff, Mr. Kristo</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349217</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>879</th>
      <td>880</td>
      <td>1</td>
      <td>1</td>
      <td>Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>11767</td>
      <td>83.1583</td>
      <td>C50</td>
      <td>C</td>
    </tr>
    <tr>
      <th>880</th>
      <td>881</td>
      <td>1</td>
      <td>2</td>
      <td>Shelley, Mrs. William (Imanita Parrish Hall)</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>1</td>
      <td>230433</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>881</th>
      <td>882</td>
      <td>0</td>
      <td>3</td>
      <td>Markun, Mr. Johann</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>349257</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>882</th>
      <td>883</td>
      <td>0</td>
      <td>3</td>
      <td>Dahlberg, Miss. Gerda Ulrika</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7552</td>
      <td>10.5167</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>883</th>
      <td>884</td>
      <td>0</td>
      <td>2</td>
      <td>Banfield, Mr. Frederick James</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A./SOTON 34068</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>884</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="To-be-cleaned:">To be cleaned:<a class="anchor-link" href="#To-be-cleaned:">&#182;</a></h4><ul>
<li>Nans</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Find out where NaNs occur</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">training_set</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span>
            <span class="n">yticklabels</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cbar</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cmap</span> <span class="o">=</span> <span class="s1">&#39;Blues&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[20]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a168406a0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
SUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Columns-we-don't-need:">Columns we don't need:<a class="anchor-link" href="#Columns-we-don't-need:">&#182;</a></h4><ul>
<li>Cabin</li>
<li>Name</li>
<li>Ticket</li>
<li>Embarked</li>
<li>Passenger ID</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># drop Cabin Data</span>
<span class="n">training_set</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Cabin&#39;</span><span class="p">,</span>
                  <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                  <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[22]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># drop rest of columns not needed:</span>
<span class="n">training_set</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Name&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;Ticket&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;Embarked&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;PassengerId&#39;</span><span class="p">],</span>
                    <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                    <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[24]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Find out where NaNs stil occur</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">training_set</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span>
            <span class="n">yticklabels</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cbar</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cmap</span> <span class="o">=</span> <span class="s1">&#39;Blues&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a1937f828&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#plot average ages</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="s1">&#39;Sex&#39;</span><span class="p">,</span>
            <span class="n">y</span> <span class="o">=</span> <span class="s1">&#39;Age&#39;</span><span class="p">,</span>
            <span class="n">data</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a16a4ac50&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
vzX9CG8eAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># replace NaN Ages with average ages based on Sex</span>
<span class="k">def</span> <span class="nf">fill_age</span><span class="p">(</span><span class="n">data</span><span class="p">):</span>
    <span class="n">age</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">sex</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    
    <span class="k">if</span> <span class="n">pd</span><span class="o">.</span><span class="n">isnull</span><span class="p">(</span><span class="n">age</span><span class="p">):</span>
        <span class="k">if</span> <span class="n">sex</span> <span class="ow">is</span> <span class="s1">&#39;male&#39;</span><span class="p">:</span>
            <span class="k">return</span> <span class="mi">29</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">return</span> <span class="mi">25</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">age</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">[[</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">fill_age</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Verify NaNs no longer apear</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">training_set</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span>
            <span class="n">yticklabels</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cbar</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
            <span class="n">cmap</span> <span class="o">=</span> <span class="s1">&#39;Blues&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[29]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a1954d940&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
/QAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># see new distibution after replacing NaNs</span>
<span class="c1"># May affect prediction results with such big changes</span>
<span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">bins</span> <span class="o">=</span> <span class="mi">40</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a19b4e208&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD8CAYAAAB5Pm/hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAEIZJREFUeJzt3XGMHOV9xvHvLzilxddiu4aTY1s1kSwKxcXgE3VKVd2F
NhhSxalUKhBKTErr/kFSUiElppWapBWqK1VJGilFcgOBtCkXGkixjJUEuVyjVA1gE4JNHIobLLBx
cUjA4UCKYvrrHztuNtad93b35nbu9fcjnXbn3dnZh93hubl3Z9eRmUiSyvWmQQeQJNXLopekwln0
klQ4i16SCmfRS1LhLHpJKpxFL0mFs+glqXAWvSQVbsGgAwAsXbo0V61a1fX9XnvtNRYuXDj7gfpk
ru41NZu5utPUXNDcbP3k2rNnz0uZeU7HFTNz4D/r1q3LXjz88MM93a9u5upeU7OZqztNzZXZ3Gz9
5AJ25ww61qkbSSqcRS9JhbPoJalwFr0kFc6il6TCWfSSVDiLXpIKZ9FLUuEsekkqXCO+AkHzx6ot
D05728Gt75zDJJJmyiN6SSqcRS9JhbPoJalwFr0kFc6il6TCWfSSVDiLXpIKZ9FLUuEsekkqnEUv
SYWz6CWpcBa9JBXOopekwnUs+ohYGREPR8T+iHgqIm6uxpdExEMR8Ux1ubgaj4j4VEQciIgnI+LS
uv8jJEnTm8kR/XHglsy8AFgP3BQRFwJbgF2ZuRrYVS0DXAWsrn42A7fPempJ0ox1LPrMPJKZj1fX
XwX2A8uBjcDd1Wp3A++urm8EPpct3wAWRcSyWU8uSZqRruboI2IVcAnwCDCcmUeg9csAOLdabTnw
fNvdDlVjkqQBiMyc2YoRQ8C/A7dl5v0R8UpmLmq7/eXMXBwRDwJ/nZlfr8Z3AR/KzD0nbW8zrakd
hoeH142Pj3cdfnJykqGhoa7vV7eSc+09fGza29YsP7vn7Zb8nNXBXN1rarZ+co2Nje3JzJFO683o
nxKMiDcD9wGfz8z7q+EXI2JZZh6ppmaOVuOHgJVtd18BvHDyNjNzG7ANYGRkJEdHR2cS5adMTEzQ
y/3qVnKuG071Twle3/u2S37O6mCu7jU121zkmslZNwHcAezPzI+33bQd2FRd3wQ80Db+3ursm/XA
sRNTPJKkuTeTI/rLgfcAeyPiiWrsz4CtwL0RcSPwHHBNddtO4GrgAPA68L5ZTSxJ6krHoq/m2mOa
m6+YYv0EbuozlyRplvjJWEkqnEUvSYWz6CWpcBa9JBXOopekwln0klQ4i16SCmfRS1LhLHpJKpxF
L0mFs+glqXAWvSQVzqKXpMJZ9JJUOItekgpn0UtS4Sx6SSqcRS9JhbPoJalwFr0kFc6il6TCWfSS
VDiLXpIKZ9FLUuEsekkqnEUvSYWz6CWpcBa9JBXOopekwln0klQ4i16SCmfRS1LhLHpJKpxFL0mF
s+glqXAWvSQVzqKXpMJZ9JJUuI5FHxF3RsTRiNjXNvbRiDgcEU9UP1e33XZrRByIiKcj4sq6gkuS
ZmYmR/R3ARumGP9EZq6tfnYCRMSFwLXAr1T3+fuIOGO2wkqSutex6DPza8APZri9jcB4Zv4oM58F
DgCX9ZFPktSnfubo3x8RT1ZTO4urseXA823rHKrGJEkDEpnZeaWIVcCOzLyoWh4GXgIS+CtgWWb+
QUR8GvjPzPynar07gJ2Zed8U29wMbAYYHh5eNz4+3nX4yclJhoaGur5f3UrOtffwsWlvW7P87J63
W/JzVgdzda+p2frJNTY2ticzRzqtt6CXjWfmiyeuR8Q/ADuqxUPAyrZVVwAvTLONbcA2gJGRkRwd
He06x8TEBL3cr24l57phy4PT3nbw+t63XfJzVgdzda+p2eYiV09TNxGxrG3xd4ETZ+RsB66NiDMj
4jxgNfBofxElSf3oeEQfEfcAo8DSiDgEfAQYjYi1tKZuDgJ/DJCZT0XEvcC3gePATZn5Rj3RJUkz
0bHoM/O6KYbvOMX6twG39RNKkjR7/GSsJBXOopekwln0klQ4i16SCmfRS1LhLHpJKpxFL0mFs+gl
qXAWvSQVzqKXpMJZ9JJUOItekgpn0UtS4Sx6SSqcRS9JhbPoJalwFr0kFc6il6TCWfSSVDiLXpIK
Z9FLUuEsekkqnEUvSYWz6CWpcBa9JBXOopekwln0klQ4i16SCmfRS1LhLHpJKpxFL0mFs+glqXAW
vSQVzqKXpMJZ9JJUOItekgpn0UtS4Sx6SSqcRS9JhetY9BFxZ0QcjYh9bWNLIuKhiHimulxcjUdE
fCoiDkTEkxFxaZ3hJUmdzeSI/i5gw0ljW4Bdmbka2FUtA1wFrK5+NgO3z05MSVKvOhZ9Zn4N+MFJ
wxuBu6vrdwPvbhv/XLZ8A1gUEctmK6wkqXuRmZ1XilgF7MjMi6rlVzJzUdvtL2fm4ojYAWzNzK9X
47uAD2fm7im2uZnWUT/Dw8PrxsfHuw4/OTnJ0NBQ1/erW8m59h4+Nu1ta5af3fN2S37O6mCu7jU1
Wz+5xsbG9mTmSKf1FvS09enFFGNT/ibJzG3ANoCRkZEcHR3t+sEmJibo5X51KznXDVsenPa2g9f3
vu2Sn7M6mKt7Tc02F7l6PevmxRNTMtXl0Wr8ELCybb0VwAu9x5Mk9avXot8ObKqubwIeaBt/b3X2
zXrgWGYe6TOjJKkPHaduIuIeYBRYGhGHgI8AW4F7I+JG4Dngmmr1ncDVwAHgdeB9NWSWJHWhY9Fn
5nXT3HTFFOsmcFO/oSRJs8dPxkpS4Sx6SSqcRS9JhbPoJalwFr0kFc6il6TCWfSSVDiLXpIKZ9FL
UuEsekkqnEUvSYWz6CWpcBa9JBXOopekwln0klQ4i16SCmfRS1LhLHpJKpxFL0mFs+glqXAWvSQV
zqKXpMJZ9JJUOItekgpn0UtS4Sx6SSqcRS9JhbPoJalwFr0kFc6il6TCWfSSVDiLXpIKZ9FLUuEs
ekkqnEUvSYWz6CWpcBa9JBXOopekwi3o584RcRB4FXgDOJ6ZIxGxBPgCsAo4CPx+Zr7cX0xJUq9m
44h+LDPXZuZItbwF2JWZq4Fd1bIkaUD6OqKfxkZgtLp+NzABfLiGx1GPVm15cNrbDm595xwmkTQX
+j2iT+CrEbEnIjZXY8OZeQSgujy3z8eQJPUhMrP3O0e8JTNfiIhzgYeADwDbM3NR2zovZ+biKe67
GdgMMDw8vG58fLzrx5+cnGRoaKjn/HVpeq69h49Nu86a5Wefchv93Hcm2ZrGXN1pai5obrZ+co2N
je1pmzafVl9F/1MbivgoMAn8ETCamUciYhkwkZnnn+q+IyMjuXv37q4fc2JigtHR0R7S1qvpufqZ
uqlr2qfpz1nTmKt7Tc3WT66ImFHR9zxHHxELgTdl5qvV9XcAfwlsBzYBW6vLB3p9DJXlVL8kbllz
/P/f2JE0u/p5M3YY+FJEnNjOP2fmlyPiMeDeiLgReA64pv+YmiunKmNJ81PPRZ+Z3wUunmL8+8AV
/YSSJM0ePxkrSYWr4zx6naac9pGaySN6SSqcR/Qqgp/2laZn0RdoutK7Zc1xbnB6RTrtWPTzkHPh
krrhHL0kFc4jejWG8+xSPTyil6TCWfSSVDiLXpIKZ9FLUuF8M1bzgqeUSr3ziF6SCmfRS1LhLHpJ
Kpxz9Drt+U8cqnQe0UtS4Sx6SSrcvJ+66XTand+RIul0N++LvlSeNy5ptjh1I0mF84he6sCvT9Z8
5xG9JBXOopekwjl1o+I19Y1tzxjTXLHopRo19ZeMTi/FF71HTZqv/GoGzRbn6CWpcMUf0Ut1cmpG
84FFL81Tnt+vmXLqRpIK5xF9H6Y7ovKNMklNYtFLBfJsM7Wz6AfEN/EkzZXTvugtXJ2O+tnvO/01
4JvEzXPaF72k7nT6IJe10jy+IjXxLwVJTVFb0UfEBuDvgDOAz2Tm1roeS9L80O8BkFM/vaml6CPi
DODTwG8Dh4DHImJ7Zn67jseTpE72Hj7GDdP8oin9F0hdR/SXAQcy87sAETEObAQsekm16PTXwi1r
5ihIA9VV9MuB59uWDwG/VtNjSVJf6vzcQadt37VhYc/bnqnIzNnfaMQ1wJWZ+YfV8nuAyzLzA23r
bAY2V4vnA0/38FBLgZf6jFsHc3WvqdnM1Z2m5oLmZusn1y9l5jmdVqrriP4QsLJteQXwQvsKmbkN
2NbPg0TE7swc6WcbdTBX95qazVzdaWouaG62uchV15eaPQasjojzIuJngGuB7TU9liTpFGo5os/M
4xHxfuArtE6vvDMzn6rjsSRJp1bbefSZuRPYWdf2K31N/dTIXN1rajZzdaepuaC52WrPVcubsZKk
5vAfHpGkws3Loo+IDRHxdEQciIgtA85yZ0QcjYh9bWNLIuKhiHimulw8gFwrI+LhiNgfEU9FxM1N
yBYRPxsRj0bEt6pcH6vGz4uIR6pcX6jexJ9zEXFGRHwzInY0LNfBiNgbEU9ExO5qrAn72aKI+GJE
fKfa19426FwRcX71PJ34+WFEfHDQuapsf1rt9/si4p7q/4fa97F5V/RtX69wFXAhcF1EXDjASHcB
G04a2wLsyszVwK5qea4dB27JzAuA9cBN1fM06Gw/At6emRcDa4ENEbEe+BvgE1Wul4Eb5zjXCTcD
+9uWm5ILYCwz17adijfo1xJa32f15cz8ZeBiWs/dQHNl5tPV87QWWAe8Dnxp0LkiYjnwJ8BIZl5E
60SVa5mLfSwz59UP8DbgK23LtwK3DjjTKmBf2/LTwLLq+jLg6QY8bw/Q+u6hxmQDzgIep/Wp6ZeA
BVO9xnOYZwWtAng7sAOIJuSqHvsgsPSksYG+lsAvAM9SvdfXlFwnZXkH8B9NyMVPvjFgCa0TYXYA
V87FPjbvjuiZ+usVlg8oy3SGM/MIQHV57iDDRMQq4BLgERqQrZoeeQI4CjwE/DfwSmYer1YZ1Gv6
SeBDwP9Wy7/YkFwACXw1IvZUnyqHwb+WbwW+B3y2mu76TEQsbECudtcC91TXB5orMw8Dfws8BxwB
jgF7mIN9bD4WfUwx5qlD04iIIeA+4IOZ+cNB5wHIzDey9Wf1ClpfgHfBVKvNZaaI+B3gaGbuaR+e
YtVB7WuXZ+altKYsb4qI3xxQjnYLgEuB2zPzEuA1BjN9NKVqrvtdwL8MOgtA9Z7ARuA84C3AQlqv
58lmfR+bj0Xf8esVGuDFiFgGUF0eHUSIiHgzrZL/fGbe36RsAJn5CjBB6z2ERRFx4nMdg3hNLwfe
FREHgXFa0zefbEAuADLzheryKK355ssY/Gt5CDiUmY9Uy1+kVfyDznXCVcDjmflitTzoXL8FPJuZ
38vMHwP3A7/OHOxj87Ho58PXK2wHNlXXN9GaH59TERHAHcD+zPx4U7JFxDkRsai6/nO0dv79wMPA
7w0qV2bempkrMnMVrX3q3zLz+kHnAoiIhRHx8yeu05p33seAX8vM/B/g+Yg4vxq6gtZXkQ98/69c
x0+mbWDwuZ4D1kfEWdX/nyeer/r3sUG9SdLnmxpXA/9Fa273zwec5R5a820/pnWEcyOtud1dwDPV
5ZIB5PoNWn8CPgk8Uf1cPehswK8C36xy7QP+ohp/K/AocIDWn9pnDvA1HQV2NCVXleFb1c9TJ/b5
Qb+WVYa1wO7q9fxXYHFDcp0FfB84u22sCbk+Bnyn2vf/EThzLvYxPxkrSYWbj1M3kqQuWPSSVDiL
XpIKZ9FLUuEsekkqnEUvSYWz6CWpcBa9JBXu/wDdQTd6WmTOPgAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[31]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>51.8625</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>11.1333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>30.0708</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>16.7000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.5500</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>31.2750</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8542</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>16.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>29.1250</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>18.0000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.0000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0292</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>35.5000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>8.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>5</td>
      <td>31.3875</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>263.0000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8792</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>21.0</td>
      <td>1</td>
      <td>0</td>
      <td>11.5000</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>48.0</td>
      <td>0</td>
      <td>0</td>
      <td>25.9292</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>25.0</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
    </tr>
    <tr>
      <th>864</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>13.8583</td>
    </tr>
    <tr>
      <th>867</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>50.4958</td>
    </tr>
    <tr>
      <th>868</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5000</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>11.1333</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
    </tr>
    <tr>
      <th>871</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>52.5542</td>
    </tr>
    <tr>
      <th>872</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0000</td>
    </tr>
    <tr>
      <th>873</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.0000</td>
    </tr>
    <tr>
      <th>874</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>28.0</td>
      <td>1</td>
      <td>0</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>875</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
    </tr>
    <tr>
      <th>876</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.8458</td>
    </tr>
    <tr>
      <th>877</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
    </tr>
    <tr>
      <th>878</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
    </tr>
    <tr>
      <th>879</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>83.1583</td>
    </tr>
    <tr>
      <th>880</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>25.0</td>
      <td>0</td>
      <td>1</td>
      <td>26.0000</td>
    </tr>
    <tr>
      <th>881</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
    </tr>
    <tr>
      <th>882</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5167</td>
    </tr>
    <tr>
      <th>883</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>25.0</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">male</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">male</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[33]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>868</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>869</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>871</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>873</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>874</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>877</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>878</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>879</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>882</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>885</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 2 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop column because we only need one</span>
<span class="n">male</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">],</span>
                      <span class="n">drop_first</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">male</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[35]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>1</td>
    </tr>
    <tr>
      <th>862</th>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>1</td>
    </tr>
    <tr>
      <th>865</th>
      <td>0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>1</td>
    </tr>
    <tr>
      <th>868</th>
      <td>1</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1</td>
    </tr>
    <tr>
      <th>870</th>
      <td>1</td>
    </tr>
    <tr>
      <th>871</th>
      <td>0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>1</td>
    </tr>
    <tr>
      <th>873</th>
      <td>1</td>
    </tr>
    <tr>
      <th>874</th>
      <td>0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>1</td>
    </tr>
    <tr>
      <th>877</th>
      <td>1</td>
    </tr>
    <tr>
      <th>878</th>
      <td>1</td>
    </tr>
    <tr>
      <th>879</th>
      <td>0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>1</td>
    </tr>
    <tr>
      <th>882</th>
      <td>0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>1</td>
    </tr>
    <tr>
      <th>884</th>
      <td>1</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 1 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop Sex Column</span>
<span class="n">training_set</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Sex&#39;</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">training_set</span><span class="p">,</span> <span class="n">male</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">training_set</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[38]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Assign-Data-and-Labels">Assign Data and Labels<a class="anchor-link" href="#Assign-Data-and-Labels">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">training_set</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">values</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[40]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[  3.    ,  22.    ,   1.    ,   0.    ,   7.25  ,   1.    ],
       [  1.    ,  38.    ,   1.    ,   0.    ,  71.2833,   0.    ],
       [  3.    ,  26.    ,   0.    ,   0.    ,   7.925 ,   0.    ],
       ..., 
       [  3.    ,  25.    ,   1.    ,   2.    ,  23.45  ,   0.    ],
       [  1.    ,  26.    ,   0.    ,   0.    ,  30.    ,   1.    ],
       [  3.    ,  32.    ,   0.    ,   0.    ,   7.75  ,   1.    ]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">training_set</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1,
       1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
       0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
       1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
       1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
       1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
       1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
       0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
       0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
       1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
       1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Training-the-Model">Training the Model<a class="anchor-link" href="#Training-the-Model">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Train Test Split the data</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span>
                                                    <span class="n">y</span><span class="p">,</span>
                                                    <span class="n">test_size</span> <span class="o">=</span> <span class="mf">0.2</span><span class="p">,</span>
                                                    <span class="n">random_state</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Train Logistic Regression Model</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
<span class="n">classifier</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">random_state</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
<span class="n">classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[44]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class=&#39;ovr&#39;, n_jobs=1,
          penalty=&#39;l2&#39;, random_state=0, solver=&#39;liblinear&#39;, tol=0.0001,
          verbose=0, warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Model-Evaluation">Model Evaluation<a class="anchor-link" href="#Model-Evaluation">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict</span> <span class="o">=</span> <span class="n">classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[46]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[48]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_predict</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[49]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span> <span class="n">fmt</span> <span class="o">=</span> <span class="s1">&#39;d&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[49]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a1a2c24e0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[50]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_predict</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>             precision    recall  f1-score   support

          0       0.85      0.89      0.87       117
          1       0.77      0.69      0.73        62

avg / total       0.82      0.82      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[51]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Simple Score output</span>
<span class="n">classifier</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[51]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.82122905027932958</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Shap-Values">Shap Values<a class="anchor-link" href="#Shap-Values">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[52]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">shap</span>
<span class="c1"># print the JS visualization code to the notebook</span>
<span class="n">shap</span><span class="o">.</span><span class="n">initjs</span><span class="p">()</span>

<span class="c1"># explain all the predictions in the test set</span>
<span class="n">explainer</span> <span class="o">=</span> <span class="n">shap</span><span class="o">.</span><span class="n">KernelExplainer</span><span class="p">(</span><span class="n">classifier</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">,</span> <span class="n">X_train</span><span class="p">)</span>
<span class="n">shap_values</span> <span class="o">=</span> <span class="n">explainer</span><span class="o">.</span><span class="n">shap_values</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">shap</span><span class="o">.</span><span class="n">force_plot</span><span class="p">(</span><span class="n">explainer</span><span class="o">.</span><span class="n">expected_value</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">shap_values</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<div align='center'><img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC' /></div><script>!function(t){function e(r){if(n[r])return n[r].exports;var i=n[r]={i:r,l:!1,exports:{}};return t[r].call(i.exports,i,i.exports,e),i.l=!0,i.exports}var n={};e.m=t,e.c=n,e.i=function(t){return t},e.d=function(t,n,r){e.o(t,n)||Object.defineProperty(t,n,{configurable:!1,enumerable:!0,get:r})},e.n=function(t){var n=t&&t.__esModule?function(){return t.default}:function(){return t};return e.d(n,"a",n),n},e.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},e.p="",e(e.s=189)}([function(t,e,n){"use strict";function r(t,e,n,r,o,a,u,c){if(i(e),!t){var s;if(void 0===e)s=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var l=[n,r,o,a,u,c],f=0;s=new Error(e.replace(/%s/g,function(){return l[f++]})),s.name="Invariant Violation"}throw s.framesToPop=1,s}}var i=function(t){};t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";var r=n(11),i=r;t.exports=i},function(t,e,n){"use strict";function r(t){if(null===t||void 0===t)throw new TypeError("Object.assign cannot be called with null or undefined");return Object(t)}/*
object-assign
(c) Sindre Sorhus
@license MIT
*/
var i=Object.getOwnPropertySymbols,o=Object.prototype.hasOwnProperty,a=Object.prototype.propertyIsEnumerable;t.exports=function(){try{if(!Object.assign)return!1;var t=new String("abc");if(t[5]="de","5"===Object.getOwnPropertyNames(t)[0])return!1;for(var e={},n=0;n<10;n++)e["_"+String.fromCharCode(n)]=n;if("0123456789"!==Object.getOwnPropertyNames(e).map(function(t){return e[t]}).join(""))return!1;var r={};return"abcdefghijklmnopqrst".split("").forEach(function(t){r[t]=t}),"abcdefghijklmnopqrst"===Object.keys(Object.assign({},r)).join("")}catch(t){return!1}}()?Object.assign:function(t,e){for(var n,u,c=r(t),s=1;s<arguments.length;s++){n=Object(arguments[s]);for(var l in n)o.call(n,l)&&(c[l]=n[l]);if(i){u=i(n);for(var f=0;f<u.length;f++)a.call(n,u[f])&&(c[u[f]]=n[u[f]])}}return c}},function(t,e,n){"use strict";function r(t,e){return 1===t.nodeType&&t.getAttribute(d)===String(e)||8===t.nodeType&&t.nodeValue===" react-text: "+e+" "||8===t.nodeType&&t.nodeValue===" react-empty: "+e+" "}function i(t){for(var e;e=t._renderedComponent;)t=e;return t}function o(t,e){var n=i(t);n._hostNode=e,e[g]=n}function a(t){var e=t._hostNode;e&&(delete e[g],t._hostNode=null)}function u(t,e){if(!(t._flags&v.hasCachedChildNodes)){var n=t._renderedChildren,a=e.firstChild;t:for(var u in n)if(n.hasOwnProperty(u)){var c=n[u],s=i(c)._domID;if(0!==s){for(;null!==a;a=a.nextSibling)if(r(a,s)){o(c,a);continue t}f("32",s)}}t._flags|=v.hasCachedChildNodes}}function c(t){if(t[g])return t[g];for(var e=[];!t[g];){if(e.push(t),!t.parentNode)return null;t=t.parentNode}for(var n,r;t&&(r=t[g]);t=e.pop())n=r,e.length&&u(r,t);return n}function s(t){var e=c(t);return null!=e&&e._hostNode===t?e:null}function l(t){if(void 0===t._hostNode&&f("33"),t._hostNode)return t._hostNode;for(var e=[];!t._hostNode;)e.push(t),t._hostParent||f("34"),t=t._hostParent;for(;e.length;t=e.pop())u(t,t._hostNode);return t._hostNode}var f=n(1),p=n(21),h=n(161),d=(n(0),p.ID_ATTRIBUTE_NAME),v=h,g="__reactInternalInstance$"+Math.random().toString(36).slice(2),m={getClosestInstanceFromNode:c,getInstanceFromNode:s,getNodeFromInstance:l,precacheChildNodes:u,precacheNode:o,uncacheNode:a};t.exports=m},function(t,e,n){"use strict";function r(t,e,n,a){function u(e){return t(e=new Date(+e)),e}return u.floor=u,u.ceil=function(n){return t(n=new Date(n-1)),e(n,1),t(n),n},u.round=function(t){var e=u(t),n=u.ceil(t);return t-e<n-t?e:n},u.offset=function(t,n){return e(t=new Date(+t),null==n?1:Math.floor(n)),t},u.range=function(n,r,i){var o,a=[];if(n=u.ceil(n),i=null==i?1:Math.floor(i),!(n<r&&i>0))return a;do{a.push(o=new Date(+n)),e(n,i),t(n)}while(o<n&&n<r);return a},u.filter=function(n){return r(function(e){if(e>=e)for(;t(e),!n(e);)e.setTime(e-1)},function(t,r){if(t>=t)if(r<0)for(;++r<=0;)for(;e(t,-1),!n(t););else for(;--r>=0;)for(;e(t,1),!n(t););})},n&&(u.count=function(e,r){return i.setTime(+e),o.setTime(+r),t(i),t(o),Math.floor(n(i,o))},u.every=function(t){return t=Math.floor(t),isFinite(t)&&t>0?t>1?u.filter(a?function(e){return a(e)%t==0}:function(e){return u.count(0,e)%t==0}):u:null}),u}e.a=r;var i=new Date,o=new Date},function(t,e,n){"use strict";var r=!("undefined"==typeof window||!window.document||!window.document.createElement),i={canUseDOM:r,canUseWorkers:"undefined"!=typeof Worker,canUseEventListeners:r&&!(!window.addEventListener&&!window.attachEvent),canUseViewport:r&&!!window.screen,isInWorker:!r};t.exports=i},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(101);n.d(e,"bisect",function(){return r.a}),n.d(e,"bisectRight",function(){return r.b}),n.d(e,"bisectLeft",function(){return r.c});var i=n(19);n.d(e,"ascending",function(){return i.a});var o=n(102);n.d(e,"bisector",function(){return o.a});var a=n(193);n.d(e,"cross",function(){return a.a});var u=n(194);n.d(e,"descending",function(){return u.a});var c=n(103);n.d(e,"deviation",function(){return c.a});var s=n(104);n.d(e,"extent",function(){return s.a});var l=n(195);n.d(e,"histogram",function(){return l.a});var f=n(205);n.d(e,"thresholdFreedmanDiaconis",function(){return f.a});var p=n(206);n.d(e,"thresholdScott",function(){return p.a});var h=n(108);n.d(e,"thresholdSturges",function(){return h.a});var d=n(197);n.d(e,"max",function(){return d.a});var v=n(198);n.d(e,"mean",function(){return v.a});var g=n(199);n.d(e,"median",function(){return g.a});var m=n(200);n.d(e,"merge",function(){return m.a});var y=n(105);n.d(e,"min",function(){return y.a});var _=n(106);n.d(e,"pairs",function(){return _.a});var b=n(201);n.d(e,"permute",function(){return b.a});var x=n(59);n.d(e,"quantile",function(){return x.a});var w=n(107);n.d(e,"range",function(){return w.a});var C=n(202);n.d(e,"scan",function(){return C.a});var k=n(203);n.d(e,"shuffle",function(){return k.a});var E=n(204);n.d(e,"sum",function(){return E.a});var M=n(109);n.d(e,"ticks",function(){return M.a}),n.d(e,"tickIncrement",function(){return M.b}),n.d(e,"tickStep",function(){return M.c});var T=n(110);n.d(e,"transpose",function(){return T.a});var S=n(111);n.d(e,"variance",function(){return S.a});var N=n(207);n.d(e,"zip",function(){return N.a})},function(t,e,n){"use strict";function r(t,e){this._groups=t,this._parents=e}function i(){return new r([[document.documentElement]],R)}n.d(e,"c",function(){return R}),e.b=r;var o=n(283),a=n(284),u=n(272),c=n(266),s=n(132),l=n(271),f=n(276),p=n(279),h=n(286),d=n(263),v=n(278),g=n(277),m=n(285),y=n(270),_=n(269),b=n(262),x=n(134),w=n(280),C=n(264),k=n(287),E=n(273),M=n(281),T=n(275),S=n(261),N=n(274),A=n(282),P=n(265),O=n(267),I=n(70),D=n(268),R=[null];r.prototype=i.prototype={constructor:r,select:o.a,selectAll:a.a,filter:u.a,data:c.a,enter:s.a,exit:l.a,merge:f.a,order:p.a,sort:h.a,call:d.a,nodes:v.a,node:g.a,size:m.a,empty:y.a,each:_.a,attr:b.a,style:x.b,property:w.a,classed:C.a,text:k.a,html:E.a,raise:M.a,lower:T.a,append:S.a,insert:N.a,remove:A.a,clone:P.a,datum:O.a,on:I.c,dispatch:D.a},e.a=i},function(t,e,n){"use strict";var r=null;t.exports={debugTool:r}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(61);n.d(e,"color",function(){return r.a}),n.d(e,"rgb",function(){return r.b}),n.d(e,"hsl",function(){return r.c});var i=n(218);n.d(e,"lab",function(){return i.a}),n.d(e,"hcl",function(){return i.b});var o=n(217);n.d(e,"cubehelix",function(){return o.a})},function(t,e,n){"use strict";function r(t){return function(){return t}}var i=function(){};i.thatReturns=r,i.thatReturnsFalse=r(!1),i.thatReturnsTrue=r(!0),i.thatReturnsNull=r(null),i.thatReturnsThis=function(){return this},i.thatReturnsArgument=function(t){return t},t.exports=i},function(t,e,n){"use strict";function r(){S.ReactReconcileTransaction&&w||l("123")}function i(){this.reinitializeTransaction(),this.dirtyComponentsLength=null,this.callbackQueue=p.getPooled(),this.reconcileTransaction=S.ReactReconcileTransaction.getPooled(!0)}function o(t,e,n,i,o,a){return r(),w.batchedUpdates(t,e,n,i,o,a)}function a(t,e){return t._mountOrder-e._mountOrder}function u(t){var e=t.dirtyComponentsLength;e!==y.length&&l("124",e,y.length),y.sort(a),_++;for(var n=0;n<e;n++){var r=y[n],i=r._pendingCallbacks;r._pendingCallbacks=null;var o;if(d.logTopLevelRenders){var u=r;r._currentElement.type.isReactTopLevelWrapper&&(u=r._renderedComponent),o="React update: "+u.getName(),console.time(o)}if(v.performUpdateIfNecessary(r,t.reconcileTransaction,_),o&&console.timeEnd(o),i)for(var c=0;c<i.length;c++)t.callbackQueue.enqueue(i[c],r.getPublicInstance())}}function c(t){if(r(),!w.isBatchingUpdates)return void w.batchedUpdates(c,t);y.push(t),null==t._updateBatchNumber&&(t._updateBatchNumber=_+1)}function s(t,e){m(w.isBatchingUpdates,"ReactUpdates.asap: Can't enqueue an asap callback in a context whereupdates are not being batched."),b.enqueue(t,e),x=!0}var l=n(1),f=n(3),p=n(159),h=n(18),d=n(164),v=n(24),g=n(55),m=n(0),y=[],_=0,b=p.getPooled(),x=!1,w=null,C={initialize:function(){this.dirtyComponentsLength=y.length},close:function(){this.dirtyComponentsLength!==y.length?(y.splice(0,this.dirtyComponentsLength),M()):y.length=0}},k={initialize:function(){this.callbackQueue.reset()},close:function(){this.callbackQueue.notifyAll()}},E=[C,k];f(i.prototype,g,{getTransactionWrappers:function(){return E},destructor:function(){this.dirtyComponentsLength=null,p.release(this.callbackQueue),this.callbackQueue=null,S.ReactReconcileTransaction.release(this.reconcileTransaction),this.reconcileTransaction=null},perform:function(t,e,n){return g.perform.call(this,this.reconcileTransaction.perform,this.reconcileTransaction,t,e,n)}}),h.addPoolingTo(i);var M=function(){for(;y.length||x;){if(y.length){var t=i.getPooled();t.perform(u,null,t),i.release(t)}if(x){x=!1;var e=b;b=p.getPooled(),e.notifyAll(),p.release(e)}}},T={injectReconcileTransaction:function(t){t||l("126"),S.ReactReconcileTransaction=t},injectBatchingStrategy:function(t){t||l("127"),"function"!=typeof t.batchedUpdates&&l("128"),"boolean"!=typeof t.isBatchingUpdates&&l("129"),w=t}},S={ReactReconcileTransaction:null,batchedUpdates:o,enqueueUpdate:c,flushBatchedUpdates:M,injection:T,asap:s};t.exports=S},function(t,e,n){"use strict";n.d(e,"e",function(){return r}),n.d(e,"d",function(){return i}),n.d(e,"c",function(){return o}),n.d(e,"b",function(){return a}),n.d(e,"a",function(){return u});var r=1e3,i=6e4,o=36e5,a=864e5,u=6048e5},function(t,e,n){"use strict";function r(t,e,n,r){this.dispatchConfig=t,this._targetInst=e,this.nativeEvent=n;var i=this.constructor.Interface;for(var o in i)if(i.hasOwnProperty(o)){var u=i[o];u?this[o]=u(n):"target"===o?this.target=r:this[o]=n[o]}var c=null!=n.defaultPrevented?n.defaultPrevented:!1===n.returnValue;return this.isDefaultPrevented=c?a.thatReturnsTrue:a.thatReturnsFalse,this.isPropagationStopped=a.thatReturnsFalse,this}var i=n(3),o=n(18),a=n(11),u=(n(2),["dispatchConfig","_targetInst","nativeEvent","isDefaultPrevented","isPropagationStopped","_dispatchListeners","_dispatchInstances"]),c={type:null,target:null,currentTarget:a.thatReturnsNull,eventPhase:null,bubbles:null,cancelable:null,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:null,isTrusted:null};i(r.prototype,{preventDefault:function(){this.defaultPrevented=!0;var t=this.nativeEvent;t&&(t.preventDefault?t.preventDefault():"unknown"!=typeof t.returnValue&&(t.returnValue=!1),this.isDefaultPrevented=a.thatReturnsTrue)},stopPropagation:function(){var t=this.nativeEvent;t&&(t.stopPropagation?t.stopPropagation():"unknown"!=typeof t.cancelBubble&&(t.cancelBubble=!0),this.isPropagationStopped=a.thatReturnsTrue)},persist:function(){this.isPersistent=a.thatReturnsTrue},isPersistent:a.thatReturnsFalse,destructor:function(){var t=this.constructor.Interface;for(var e in t)this[e]=null;for(var n=0;n<u.length;n++)this[u[n]]=null}}),r.Interface=c,r.augmentClass=function(t,e){var n=this,r=function(){};r.prototype=n.prototype;var a=new r;i(a,t.prototype),t.prototype=a,t.prototype.constructor=t,t.Interface=i({},n.Interface,e),t.augmentClass=n.augmentClass,o.addPoolingTo(t,o.fourArgumentPooler)},o.addPoolingTo(r,o.fourArgumentPooler),t.exports=r},function(t,e,n){"use strict";var r={current:null};t.exports=r},function(t,e,n){"use strict";n.d(e,"a",function(){return i}),n.d(e,"b",function(){return o});var r=Array.prototype,i=r.map,o=r.slice},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(1),i=(n(0),function(t){var e=this;if(e.instancePool.length){var n=e.instancePool.pop();return e.call(n,t),n}return new e(t)}),o=function(t,e){var n=this;if(n.instancePool.length){var r=n.instancePool.pop();return n.call(r,t,e),r}return new n(t,e)},a=function(t,e,n){var r=this;if(r.instancePool.length){var i=r.instancePool.pop();return r.call(i,t,e,n),i}return new r(t,e,n)},u=function(t,e,n,r){var i=this;if(i.instancePool.length){var o=i.instancePool.pop();return i.call(o,t,e,n,r),o}return new i(t,e,n,r)},c=function(t){var e=this;t instanceof e||r("25"),t.destructor(),e.instancePool.length<e.poolSize&&e.instancePool.push(t)},s=i,l=function(t,e){var n=t;return n.instancePool=[],n.getPooled=e||s,n.poolSize||(n.poolSize=10),n.release=c,n},f={addPoolingTo:l,oneArgumentPooler:i,twoArgumentPooler:o,threeArgumentPooler:a,fourArgumentPooler:u};t.exports=f},function(t,e,n){"use strict";e.a=function(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}},function(t,e,n){"use strict";function r(t){if(d){var e=t.node,n=t.children;if(n.length)for(var r=0;r<n.length;r++)v(e,n[r],null);else null!=t.html?f(e,t.html):null!=t.text&&h(e,t.text)}}function i(t,e){t.parentNode.replaceChild(e.node,t),r(e)}function o(t,e){d?t.children.push(e):t.node.appendChild(e.node)}function a(t,e){d?t.html=e:f(t.node,e)}function u(t,e){d?t.text=e:h(t.node,e)}function c(){return this.node.nodeName}function s(t){return{node:t,children:[],html:null,text:null,toString:c}}var l=n(83),f=n(57),p=n(91),h=n(176),d="undefined"!=typeof document&&"number"==typeof document.documentMode||"undefined"!=typeof navigator&&"string"==typeof navigator.userAgent&&/\bEdge\/\d/.test(navigator.userAgent),v=p(function(t,e,n){11===e.node.nodeType||1===e.node.nodeType&&"object"===e.node.nodeName.toLowerCase()&&(null==e.node.namespaceURI||e.node.namespaceURI===l.html)?(r(e),t.insertBefore(e.node,n)):(t.insertBefore(e.node,n),r(e))});s.insertTreeBefore=v,s.replaceChildWithTree=i,s.queueChild=o,s.queueHTML=a,s.queueText=u,t.exports=s},function(t,e,n){"use strict";function r(t,e){return(t&e)===e}var i=n(1),o=(n(0),{MUST_USE_PROPERTY:1,HAS_BOOLEAN_VALUE:4,HAS_NUMERIC_VALUE:8,HAS_POSITIVE_NUMERIC_VALUE:24,HAS_OVERLOADED_BOOLEAN_VALUE:32,injectDOMPropertyConfig:function(t){var e=o,n=t.Properties||{},a=t.DOMAttributeNamespaces||{},c=t.DOMAttributeNames||{},s=t.DOMPropertyNames||{},l=t.DOMMutationMethods||{};t.isCustomAttribute&&u._isCustomAttributeFunctions.push(t.isCustomAttribute);for(var f in n){u.properties.hasOwnProperty(f)&&i("48",f);var p=f.toLowerCase(),h=n[f],d={attributeName:p,attributeNamespace:null,propertyName:f,mutationMethod:null,mustUseProperty:r(h,e.MUST_USE_PROPERTY),hasBooleanValue:r(h,e.HAS_BOOLEAN_VALUE),hasNumericValue:r(h,e.HAS_NUMERIC_VALUE),hasPositiveNumericValue:r(h,e.HAS_POSITIVE_NUMERIC_VALUE),hasOverloadedBooleanValue:r(h,e.HAS_OVERLOADED_BOOLEAN_VALUE)};if(d.hasBooleanValue+d.hasNumericValue+d.hasOverloadedBooleanValue<=1||i("50",f),c.hasOwnProperty(f)){var v=c[f];d.attributeName=v}a.hasOwnProperty(f)&&(d.attributeNamespace=a[f]),s.hasOwnProperty(f)&&(d.propertyName=s[f]),l.hasOwnProperty(f)&&(d.mutationMethod=l[f]),u.properties[f]=d}}}),a=":A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD",u={ID_ATTRIBUTE_NAME:"data-reactid",ROOT_ATTRIBUTE_NAME:"data-reactroot",ATTRIBUTE_NAME_START_CHAR:a,ATTRIBUTE_NAME_CHAR:a+"\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040",properties:{},getPossibleStandardName:null,_isCustomAttributeFunctions:[],isCustomAttribute:function(t){for(var e=0;e<u._isCustomAttributeFunctions.length;e++){if((0,u._isCustomAttributeFunctions[e])(t))return!0}return!1},injection:o};t.exports=u},function(t,e,n){"use strict";function r(t){return"button"===t||"input"===t||"select"===t||"textarea"===t}function i(t,e,n){switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":return!(!n.disabled||!r(e));default:return!1}}var o=n(1),a=n(84),u=n(52),c=n(88),s=n(169),l=n(170),f=(n(0),{}),p=null,h=function(t,e){t&&(u.executeDispatchesInOrder(t,e),t.isPersistent()||t.constructor.release(t))},d=function(t){return h(t,!0)},v=function(t){return h(t,!1)},g=function(t){return"."+t._rootNodeID},m={injection:{injectEventPluginOrder:a.injectEventPluginOrder,injectEventPluginsByName:a.injectEventPluginsByName},putListener:function(t,e,n){"function"!=typeof n&&o("94",e,typeof n);var r=g(t);(f[e]||(f[e]={}))[r]=n;var i=a.registrationNameModules[e];i&&i.didPutListener&&i.didPutListener(t,e,n)},getListener:function(t,e){var n=f[e];if(i(e,t._currentElement.type,t._currentElement.props))return null;var r=g(t);return n&&n[r]},deleteListener:function(t,e){var n=a.registrationNameModules[e];n&&n.willDeleteListener&&n.willDeleteListener(t,e);var r=f[e];if(r){delete r[g(t)]}},deleteAllListeners:function(t){var e=g(t);for(var n in f)if(f.hasOwnProperty(n)&&f[n][e]){var r=a.registrationNameModules[n];r&&r.willDeleteListener&&r.willDeleteListener(t,n),delete f[n][e]}},extractEvents:function(t,e,n,r){for(var i,o=a.plugins,u=0;u<o.length;u++){var c=o[u];if(c){var l=c.extractEvents(t,e,n,r);l&&(i=s(i,l))}}return i},enqueueEvents:function(t){t&&(p=s(p,t))},processEventQueue:function(t){var e=p;p=null,t?l(e,d):l(e,v),p&&o("95"),c.rethrowCaughtError()},__purge:function(){f={}},__getListenerBank:function(){return f}};t.exports=m},function(t,e,n){"use strict";function r(t,e,n){var r=e.dispatchConfig.phasedRegistrationNames[n];return m(t,r)}function i(t,e,n){var i=r(t,n,e);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}function o(t){t&&t.dispatchConfig.phasedRegistrationNames&&d.traverseTwoPhase(t._targetInst,i,t)}function a(t){if(t&&t.dispatchConfig.phasedRegistrationNames){var e=t._targetInst,n=e?d.getParentInstance(e):null;d.traverseTwoPhase(n,i,t)}}function u(t,e,n){if(n&&n.dispatchConfig.registrationName){var r=n.dispatchConfig.registrationName,i=m(t,r);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}}function c(t){t&&t.dispatchConfig.registrationName&&u(t._targetInst,null,t)}function s(t){g(t,o)}function l(t){g(t,a)}function f(t,e,n,r){d.traverseEnterLeave(n,r,u,t,e)}function p(t){g(t,c)}var h=n(22),d=n(52),v=n(169),g=n(170),m=(n(2),h.getListener),y={accumulateTwoPhaseDispatches:s,accumulateTwoPhaseDispatchesSkipTarget:l,accumulateDirectDispatches:p,accumulateEnterLeaveDispatches:f};t.exports=y},function(t,e,n){"use strict";function r(){i.attachRefs(this,this._currentElement)}var i=n(382),o=(n(9),n(2),{mountComponent:function(t,e,n,i,o,a){var u=t.mountComponent(e,n,i,o,a);return t._currentElement&&null!=t._currentElement.ref&&e.getReactMountReady().enqueue(r,t),u},getHostNode:function(t){return t.getHostNode()},unmountComponent:function(t,e){i.detachRefs(t,t._currentElement),t.unmountComponent(e)},receiveComponent:function(t,e,n,o){var a=t._currentElement;if(e!==a||o!==t._context){var u=i.shouldUpdateRefs(a,e);u&&i.detachRefs(t,a),t.receiveComponent(e,n,o),u&&t._currentElement&&null!=t._currentElement.ref&&n.getReactMountReady().enqueue(r,t)}},performUpdateIfNecessary:function(t,e,n){t._updateBatchNumber===n&&t.performUpdateIfNecessary(e)}});t.exports=o},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o=n(94),a={view:function(t){if(t.view)return t.view;var e=o(t);if(e.window===e)return e;var n=e.ownerDocument;return n?n.defaultView||n.parentWindow:window},detail:function(t){return t.detail||0}};i.augmentClass(r,a),t.exports=r},function(t,e,n){"use strict";var r=n(3),i=n(178),o=n(414),a=n(415),u=n(27),c=n(416),s=n(417),l=n(418),f=n(422),p=u.createElement,h=u.createFactory,d=u.cloneElement,v=r,g=function(t){return t},m={Children:{map:o.map,forEach:o.forEach,count:o.count,toArray:o.toArray,only:f},Component:i.Component,PureComponent:i.PureComponent,createElement:p,cloneElement:d,isValidElement:u.isValidElement,PropTypes:c,createClass:l,createFactory:h,createMixin:g,DOM:a,version:s,__spread:v};t.exports=m},function(t,e,n){"use strict";function r(t){return void 0!==t.ref}function i(t){return void 0!==t.key}var o=n(3),a=n(15),u=(n(2),n(182),Object.prototype.hasOwnProperty),c=n(180),s={key:!0,ref:!0,__self:!0,__source:!0},l=function(t,e,n,r,i,o,a){var u={$$typeof:c,type:t,key:e,ref:n,props:a,_owner:o};return u};l.createElement=function(t,e,n){var o,c={},f=null,p=null;if(null!=e){r(e)&&(p=e.ref),i(e)&&(f=""+e.key),void 0===e.__self?null:e.__self,void 0===e.__source?null:e.__source;for(o in e)u.call(e,o)&&!s.hasOwnProperty(o)&&(c[o]=e[o])}var h=arguments.length-2;if(1===h)c.children=n;else if(h>1){for(var d=Array(h),v=0;v<h;v++)d[v]=arguments[v+2];c.children=d}if(t&&t.defaultProps){var g=t.defaultProps;for(o in g)void 0===c[o]&&(c[o]=g[o])}return l(t,f,p,0,0,a.current,c)},l.createFactory=function(t){var e=l.createElement.bind(null,t);return e.type=t,e},l.cloneAndReplaceKey=function(t,e){return l(t.type,e,t.ref,t._self,t._source,t._owner,t.props)},l.cloneElement=function(t,e,n){var c,f=o({},t.props),p=t.key,h=t.ref,d=(t._self,t._source,t._owner);if(null!=e){r(e)&&(h=e.ref,d=a.current),i(e)&&(p=""+e.key);var v;t.type&&t.type.defaultProps&&(v=t.type.defaultProps);for(c in e)u.call(e,c)&&!s.hasOwnProperty(c)&&(void 0===e[c]&&void 0!==v?f[c]=v[c]:f[c]=e[c])}var g=arguments.length-2;if(1===g)f.children=n;else if(g>1){for(var m=Array(g),y=0;y<g;y++)m[y]=arguments[y+2];f.children=m}return l(t.type,p,h,0,0,d,f)},l.isValidElement=function(t){return"object"==typeof t&&null!==t&&t.$$typeof===c},t.exports=l},function(t,e,n){"use strict";e.a=function(t){return null===t?NaN:+t}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(219);n.d(e,"formatDefaultLocale",function(){return r.a}),n.d(e,"format",function(){return r.b}),n.d(e,"formatPrefix",function(){return r.c});var i=n(117);n.d(e,"formatLocale",function(){return i.a});var o=n(115);n.d(e,"formatSpecifier",function(){return o.a});var a=n(225);n.d(e,"precisionFixed",function(){return a.a});var u=n(226);n.d(e,"precisionPrefix",function(){return u.a});var c=n(227);n.d(e,"precisionRound",function(){return c.a})},function(t,e,n){"use strict";var r=n(65);n.d(e,"b",function(){return r.a});var i=(n(118),n(64),n(119),n(121),n(43));n.d(e,"a",function(){return i.a});var o=(n(122),n(233));n.d(e,"c",function(){return o.a});var a=(n(124),n(235),n(237),n(123),n(230),n(231),n(229),n(228));n.d(e,"d",function(){return a.a});n(232)},function(t,e,n){"use strict";function r(t,e){return function(n){return t+n*e}}function i(t,e,n){return t=Math.pow(t,n),e=Math.pow(e,n)-t,n=1/n,function(r){return Math.pow(t+r*e,n)}}function o(t,e){var i=e-t;return i?r(t,i>180||i<-180?i-360*Math.round(i/360):i):n.i(c.a)(isNaN(t)?e:t)}function a(t){return 1==(t=+t)?u:function(e,r){return r-e?i(e,r,t):n.i(c.a)(isNaN(e)?r:e)}}function u(t,e){var i=e-t;return i?r(t,i):n.i(c.a)(isNaN(t)?e:t)}e.b=o,e.c=a,e.a=u;var c=n(120)},function(t,e,n){"use strict";var r=n(238);n.d(e,"a",function(){return r.a})},function(t,e,n){"use strict";e.a=function(t){return t.match(/.{6}/g).map(function(t){return"#"+t})}},function(t,e,n){"use strict";function r(t){var e=t.domain;return t.ticks=function(t){var r=e();return n.i(o.ticks)(r[0],r[r.length-1],null==t?10:t)},t.tickFormat=function(t,r){return n.i(c.a)(e(),t,r)},t.nice=function(r){null==r&&(r=10);var i,a=e(),u=0,c=a.length-1,s=a[u],l=a[c];return l<s&&(i=s,s=l,l=i,i=u,u=c,c=i),i=n.i(o.tickIncrement)(s,l,r),i>0?(s=Math.floor(s/i)*i,l=Math.ceil(l/i)*i,i=n.i(o.tickIncrement)(s,l,r)):i<0&&(s=Math.ceil(s*i)/i,l=Math.floor(l*i)/i,i=n.i(o.tickIncrement)(s,l,r)),i>0?(a[u]=Math.floor(s/i)*i,a[c]=Math.ceil(l/i)*i,e(a)):i<0&&(a[u]=Math.ceil(s*i)/i,a[c]=Math.floor(l*i)/i,e(a)),t},t}function i(){var t=n.i(u.a)(u.b,a.a);return t.copy=function(){return n.i(u.c)(t,i())},r(t)}e.b=r,e.a=i;var o=n(7),a=n(30),u=n(44),c=n(253)},function(t,e,n){"use strict";function r(t){return t>1?0:t<-1?h:Math.acos(t)}function i(t){return t>=1?d:t<=-1?-d:Math.asin(t)}n.d(e,"g",function(){return o}),n.d(e,"m",function(){return a}),n.d(e,"h",function(){return u}),n.d(e,"e",function(){return c}),n.d(e,"j",function(){return s}),n.d(e,"i",function(){return l}),n.d(e,"d",function(){return f}),n.d(e,"a",function(){return p}),n.d(e,"b",function(){return h}),n.d(e,"f",function(){return d}),n.d(e,"c",function(){return v}),e.l=r,e.k=i;var o=Math.abs,a=Math.atan2,u=Math.cos,c=Math.max,s=Math.min,l=Math.sin,f=Math.sqrt,p=1e-12,h=Math.PI,d=h/2,v=2*h},function(t,e,n){"use strict";e.a=function(t,e){if((i=t.length)>1)for(var n,r,i,o=1,a=t[e[0]],u=a.length;o<i;++o)for(r=a,a=t[e[o]],n=0;n<u;++n)a[n][1]+=a[n][0]=isNaN(r[n][1])?r[n][0]:r[n][1]}},function(t,e,n){"use strict";e.a=function(t){for(var e=t.length,n=new Array(e);--e>=0;)n[e]=e;return n}},function(t,e,n){(function(t,r){var i;(function(){function o(t,e,n){switch(n.length){case 0:return t.call(e);case 1:return t.call(e,n[0]);case 2:return t.call(e,n[0],n[1]);case 3:return t.call(e,n[0],n[1],n[2])}return t.apply(e,n)}function a(t,e,n,r){for(var i=-1,o=null==t?0:t.length;++i<o;){var a=t[i];e(r,a,n(a),t)}return r}function u(t,e){for(var n=-1,r=null==t?0:t.length;++n<r&&!1!==e(t[n],n,t););return t}function c(t,e){for(var n=null==t?0:t.length;n--&&!1!==e(t[n],n,t););return t}function s(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(!e(t[n],n,t))return!1;return!0}function l(t,e){for(var n=-1,r=null==t?0:t.length,i=0,o=[];++n<r;){var a=t[n];e(a,n,t)&&(o[i++]=a)}return o}function f(t,e){return!!(null==t?0:t.length)&&w(t,e,0)>-1}function p(t,e,n){for(var r=-1,i=null==t?0:t.length;++r<i;)if(n(e,t[r]))return!0;return!1}function h(t,e){for(var n=-1,r=null==t?0:t.length,i=Array(r);++n<r;)i[n]=e(t[n],n,t);return i}function d(t,e){for(var n=-1,r=e.length,i=t.length;++n<r;)t[i+n]=e[n];return t}function v(t,e,n,r){var i=-1,o=null==t?0:t.length;for(r&&o&&(n=t[++i]);++i<o;)n=e(n,t[i],i,t);return n}function g(t,e,n,r){var i=null==t?0:t.length;for(r&&i&&(n=t[--i]);i--;)n=e(n,t[i],i,t);return n}function m(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(e(t[n],n,t))return!0;return!1}function y(t){return t.split("")}function _(t){return t.match(Ue)||[]}function b(t,e,n){var r;return n(t,function(t,n,i){if(e(t,n,i))return r=n,!1}),r}function x(t,e,n,r){for(var i=t.length,o=n+(r?1:-1);r?o--:++o<i;)if(e(t[o],o,t))return o;return-1}function w(t,e,n){return e===e?$(t,e,n):x(t,k,n)}function C(t,e,n,r){for(var i=n-1,o=t.length;++i<o;)if(r(t[i],e))return i;return-1}function k(t){return t!==t}function E(t,e){var n=null==t?0:t.length;return n?A(t,e)/n:It}function M(t){return function(e){return null==e?nt:e[t]}}function T(t){return function(e){return null==t?nt:t[e]}}function S(t,e,n,r,i){return i(t,function(t,i,o){n=r?(r=!1,t):e(n,t,i,o)}),n}function N(t,e){var n=t.length;for(t.sort(e);n--;)t[n]=t[n].value;return t}function A(t,e){for(var n,r=-1,i=t.length;++r<i;){var o=e(t[r]);o!==nt&&(n=n===nt?o:n+o)}return n}function P(t,e){for(var n=-1,r=Array(t);++n<t;)r[n]=e(n);return r}function O(t,e){return h(e,function(e){return[e,t[e]]})}function I(t){return function(e){return t(e)}}function D(t,e){return h(e,function(e){return t[e]})}function R(t,e){return t.has(e)}function L(t,e){for(var n=-1,r=t.length;++n<r&&w(e,t[n],0)>-1;);return n}function U(t,e){for(var n=t.length;n--&&w(e,t[n],0)>-1;);return n}function F(t,e){for(var n=t.length,r=0;n--;)t[n]===e&&++r;return r}function j(t){return"\\"+En[t]}function B(t,e){return null==t?nt:t[e]}function V(t){return gn.test(t)}function W(t){return mn.test(t)}function z(t){for(var e,n=[];!(e=t.next()).done;)n.push(e.value);return n}function H(t){var e=-1,n=Array(t.size);return t.forEach(function(t,r){n[++e]=[r,t]}),n}function q(t,e){return function(n){return t(e(n))}}function Y(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n];a!==e&&a!==ct||(t[n]=ct,o[i++]=n)}return o}function K(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=t}),n}function G(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=[t,t]}),n}function $(t,e,n){for(var r=n-1,i=t.length;++r<i;)if(t[r]===e)return r;return-1}function X(t,e,n){for(var r=n+1;r--;)if(t[r]===e)return r;return r}function Q(t){return V(t)?J(t):Wn(t)}function Z(t){return V(t)?tt(t):y(t)}function J(t){for(var e=dn.lastIndex=0;dn.test(t);)++e;return e}function tt(t){return t.match(dn)||[]}function et(t){return t.match(vn)||[]}var nt,rt=200,it="Unsupported core-js use. Try https://npms.io/search?q=ponyfill.",ot="Expected a function",at="__lodash_hash_undefined__",ut=500,ct="__lodash_placeholder__",st=1,lt=2,ft=4,pt=1,ht=2,dt=1,vt=2,gt=4,mt=8,yt=16,_t=32,bt=64,xt=128,wt=256,Ct=512,kt=30,Et="...",Mt=800,Tt=16,St=1,Nt=2,At=1/0,Pt=9007199254740991,Ot=1.7976931348623157e308,It=NaN,Dt=4294967295,Rt=Dt-1,Lt=Dt>>>1,Ut=[["ary",xt],["bind",dt],["bindKey",vt],["curry",mt],["curryRight",yt],["flip",Ct],["partial",_t],["partialRight",bt],["rearg",wt]],Ft="[object Arguments]",jt="[object Array]",Bt="[object AsyncFunction]",Vt="[object Boolean]",Wt="[object Date]",zt="[object DOMException]",Ht="[object Error]",qt="[object Function]",Yt="[object GeneratorFunction]",Kt="[object Map]",Gt="[object Number]",$t="[object Null]",Xt="[object Object]",Qt="[object Proxy]",Zt="[object RegExp]",Jt="[object Set]",te="[object String]",ee="[object Symbol]",ne="[object Undefined]",re="[object WeakMap]",ie="[object WeakSet]",oe="[object ArrayBuffer]",ae="[object DataView]",ue="[object Float32Array]",ce="[object Float64Array]",se="[object Int8Array]",le="[object Int16Array]",fe="[object Int32Array]",pe="[object Uint8Array]",he="[object Uint8ClampedArray]",de="[object Uint16Array]",ve="[object Uint32Array]",ge=/\b__p \+= '';/g,me=/\b(__p \+=) '' \+/g,ye=/(__e\(.*?\)|\b__t\)) \+\n'';/g,_e=/&(?:amp|lt|gt|quot|#39);/g,be=/[&<>"']/g,xe=RegExp(_e.source),we=RegExp(be.source),Ce=/<%-([\s\S]+?)%>/g,ke=/<%([\s\S]+?)%>/g,Ee=/<%=([\s\S]+?)%>/g,Me=/\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,Te=/^\w*$/,Se=/[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g,Ne=/[\\^$.*+?()[\]{}|]/g,Ae=RegExp(Ne.source),Pe=/^\s+|\s+$/g,Oe=/^\s+/,Ie=/\s+$/,De=/\{(?:\n\/\* \[wrapped with .+\] \*\/)?\n?/,Re=/\{\n\/\* \[wrapped with (.+)\] \*/,Le=/,? & /,Ue=/[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g,Fe=/\\(\\)?/g,je=/\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g,Be=/\w*$/,Ve=/^[-+]0x[0-9a-f]+$/i,We=/^0b[01]+$/i,ze=/^\[object .+?Constructor\]$/,He=/^0o[0-7]+$/i,qe=/^(?:0|[1-9]\d*)$/,Ye=/[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g,Ke=/($^)/,Ge=/['\n\r\u2028\u2029\\]/g,$e="\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff",Xe="\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000",Qe="["+Xe+"]",Ze="["+$e+"]",Je="[a-z\\xdf-\\xf6\\xf8-\\xff]",tn="[^\\ud800-\\udfff"+Xe+"\\d+\\u2700-\\u27bfa-z\\xdf-\\xf6\\xf8-\\xffA-Z\\xc0-\\xd6\\xd8-\\xde]",en="\\ud83c[\\udffb-\\udfff]",nn="(?:\\ud83c[\\udde6-\\uddff]){2}",rn="[\\ud800-\\udbff][\\udc00-\\udfff]",on="[A-Z\\xc0-\\xd6\\xd8-\\xde]",an="(?:"+Je+"|"+tn+")",un="(?:[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]|\\ud83c[\\udffb-\\udfff])?",cn="(?:\\u200d(?:"+["[^\\ud800-\\udfff]",nn,rn].join("|")+")[\\ufe0e\\ufe0f]?"+un+")*",sn="[\\ufe0e\\ufe0f]?"+un+cn,ln="(?:"+["[\\u2700-\\u27bf]",nn,rn].join("|")+")"+sn,fn="(?:"+["[^\\ud800-\\udfff]"+Ze+"?",Ze,nn,rn,"[\\ud800-\\udfff]"].join("|")+")",pn=RegExp("['’]","g"),hn=RegExp(Ze,"g"),dn=RegExp(en+"(?="+en+")|"+fn+sn,"g"),vn=RegExp([on+"?"+Je+"+(?:['’](?:d|ll|m|re|s|t|ve))?(?="+[Qe,on,"$"].join("|")+")","(?:[A-Z\\xc0-\\xd6\\xd8-\\xde]|[^\\ud800-\\udfff\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000\\d+\\u2700-\\u27bfa-z\\xdf-\\xf6\\xf8-\\xffA-Z\\xc0-\\xd6\\xd8-\\xde])+(?:['’](?:D|LL|M|RE|S|T|VE))?(?="+[Qe,on+an,"$"].join("|")+")",on+"?"+an+"+(?:['’](?:d|ll|m|re|s|t|ve))?",on+"+(?:['’](?:D|LL|M|RE|S|T|VE))?","\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])","\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])","\\d+",ln].join("|"),"g"),gn=RegExp("[\\u200d\\ud800-\\udfff"+$e+"\\ufe0e\\ufe0f]"),mn=/[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/,yn=["Array","Buffer","DataView","Date","Error","Float32Array","Float64Array","Function","Int8Array","Int16Array","Int32Array","Map","Math","Object","Promise","RegExp","Set","String","Symbol","TypeError","Uint8Array","Uint8ClampedArray","Uint16Array","Uint32Array","WeakMap","_","clearTimeout","isFinite","parseInt","setTimeout"],_n=-1,bn={};bn[ue]=bn[ce]=bn[se]=bn[le]=bn[fe]=bn[pe]=bn[he]=bn[de]=bn[ve]=!0,bn[Ft]=bn[jt]=bn[oe]=bn[Vt]=bn[ae]=bn[Wt]=bn[Ht]=bn[qt]=bn[Kt]=bn[Gt]=bn[Xt]=bn[Zt]=bn[Jt]=bn[te]=bn[re]=!1;var xn={};xn[Ft]=xn[jt]=xn[oe]=xn[ae]=xn[Vt]=xn[Wt]=xn[ue]=xn[ce]=xn[se]=xn[le]=xn[fe]=xn[Kt]=xn[Gt]=xn[Xt]=xn[Zt]=xn[Jt]=xn[te]=xn[ee]=xn[pe]=xn[he]=xn[de]=xn[ve]=!0,xn[Ht]=xn[qt]=xn[re]=!1;var wn={"À":"A","Á":"A","Â":"A","Ã":"A","Ä":"A","Å":"A","à":"a","á":"a","â":"a","ã":"a","ä":"a","å":"a","Ç":"C","ç":"c","Ð":"D","ð":"d","È":"E","É":"E","Ê":"E","Ë":"E","è":"e","é":"e","ê":"e","ë":"e","Ì":"I","Í":"I","Î":"I","Ï":"I","ì":"i","í":"i","î":"i","ï":"i","Ñ":"N","ñ":"n","Ò":"O","Ó":"O","Ô":"O","Õ":"O","Ö":"O","Ø":"O","ò":"o","ó":"o","ô":"o","õ":"o","ö":"o","ø":"o","Ù":"U","Ú":"U","Û":"U","Ü":"U","ù":"u","ú":"u","û":"u","ü":"u","Ý":"Y","ý":"y","ÿ":"y","Æ":"Ae","æ":"ae","Þ":"Th","þ":"th","ß":"ss","Ā":"A","Ă":"A","Ą":"A","ā":"a","ă":"a","ą":"a","Ć":"C","Ĉ":"C","Ċ":"C","Č":"C","ć":"c","ĉ":"c","ċ":"c","č":"c","Ď":"D","Đ":"D","ď":"d","đ":"d","Ē":"E","Ĕ":"E","Ė":"E","Ę":"E","Ě":"E","ē":"e","ĕ":"e","ė":"e","ę":"e","ě":"e","Ĝ":"G","Ğ":"G","Ġ":"G","Ģ":"G","ĝ":"g","ğ":"g","ġ":"g","ģ":"g","Ĥ":"H","Ħ":"H","ĥ":"h","ħ":"h","Ĩ":"I","Ī":"I","Ĭ":"I","Į":"I","İ":"I","ĩ":"i","ī":"i","ĭ":"i","į":"i","ı":"i","Ĵ":"J","ĵ":"j","Ķ":"K","ķ":"k","ĸ":"k","Ĺ":"L","Ļ":"L","Ľ":"L","Ŀ":"L","Ł":"L","ĺ":"l","ļ":"l","ľ":"l","ŀ":"l","ł":"l","Ń":"N","Ņ":"N","Ň":"N","Ŋ":"N","ń":"n","ņ":"n","ň":"n","ŋ":"n","Ō":"O","Ŏ":"O","Ő":"O","ō":"o","ŏ":"o","ő":"o","Ŕ":"R","Ŗ":"R","Ř":"R","ŕ":"r","ŗ":"r","ř":"r","Ś":"S","Ŝ":"S","Ş":"S","Š":"S","ś":"s","ŝ":"s","ş":"s","š":"s","Ţ":"T","Ť":"T","Ŧ":"T","ţ":"t","ť":"t","ŧ":"t","Ũ":"U","Ū":"U","Ŭ":"U","Ů":"U","Ű":"U","Ų":"U","ũ":"u","ū":"u","ŭ":"u","ů":"u","ű":"u","ų":"u","Ŵ":"W","ŵ":"w","Ŷ":"Y","ŷ":"y","Ÿ":"Y","Ź":"Z","Ż":"Z","Ž":"Z","ź":"z","ż":"z","ž":"z","Ĳ":"IJ","ĳ":"ij","Œ":"Oe","œ":"oe","ŉ":"'n","ſ":"s"},Cn={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},kn={"&amp;":"&","&lt;":"<","&gt;":">","&quot;":'"',"&#39;":"'"},En={"\\":"\\","'":"'","\n":"n","\r":"r","\u2028":"u2028","\u2029":"u2029"},Mn=parseFloat,Tn=parseInt,Sn="object"==typeof t&&t&&t.Object===Object&&t,Nn="object"==typeof self&&self&&self.Object===Object&&self,An=Sn||Nn||Function("return this")(),Pn="object"==typeof e&&e&&!e.nodeType&&e,On=Pn&&"object"==typeof r&&r&&!r.nodeType&&r,In=On&&On.exports===Pn,Dn=In&&Sn.process,Rn=function(){try{var t=On&&On.require&&On.require("util").types;return t||Dn&&Dn.binding&&Dn.binding("util")}catch(t){}}(),Ln=Rn&&Rn.isArrayBuffer,Un=Rn&&Rn.isDate,Fn=Rn&&Rn.isMap,jn=Rn&&Rn.isRegExp,Bn=Rn&&Rn.isSet,Vn=Rn&&Rn.isTypedArray,Wn=M("length"),zn=T(wn),Hn=T(Cn),qn=T(kn),Yn=function t(e){function n(t){if(ec(t)&&!hp(t)&&!(t instanceof y)){if(t instanceof i)return t;if(pl.call(t,"__wrapped__"))return Zo(t)}return new i(t)}function r(){}function i(t,e){this.__wrapped__=t,this.__actions__=[],this.__chain__=!!e,this.__index__=0,this.__values__=nt}function y(t){this.__wrapped__=t,this.__actions__=[],this.__dir__=1,this.__filtered__=!1,this.__iteratees__=[],this.__takeCount__=Dt,this.__views__=[]}function T(){var t=new y(this.__wrapped__);return t.__actions__=Oi(this.__actions__),t.__dir__=this.__dir__,t.__filtered__=this.__filtered__,t.__iteratees__=Oi(this.__iteratees__),t.__takeCount__=this.__takeCount__,t.__views__=Oi(this.__views__),t}function $(){if(this.__filtered__){var t=new y(this);t.__dir__=-1,t.__filtered__=!0}else t=this.clone(),t.__dir__*=-1;return t}function J(){var t=this.__wrapped__.value(),e=this.__dir__,n=hp(t),r=e<0,i=n?t.length:0,o=wo(0,i,this.__views__),a=o.start,u=o.end,c=u-a,s=r?u:a-1,l=this.__iteratees__,f=l.length,p=0,h=Wl(c,this.__takeCount__);if(!n||!r&&i==c&&h==c)return vi(t,this.__actions__);var d=[];t:for(;c--&&p<h;){s+=e;for(var v=-1,g=t[s];++v<f;){var m=l[v],y=m.iteratee,_=m.type,b=y(g);if(_==Nt)g=b;else if(!b){if(_==St)continue t;break t}}d[p++]=g}return d}function tt(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function Ue(){this.__data__=Zl?Zl(null):{},this.size=0}function $e(t){var e=this.has(t)&&delete this.__data__[t];return this.size-=e?1:0,e}function Xe(t){var e=this.__data__;if(Zl){var n=e[t];return n===at?nt:n}return pl.call(e,t)?e[t]:nt}function Qe(t){var e=this.__data__;return Zl?e[t]!==nt:pl.call(e,t)}function Ze(t,e){var n=this.__data__;return this.size+=this.has(t)?0:1,n[t]=Zl&&e===nt?at:e,this}function Je(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function tn(){this.__data__=[],this.size=0}function en(t){var e=this.__data__,n=Kn(e,t);return!(n<0)&&(n==e.length-1?e.pop():Ml.call(e,n,1),--this.size,!0)}function nn(t){var e=this.__data__,n=Kn(e,t);return n<0?nt:e[n][1]}function rn(t){return Kn(this.__data__,t)>-1}function on(t,e){var n=this.__data__,r=Kn(n,t);return r<0?(++this.size,n.push([t,e])):n[r][1]=e,this}function an(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function un(){this.size=0,this.__data__={hash:new tt,map:new(Gl||Je),string:new tt}}function cn(t){var e=yo(this,t).delete(t);return this.size-=e?1:0,e}function sn(t){return yo(this,t).get(t)}function ln(t){return yo(this,t).has(t)}function fn(t,e){var n=yo(this,t),r=n.size;return n.set(t,e),this.size+=n.size==r?0:1,this}function dn(t){var e=-1,n=null==t?0:t.length;for(this.__data__=new an;++e<n;)this.add(t[e])}function vn(t){return this.__data__.set(t,at),this}function gn(t){return this.__data__.has(t)}function mn(t){var e=this.__data__=new Je(t);this.size=e.size}function wn(){this.__data__=new Je,this.size=0}function Cn(t){var e=this.__data__,n=e.delete(t);return this.size=e.size,n}function kn(t){return this.__data__.get(t)}function En(t){return this.__data__.has(t)}function Sn(t,e){var n=this.__data__;if(n instanceof Je){var r=n.__data__;if(!Gl||r.length<rt-1)return r.push([t,e]),this.size=++n.size,this;n=this.__data__=new an(r)}return n.set(t,e),this.size=n.size,this}function Nn(t,e){var n=hp(t),r=!n&&pp(t),i=!n&&!r&&vp(t),o=!n&&!r&&!i&&bp(t),a=n||r||i||o,u=a?P(t.length,ol):[],c=u.length;for(var s in t)!e&&!pl.call(t,s)||a&&("length"==s||i&&("offset"==s||"parent"==s)||o&&("buffer"==s||"byteLength"==s||"byteOffset"==s)||Ao(s,c))||u.push(s);return u}function Pn(t){var e=t.length;return e?t[Xr(0,e-1)]:nt}function On(t,e){return Go(Oi(t),Jn(e,0,t.length))}function Dn(t){return Go(Oi(t))}function Rn(t,e,n){(n===nt||Vu(t[e],n))&&(n!==nt||e in t)||Qn(t,e,n)}function Wn(t,e,n){var r=t[e];pl.call(t,e)&&Vu(r,n)&&(n!==nt||e in t)||Qn(t,e,n)}function Kn(t,e){for(var n=t.length;n--;)if(Vu(t[n][0],e))return n;return-1}function Gn(t,e,n,r){return ff(t,function(t,i,o){e(r,t,n(t),o)}),r}function $n(t,e){return t&&Ii(e,Lc(e),t)}function Xn(t,e){return t&&Ii(e,Uc(e),t)}function Qn(t,e,n){"__proto__"==e&&Al?Al(t,e,{configurable:!0,enumerable:!0,value:n,writable:!0}):t[e]=n}function Zn(t,e){for(var n=-1,r=e.length,i=Zs(r),o=null==t;++n<r;)i[n]=o?nt:Ic(t,e[n]);return i}function Jn(t,e,n){return t===t&&(n!==nt&&(t=t<=n?t:n),e!==nt&&(t=t>=e?t:e)),t}function tr(t,e,n,r,i,o){var a,c=e&st,s=e&lt,l=e&ft;if(n&&(a=i?n(t,r,i,o):n(t)),a!==nt)return a;if(!tc(t))return t;var f=hp(t);if(f){if(a=Eo(t),!c)return Oi(t,a)}else{var p=Cf(t),h=p==qt||p==Yt;if(vp(t))return wi(t,c);if(p==Xt||p==Ft||h&&!i){if(a=s||h?{}:Mo(t),!c)return s?Ri(t,Xn(a,t)):Di(t,$n(a,t))}else{if(!xn[p])return i?t:{};a=To(t,p,c)}}o||(o=new mn);var d=o.get(t);if(d)return d;if(o.set(t,a),_p(t))return t.forEach(function(r){a.add(tr(r,e,n,r,t,o))}),a;if(mp(t))return t.forEach(function(r,i){a.set(i,tr(r,e,n,i,t,o))}),a;var v=l?s?ho:po:s?Uc:Lc,g=f?nt:v(t);return u(g||t,function(r,i){g&&(i=r,r=t[i]),Wn(a,i,tr(r,e,n,i,t,o))}),a}function er(t){var e=Lc(t);return function(n){return nr(n,t,e)}}function nr(t,e,n){var r=n.length;if(null==t)return!r;for(t=rl(t);r--;){var i=n[r],o=e[i],a=t[i];if(a===nt&&!(i in t)||!o(a))return!1}return!0}function rr(t,e,n){if("function"!=typeof t)throw new al(ot);return Mf(function(){t.apply(nt,n)},e)}function ir(t,e,n,r){var i=-1,o=f,a=!0,u=t.length,c=[],s=e.length;if(!u)return c;n&&(e=h(e,I(n))),r?(o=p,a=!1):e.length>=rt&&(o=R,a=!1,e=new dn(e));t:for(;++i<u;){var l=t[i],d=null==n?l:n(l);if(l=r||0!==l?l:0,a&&d===d){for(var v=s;v--;)if(e[v]===d)continue t;c.push(l)}else o(e,d,r)||c.push(l)}return c}function or(t,e){var n=!0;return ff(t,function(t,r,i){return n=!!e(t,r,i)}),n}function ar(t,e,n){for(var r=-1,i=t.length;++r<i;){var o=t[r],a=e(o);if(null!=a&&(u===nt?a===a&&!pc(a):n(a,u)))var u=a,c=o}return c}function ur(t,e,n,r){var i=t.length;for(n=yc(n),n<0&&(n=-n>i?0:i+n),r=r===nt||r>i?i:yc(r),r<0&&(r+=i),r=n>r?0:_c(r);n<r;)t[n++]=e;return t}function cr(t,e){var n=[];return ff(t,function(t,r,i){e(t,r,i)&&n.push(t)}),n}function sr(t,e,n,r,i){var o=-1,a=t.length;for(n||(n=No),i||(i=[]);++o<a;){var u=t[o];e>0&&n(u)?e>1?sr(u,e-1,n,r,i):d(i,u):r||(i[i.length]=u)}return i}function lr(t,e){return t&&hf(t,e,Lc)}function fr(t,e){return t&&df(t,e,Lc)}function pr(t,e){return l(e,function(e){return Qu(t[e])})}function hr(t,e){e=bi(e,t);for(var n=0,r=e.length;null!=t&&n<r;)t=t[$o(e[n++])];return n&&n==r?t:nt}function dr(t,e,n){var r=e(t);return hp(t)?r:d(r,n(t))}function vr(t){return null==t?t===nt?ne:$t:Nl&&Nl in rl(t)?xo(t):Vo(t)}function gr(t,e){return t>e}function mr(t,e){return null!=t&&pl.call(t,e)}function yr(t,e){return null!=t&&e in rl(t)}function _r(t,e,n){return t>=Wl(e,n)&&t<Vl(e,n)}function br(t,e,n){for(var r=n?p:f,i=t[0].length,o=t.length,a=o,u=Zs(o),c=1/0,s=[];a--;){var l=t[a];a&&e&&(l=h(l,I(e))),c=Wl(l.length,c),u[a]=!n&&(e||i>=120&&l.length>=120)?new dn(a&&l):nt}l=t[0];var d=-1,v=u[0];t:for(;++d<i&&s.length<c;){var g=l[d],m=e?e(g):g;if(g=n||0!==g?g:0,!(v?R(v,m):r(s,m,n))){for(a=o;--a;){var y=u[a];if(!(y?R(y,m):r(t[a],m,n)))continue t}v&&v.push(m),s.push(g)}}return s}function xr(t,e,n,r){return lr(t,function(t,i,o){e(r,n(t),i,o)}),r}function wr(t,e,n){e=bi(e,t),t=zo(t,e);var r=null==t?t:t[$o(ma(e))];return null==r?nt:o(r,t,n)}function Cr(t){return ec(t)&&vr(t)==Ft}function kr(t){return ec(t)&&vr(t)==oe}function Er(t){return ec(t)&&vr(t)==Wt}function Mr(t,e,n,r,i){return t===e||(null==t||null==e||!ec(t)&&!ec(e)?t!==t&&e!==e:Tr(t,e,n,r,Mr,i))}function Tr(t,e,n,r,i,o){var a=hp(t),u=hp(e),c=a?jt:Cf(t),s=u?jt:Cf(e);c=c==Ft?Xt:c,s=s==Ft?Xt:s;var l=c==Xt,f=s==Xt,p=c==s;if(p&&vp(t)){if(!vp(e))return!1;a=!0,l=!1}if(p&&!l)return o||(o=new mn),a||bp(t)?co(t,e,n,r,i,o):so(t,e,c,n,r,i,o);if(!(n&pt)){var h=l&&pl.call(t,"__wrapped__"),d=f&&pl.call(e,"__wrapped__");if(h||d){var v=h?t.value():t,g=d?e.value():e;return o||(o=new mn),i(v,g,n,r,o)}}return!!p&&(o||(o=new mn),lo(t,e,n,r,i,o))}function Sr(t){return ec(t)&&Cf(t)==Kt}function Nr(t,e,n,r){var i=n.length,o=i,a=!r;if(null==t)return!o;for(t=rl(t);i--;){var u=n[i];if(a&&u[2]?u[1]!==t[u[0]]:!(u[0]in t))return!1}for(;++i<o;){u=n[i];var c=u[0],s=t[c],l=u[1];if(a&&u[2]){if(s===nt&&!(c in t))return!1}else{var f=new mn;if(r)var p=r(s,l,c,t,e,f);if(!(p===nt?Mr(l,s,pt|ht,r,f):p))return!1}}return!0}function Ar(t){return!(!tc(t)||Ro(t))&&(Qu(t)?yl:ze).test(Xo(t))}function Pr(t){return ec(t)&&vr(t)==Zt}function Or(t){return ec(t)&&Cf(t)==Jt}function Ir(t){return ec(t)&&Ju(t.length)&&!!bn[vr(t)]}function Dr(t){return"function"==typeof t?t:null==t?Ms:"object"==typeof t?hp(t)?Br(t[0],t[1]):jr(t):Ds(t)}function Rr(t){if(!Lo(t))return Bl(t);var e=[];for(var n in rl(t))pl.call(t,n)&&"constructor"!=n&&e.push(n);return e}function Lr(t){if(!tc(t))return Bo(t);var e=Lo(t),n=[];for(var r in t)("constructor"!=r||!e&&pl.call(t,r))&&n.push(r);return n}function Ur(t,e){return t<e}function Fr(t,e){var n=-1,r=Wu(t)?Zs(t.length):[];return ff(t,function(t,i,o){r[++n]=e(t,i,o)}),r}function jr(t){var e=_o(t);return 1==e.length&&e[0][2]?Fo(e[0][0],e[0][1]):function(n){return n===t||Nr(n,t,e)}}function Br(t,e){return Oo(t)&&Uo(e)?Fo($o(t),e):function(n){var r=Ic(n,t);return r===nt&&r===e?Rc(n,t):Mr(e,r,pt|ht)}}function Vr(t,e,n,r,i){t!==e&&hf(e,function(o,a){if(tc(o))i||(i=new mn),Wr(t,e,a,n,Vr,r,i);else{var u=r?r(qo(t,a),o,a+"",t,e,i):nt;u===nt&&(u=o),Rn(t,a,u)}},Uc)}function Wr(t,e,n,r,i,o,a){var u=qo(t,n),c=qo(e,n),s=a.get(c);if(s)return void Rn(t,n,s);var l=o?o(u,c,n+"",t,e,a):nt,f=l===nt;if(f){var p=hp(c),h=!p&&vp(c),d=!p&&!h&&bp(c);l=c,p||h||d?hp(u)?l=u:zu(u)?l=Oi(u):h?(f=!1,l=wi(c,!0)):d?(f=!1,l=Ti(c,!0)):l=[]:sc(c)||pp(c)?(l=u,pp(u)?l=xc(u):tc(u)&&!Qu(u)||(l=Mo(c))):f=!1}f&&(a.set(c,l),i(l,c,r,o,a),a.delete(c)),Rn(t,n,l)}function zr(t,e){var n=t.length;if(n)return e+=e<0?n:0,Ao(e,n)?t[e]:nt}function Hr(t,e,n){var r=-1;return e=h(e.length?e:[Ms],I(mo())),N(Fr(t,function(t,n,i){return{criteria:h(e,function(e){return e(t)}),index:++r,value:t}}),function(t,e){return Ni(t,e,n)})}function qr(t,e){return Yr(t,e,function(e,n){return Rc(t,n)})}function Yr(t,e,n){for(var r=-1,i=e.length,o={};++r<i;){var a=e[r],u=hr(t,a);n(u,a)&&ni(o,bi(a,t),u)}return o}function Kr(t){return function(e){return hr(e,t)}}function Gr(t,e,n,r){var i=r?C:w,o=-1,a=e.length,u=t;for(t===e&&(e=Oi(e)),n&&(u=h(t,I(n)));++o<a;)for(var c=0,s=e[o],l=n?n(s):s;(c=i(u,l,c,r))>-1;)u!==t&&Ml.call(u,c,1),Ml.call(t,c,1);return t}function $r(t,e){for(var n=t?e.length:0,r=n-1;n--;){var i=e[n];if(n==r||i!==o){var o=i;Ao(i)?Ml.call(t,i,1):pi(t,i)}}return t}function Xr(t,e){return t+Rl(ql()*(e-t+1))}function Qr(t,e,n,r){for(var i=-1,o=Vl(Dl((e-t)/(n||1)),0),a=Zs(o);o--;)a[r?o:++i]=t,t+=n;return a}function Zr(t,e){var n="";if(!t||e<1||e>Pt)return n;do{e%2&&(n+=t),(e=Rl(e/2))&&(t+=t)}while(e);return n}function Jr(t,e){return Tf(Wo(t,e,Ms),t+"")}function ti(t){return Pn($c(t))}function ei(t,e){var n=$c(t);return Go(n,Jn(e,0,n.length))}function ni(t,e,n,r){if(!tc(t))return t;e=bi(e,t);for(var i=-1,o=e.length,a=o-1,u=t;null!=u&&++i<o;){var c=$o(e[i]),s=n;if(i!=a){var l=u[c];s=r?r(l,c,u):nt,s===nt&&(s=tc(l)?l:Ao(e[i+1])?[]:{})}Wn(u,c,s),u=u[c]}return t}function ri(t){return Go($c(t))}function ii(t,e,n){var r=-1,i=t.length;e<0&&(e=-e>i?0:i+e),n=n>i?i:n,n<0&&(n+=i),i=e>n?0:n-e>>>0,e>>>=0;for(var o=Zs(i);++r<i;)o[r]=t[r+e];return o}function oi(t,e){var n;return ff(t,function(t,r,i){return!(n=e(t,r,i))}),!!n}function ai(t,e,n){var r=0,i=null==t?r:t.length;if("number"==typeof e&&e===e&&i<=Lt){for(;r<i;){var o=r+i>>>1,a=t[o];null!==a&&!pc(a)&&(n?a<=e:a<e)?r=o+1:i=o}return i}return ui(t,e,Ms,n)}function ui(t,e,n,r){e=n(e);for(var i=0,o=null==t?0:t.length,a=e!==e,u=null===e,c=pc(e),s=e===nt;i<o;){var l=Rl((i+o)/2),f=n(t[l]),p=f!==nt,h=null===f,d=f===f,v=pc(f);if(a)var g=r||d;else g=s?d&&(r||p):u?d&&p&&(r||!h):c?d&&p&&!h&&(r||!v):!h&&!v&&(r?f<=e:f<e);g?i=l+1:o=l}return Wl(o,Rt)}function ci(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n],u=e?e(a):a;if(!n||!Vu(u,c)){var c=u;o[i++]=0===a?0:a}}return o}function si(t){return"number"==typeof t?t:pc(t)?It:+t}function li(t){if("string"==typeof t)return t;if(hp(t))return h(t,li)+"";if(pc(t))return sf?sf.call(t):"";var e=t+"";return"0"==e&&1/t==-At?"-0":e}function fi(t,e,n){var r=-1,i=f,o=t.length,a=!0,u=[],c=u;if(n)a=!1,i=p;else if(o>=rt){var s=e?null:_f(t);if(s)return K(s);a=!1,i=R,c=new dn}else c=e?[]:u;t:for(;++r<o;){var l=t[r],h=e?e(l):l;if(l=n||0!==l?l:0,a&&h===h){for(var d=c.length;d--;)if(c[d]===h)continue t;e&&c.push(h),u.push(l)}else i(c,h,n)||(c!==u&&c.push(h),u.push(l))}return u}function pi(t,e){return e=bi(e,t),null==(t=zo(t,e))||delete t[$o(ma(e))]}function hi(t,e,n,r){return ni(t,e,n(hr(t,e)),r)}function di(t,e,n,r){for(var i=t.length,o=r?i:-1;(r?o--:++o<i)&&e(t[o],o,t););return n?ii(t,r?0:o,r?o+1:i):ii(t,r?o+1:0,r?i:o)}function vi(t,e){var n=t;return n instanceof y&&(n=n.value()),v(e,function(t,e){return e.func.apply(e.thisArg,d([t],e.args))},n)}function gi(t,e,n){var r=t.length;if(r<2)return r?fi(t[0]):[];for(var i=-1,o=Zs(r);++i<r;)for(var a=t[i],u=-1;++u<r;)u!=i&&(o[i]=ir(o[i]||a,t[u],e,n));return fi(sr(o,1),e,n)}function mi(t,e,n){for(var r=-1,i=t.length,o=e.length,a={};++r<i;){var u=r<o?e[r]:nt;n(a,t[r],u)}return a}function yi(t){return zu(t)?t:[]}function _i(t){return"function"==typeof t?t:Ms}function bi(t,e){return hp(t)?t:Oo(t,e)?[t]:Sf(Cc(t))}function xi(t,e,n){var r=t.length;return n=n===nt?r:n,!e&&n>=r?t:ii(t,e,n)}function wi(t,e){if(e)return t.slice();var n=t.length,r=wl?wl(n):new t.constructor(n);return t.copy(r),r}function Ci(t){var e=new t.constructor(t.byteLength);return new xl(e).set(new xl(t)),e}function ki(t,e){var n=e?Ci(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.byteLength)}function Ei(t){var e=new t.constructor(t.source,Be.exec(t));return e.lastIndex=t.lastIndex,e}function Mi(t){return cf?rl(cf.call(t)):{}}function Ti(t,e){var n=e?Ci(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.length)}function Si(t,e){if(t!==e){var n=t!==nt,r=null===t,i=t===t,o=pc(t),a=e!==nt,u=null===e,c=e===e,s=pc(e);if(!u&&!s&&!o&&t>e||o&&a&&c&&!u&&!s||r&&a&&c||!n&&c||!i)return 1;if(!r&&!o&&!s&&t<e||s&&n&&i&&!r&&!o||u&&n&&i||!a&&i||!c)return-1}return 0}function Ni(t,e,n){for(var r=-1,i=t.criteria,o=e.criteria,a=i.length,u=n.length;++r<a;){var c=Si(i[r],o[r]);if(c){if(r>=u)return c;return c*("desc"==n[r]?-1:1)}}return t.index-e.index}function Ai(t,e,n,r){for(var i=-1,o=t.length,a=n.length,u=-1,c=e.length,s=Vl(o-a,0),l=Zs(c+s),f=!r;++u<c;)l[u]=e[u];for(;++i<a;)(f||i<o)&&(l[n[i]]=t[i]);for(;s--;)l[u++]=t[i++];return l}function Pi(t,e,n,r){for(var i=-1,o=t.length,a=-1,u=n.length,c=-1,s=e.length,l=Vl(o-u,0),f=Zs(l+s),p=!r;++i<l;)f[i]=t[i];for(var h=i;++c<s;)f[h+c]=e[c];for(;++a<u;)(p||i<o)&&(f[h+n[a]]=t[i++]);return f}function Oi(t,e){var n=-1,r=t.length;for(e||(e=Zs(r));++n<r;)e[n]=t[n];return e}function Ii(t,e,n,r){var i=!n;n||(n={});for(var o=-1,a=e.length;++o<a;){var u=e[o],c=r?r(n[u],t[u],u,n,t):nt;c===nt&&(c=t[u]),i?Qn(n,u,c):Wn(n,u,c)}return n}function Di(t,e){return Ii(t,xf(t),e)}function Ri(t,e){return Ii(t,wf(t),e)}function Li(t,e){return function(n,r){var i=hp(n)?a:Gn,o=e?e():{};return i(n,t,mo(r,2),o)}}function Ui(t){return Jr(function(e,n){var r=-1,i=n.length,o=i>1?n[i-1]:nt,a=i>2?n[2]:nt;for(o=t.length>3&&"function"==typeof o?(i--,o):nt,a&&Po(n[0],n[1],a)&&(o=i<3?nt:o,i=1),e=rl(e);++r<i;){var u=n[r];u&&t(e,u,r,o)}return e})}function Fi(t,e){return function(n,r){if(null==n)return n;if(!Wu(n))return t(n,r);for(var i=n.length,o=e?i:-1,a=rl(n);(e?o--:++o<i)&&!1!==r(a[o],o,a););return n}}function ji(t){return function(e,n,r){for(var i=-1,o=rl(e),a=r(e),u=a.length;u--;){var c=a[t?u:++i];if(!1===n(o[c],c,o))break}return e}}function Bi(t,e,n){function r(){return(this&&this!==An&&this instanceof r?o:t).apply(i?n:this,arguments)}var i=e&dt,o=zi(t);return r}function Vi(t){return function(e){e=Cc(e);var n=V(e)?Z(e):nt,r=n?n[0]:e.charAt(0),i=n?xi(n,1).join(""):e.slice(1);return r[t]()+i}}function Wi(t){return function(e){return v(xs(es(e).replace(pn,"")),t,"")}}function zi(t){return function(){var e=arguments;switch(e.length){case 0:return new t;case 1:return new t(e[0]);case 2:return new t(e[0],e[1]);case 3:return new t(e[0],e[1],e[2]);case 4:return new t(e[0],e[1],e[2],e[3]);case 5:return new t(e[0],e[1],e[2],e[3],e[4]);case 6:return new t(e[0],e[1],e[2],e[3],e[4],e[5]);case 7:return new t(e[0],e[1],e[2],e[3],e[4],e[5],e[6])}var n=lf(t.prototype),r=t.apply(n,e);return tc(r)?r:n}}function Hi(t,e,n){function r(){for(var a=arguments.length,u=Zs(a),c=a,s=go(r);c--;)u[c]=arguments[c];var l=a<3&&u[0]!==s&&u[a-1]!==s?[]:Y(u,s);return(a-=l.length)<n?eo(t,e,Ki,r.placeholder,nt,u,l,nt,nt,n-a):o(this&&this!==An&&this instanceof r?i:t,this,u)}var i=zi(t);return r}function qi(t){return function(e,n,r){var i=rl(e);if(!Wu(e)){var o=mo(n,3);e=Lc(e),n=function(t){return o(i[t],t,i)}}var a=t(e,n,r);return a>-1?i[o?e[a]:a]:nt}}function Yi(t){return fo(function(e){var n=e.length,r=n,o=i.prototype.thru;for(t&&e.reverse();r--;){var a=e[r];if("function"!=typeof a)throw new al(ot);if(o&&!u&&"wrapper"==vo(a))var u=new i([],!0)}for(r=u?r:n;++r<n;){a=e[r];var c=vo(a),s="wrapper"==c?bf(a):nt;u=s&&Do(s[0])&&s[1]==(xt|mt|_t|wt)&&!s[4].length&&1==s[9]?u[vo(s[0])].apply(u,s[3]):1==a.length&&Do(a)?u[c]():u.thru(a)}return function(){var t=arguments,r=t[0];if(u&&1==t.length&&hp(r))return u.plant(r).value();for(var i=0,o=n?e[i].apply(this,t):r;++i<n;)o=e[i].call(this,o);return o}})}function Ki(t,e,n,r,i,o,a,u,c,s){function l(){for(var m=arguments.length,y=Zs(m),_=m;_--;)y[_]=arguments[_];if(d)var b=go(l),x=F(y,b);if(r&&(y=Ai(y,r,i,d)),o&&(y=Pi(y,o,a,d)),m-=x,d&&m<s){var w=Y(y,b);return eo(t,e,Ki,l.placeholder,n,y,w,u,c,s-m)}var C=p?n:this,k=h?C[t]:t;return m=y.length,u?y=Ho(y,u):v&&m>1&&y.reverse(),f&&c<m&&(y.length=c),this&&this!==An&&this instanceof l&&(k=g||zi(k)),k.apply(C,y)}var f=e&xt,p=e&dt,h=e&vt,d=e&(mt|yt),v=e&Ct,g=h?nt:zi(t);return l}function Gi(t,e){return function(n,r){return xr(n,t,e(r),{})}}function $i(t,e){return function(n,r){var i;if(n===nt&&r===nt)return e;if(n!==nt&&(i=n),r!==nt){if(i===nt)return r;"string"==typeof n||"string"==typeof r?(n=li(n),r=li(r)):(n=si(n),r=si(r)),i=t(n,r)}return i}}function Xi(t){return fo(function(e){return e=h(e,I(mo())),Jr(function(n){var r=this;return t(e,function(t){return o(t,r,n)})})})}function Qi(t,e){e=e===nt?" ":li(e);var n=e.length;if(n<2)return n?Zr(e,t):e;var r=Zr(e,Dl(t/Q(e)));return V(e)?xi(Z(r),0,t).join(""):r.slice(0,t)}function Zi(t,e,n,r){function i(){for(var e=-1,c=arguments.length,s=-1,l=r.length,f=Zs(l+c),p=this&&this!==An&&this instanceof i?u:t;++s<l;)f[s]=r[s];for(;c--;)f[s++]=arguments[++e];return o(p,a?n:this,f)}var a=e&dt,u=zi(t);return i}function Ji(t){return function(e,n,r){return r&&"number"!=typeof r&&Po(e,n,r)&&(n=r=nt),e=mc(e),n===nt?(n=e,e=0):n=mc(n),r=r===nt?e<n?1:-1:mc(r),Qr(e,n,r,t)}}function to(t){return function(e,n){return"string"==typeof e&&"string"==typeof n||(e=bc(e),n=bc(n)),t(e,n)}}function eo(t,e,n,r,i,o,a,u,c,s){var l=e&mt,f=l?a:nt,p=l?nt:a,h=l?o:nt,d=l?nt:o;e|=l?_t:bt,(e&=~(l?bt:_t))&gt||(e&=~(dt|vt));var v=[t,e,i,h,f,d,p,u,c,s],g=n.apply(nt,v);return Do(t)&&Ef(g,v),g.placeholder=r,Yo(g,t,e)}function no(t){var e=nl[t];return function(t,n){if(t=bc(t),n=null==n?0:Wl(yc(n),292)){var r=(Cc(t)+"e").split("e");return r=(Cc(e(r[0]+"e"+(+r[1]+n)))+"e").split("e"),+(r[0]+"e"+(+r[1]-n))}return e(t)}}function ro(t){return function(e){var n=Cf(e);return n==Kt?H(e):n==Jt?G(e):O(e,t(e))}}function io(t,e,n,r,i,o,a,u){var c=e&vt;if(!c&&"function"!=typeof t)throw new al(ot);var s=r?r.length:0;if(s||(e&=~(_t|bt),r=i=nt),a=a===nt?a:Vl(yc(a),0),u=u===nt?u:yc(u),s-=i?i.length:0,e&bt){var l=r,f=i;r=i=nt}var p=c?nt:bf(t),h=[t,e,n,r,i,l,f,o,a,u];if(p&&jo(h,p),t=h[0],e=h[1],n=h[2],r=h[3],i=h[4],u=h[9]=h[9]===nt?c?0:t.length:Vl(h[9]-s,0),!u&&e&(mt|yt)&&(e&=~(mt|yt)),e&&e!=dt)d=e==mt||e==yt?Hi(t,e,u):e!=_t&&e!=(dt|_t)||i.length?Ki.apply(nt,h):Zi(t,e,n,r);else var d=Bi(t,e,n);return Yo((p?vf:Ef)(d,h),t,e)}function oo(t,e,n,r){return t===nt||Vu(t,sl[n])&&!pl.call(r,n)?e:t}function ao(t,e,n,r,i,o){return tc(t)&&tc(e)&&(o.set(e,t),Vr(t,e,nt,ao,o),o.delete(e)),t}function uo(t){return sc(t)?nt:t}function co(t,e,n,r,i,o){var a=n&pt,u=t.length,c=e.length;if(u!=c&&!(a&&c>u))return!1;var s=o.get(t);if(s&&o.get(e))return s==e;var l=-1,f=!0,p=n&ht?new dn:nt;for(o.set(t,e),o.set(e,t);++l<u;){var h=t[l],d=e[l];if(r)var v=a?r(d,h,l,e,t,o):r(h,d,l,t,e,o);if(v!==nt){if(v)continue;f=!1;break}if(p){if(!m(e,function(t,e){if(!R(p,e)&&(h===t||i(h,t,n,r,o)))return p.push(e)})){f=!1;break}}else if(h!==d&&!i(h,d,n,r,o)){f=!1;break}}return o.delete(t),o.delete(e),f}function so(t,e,n,r,i,o,a){switch(n){case ae:if(t.byteLength!=e.byteLength||t.byteOffset!=e.byteOffset)return!1;t=t.buffer,e=e.buffer;case oe:return!(t.byteLength!=e.byteLength||!o(new xl(t),new xl(e)));case Vt:case Wt:case Gt:return Vu(+t,+e);case Ht:return t.name==e.name&&t.message==e.message;case Zt:case te:return t==e+"";case Kt:var u=H;case Jt:var c=r&pt;if(u||(u=K),t.size!=e.size&&!c)return!1;var s=a.get(t);if(s)return s==e;r|=ht,a.set(t,e);var l=co(u(t),u(e),r,i,o,a);return a.delete(t),l;case ee:if(cf)return cf.call(t)==cf.call(e)}return!1}function lo(t,e,n,r,i,o){var a=n&pt,u=po(t),c=u.length;if(c!=po(e).length&&!a)return!1;for(var s=c;s--;){var l=u[s];if(!(a?l in e:pl.call(e,l)))return!1}var f=o.get(t);if(f&&o.get(e))return f==e;var p=!0;o.set(t,e),o.set(e,t);for(var h=a;++s<c;){l=u[s];var d=t[l],v=e[l];if(r)var g=a?r(v,d,l,e,t,o):r(d,v,l,t,e,o);if(!(g===nt?d===v||i(d,v,n,r,o):g)){p=!1;break}h||(h="constructor"==l)}if(p&&!h){var m=t.constructor,y=e.constructor;m!=y&&"constructor"in t&&"constructor"in e&&!("function"==typeof m&&m instanceof m&&"function"==typeof y&&y instanceof y)&&(p=!1)}return o.delete(t),o.delete(e),p}function fo(t){return Tf(Wo(t,nt,sa),t+"")}function po(t){return dr(t,Lc,xf)}function ho(t){return dr(t,Uc,wf)}function vo(t){for(var e=t.name+"",n=tf[e],r=pl.call(tf,e)?n.length:0;r--;){var i=n[r],o=i.func;if(null==o||o==t)return i.name}return e}function go(t){return(pl.call(n,"placeholder")?n:t).placeholder}function mo(){var t=n.iteratee||Ts;return t=t===Ts?Dr:t,arguments.length?t(arguments[0],arguments[1]):t}function yo(t,e){var n=t.__data__;return Io(e)?n["string"==typeof e?"string":"hash"]:n.map}function _o(t){for(var e=Lc(t),n=e.length;n--;){var r=e[n],i=t[r];e[n]=[r,i,Uo(i)]}return e}function bo(t,e){var n=B(t,e);return Ar(n)?n:nt}function xo(t){var e=pl.call(t,Nl),n=t[Nl];try{t[Nl]=nt;var r=!0}catch(t){}var i=vl.call(t);return r&&(e?t[Nl]=n:delete t[Nl]),i}function wo(t,e,n){for(var r=-1,i=n.length;++r<i;){var o=n[r],a=o.size;switch(o.type){case"drop":t+=a;break;case"dropRight":e-=a;break;case"take":e=Wl(e,t+a);break;case"takeRight":t=Vl(t,e-a)}}return{start:t,end:e}}function Co(t){var e=t.match(Re);return e?e[1].split(Le):[]}function ko(t,e,n){e=bi(e,t);for(var r=-1,i=e.length,o=!1;++r<i;){var a=$o(e[r]);if(!(o=null!=t&&n(t,a)))break;t=t[a]}return o||++r!=i?o:!!(i=null==t?0:t.length)&&Ju(i)&&Ao(a,i)&&(hp(t)||pp(t))}function Eo(t){var e=t.length,n=new t.constructor(e);return e&&"string"==typeof t[0]&&pl.call(t,"index")&&(n.index=t.index,n.input=t.input),n}function Mo(t){return"function"!=typeof t.constructor||Lo(t)?{}:lf(Cl(t))}function To(t,e,n){var r=t.constructor;switch(e){case oe:return Ci(t);case Vt:case Wt:return new r(+t);case ae:return ki(t,n);case ue:case ce:case se:case le:case fe:case pe:case he:case de:case ve:return Ti(t,n);case Kt:return new r;case Gt:case te:return new r(t);case Zt:return Ei(t);case Jt:return new r;case ee:return Mi(t)}}function So(t,e){var n=e.length;if(!n)return t;var r=n-1;return e[r]=(n>1?"& ":"")+e[r],e=e.join(n>2?", ":" "),t.replace(De,"{\n/* [wrapped with "+e+"] */\n")}function No(t){return hp(t)||pp(t)||!!(Tl&&t&&t[Tl])}function Ao(t,e){var n=typeof t;return!!(e=null==e?Pt:e)&&("number"==n||"symbol"!=n&&qe.test(t))&&t>-1&&t%1==0&&t<e}function Po(t,e,n){if(!tc(n))return!1;var r=typeof e;return!!("number"==r?Wu(n)&&Ao(e,n.length):"string"==r&&e in n)&&Vu(n[e],t)}function Oo(t,e){if(hp(t))return!1;var n=typeof t;return!("number"!=n&&"symbol"!=n&&"boolean"!=n&&null!=t&&!pc(t))||(Te.test(t)||!Me.test(t)||null!=e&&t in rl(e))}function Io(t){var e=typeof t;return"string"==e||"number"==e||"symbol"==e||"boolean"==e?"__proto__"!==t:null===t}function Do(t){var e=vo(t),r=n[e];if("function"!=typeof r||!(e in y.prototype))return!1;if(t===r)return!0;var i=bf(r);return!!i&&t===i[0]}function Ro(t){return!!dl&&dl in t}function Lo(t){var e=t&&t.constructor;return t===("function"==typeof e&&e.prototype||sl)}function Uo(t){return t===t&&!tc(t)}function Fo(t,e){return function(n){return null!=n&&(n[t]===e&&(e!==nt||t in rl(n)))}}function jo(t,e){var n=t[1],r=e[1],i=n|r,o=i<(dt|vt|xt),a=r==xt&&n==mt||r==xt&&n==wt&&t[7].length<=e[8]||r==(xt|wt)&&e[7].length<=e[8]&&n==mt;if(!o&&!a)return t;r&dt&&(t[2]=e[2],i|=n&dt?0:gt);var u=e[3];if(u){var c=t[3];t[3]=c?Ai(c,u,e[4]):u,t[4]=c?Y(t[3],ct):e[4]}return u=e[5],u&&(c=t[5],t[5]=c?Pi(c,u,e[6]):u,t[6]=c?Y(t[5],ct):e[6]),u=e[7],u&&(t[7]=u),r&xt&&(t[8]=null==t[8]?e[8]:Wl(t[8],e[8])),null==t[9]&&(t[9]=e[9]),t[0]=e[0],t[1]=i,t}function Bo(t){var e=[];if(null!=t)for(var n in rl(t))e.push(n);return e}function Vo(t){return vl.call(t)}function Wo(t,e,n){return e=Vl(e===nt?t.length-1:e,0),function(){for(var r=arguments,i=-1,a=Vl(r.length-e,0),u=Zs(a);++i<a;)u[i]=r[e+i];i=-1;for(var c=Zs(e+1);++i<e;)c[i]=r[i];return c[e]=n(u),o(t,this,c)}}function zo(t,e){return e.length<2?t:hr(t,ii(e,0,-1))}function Ho(t,e){for(var n=t.length,r=Wl(e.length,n),i=Oi(t);r--;){var o=e[r];t[r]=Ao(o,n)?i[o]:nt}return t}function qo(t,e){if("__proto__"!=e)return t[e]}function Yo(t,e,n){var r=e+"";return Tf(t,So(r,Qo(Co(r),n)))}function Ko(t){var e=0,n=0;return function(){var r=zl(),i=Tt-(r-n);if(n=r,i>0){if(++e>=Mt)return arguments[0]}else e=0;return t.apply(nt,arguments)}}function Go(t,e){var n=-1,r=t.length,i=r-1;for(e=e===nt?r:e;++n<e;){var o=Xr(n,i),a=t[o];t[o]=t[n],t[n]=a}return t.length=e,t}function $o(t){if("string"==typeof t||pc(t))return t;var e=t+"";return"0"==e&&1/t==-At?"-0":e}function Xo(t){if(null!=t){try{return fl.call(t)}catch(t){}try{return t+""}catch(t){}}return""}function Qo(t,e){return u(Ut,function(n){var r="_."+n[0];e&n[1]&&!f(t,r)&&t.push(r)}),t.sort()}function Zo(t){if(t instanceof y)return t.clone();var e=new i(t.__wrapped__,t.__chain__);return e.__actions__=Oi(t.__actions__),e.__index__=t.__index__,e.__values__=t.__values__,e}function Jo(t,e,n){e=(n?Po(t,e,n):e===nt)?1:Vl(yc(e),0);var r=null==t?0:t.length;if(!r||e<1)return[];for(var i=0,o=0,a=Zs(Dl(r/e));i<r;)a[o++]=ii(t,i,i+=e);return a}function ta(t){for(var e=-1,n=null==t?0:t.length,r=0,i=[];++e<n;){var o=t[e];o&&(i[r++]=o)}return i}function ea(){var t=arguments.length;if(!t)return[];for(var e=Zs(t-1),n=arguments[0],r=t;r--;)e[r-1]=arguments[r];return d(hp(n)?Oi(n):[n],sr(e,1))}function na(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),ii(t,e<0?0:e,r)):[]}function ra(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),e=r-e,ii(t,0,e<0?0:e)):[]}function ia(t,e){return t&&t.length?di(t,mo(e,3),!0,!0):[]}function oa(t,e){return t&&t.length?di(t,mo(e,3),!0):[]}function aa(t,e,n,r){var i=null==t?0:t.length;return i?(n&&"number"!=typeof n&&Po(t,e,n)&&(n=0,r=i),ur(t,e,n,r)):[]}function ua(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:yc(n);return i<0&&(i=Vl(r+i,0)),x(t,mo(e,3),i)}function ca(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r-1;return n!==nt&&(i=yc(n),i=n<0?Vl(r+i,0):Wl(i,r-1)),x(t,mo(e,3),i,!0)}function sa(t){return(null==t?0:t.length)?sr(t,1):[]}function la(t){return(null==t?0:t.length)?sr(t,At):[]}function fa(t,e){return(null==t?0:t.length)?(e=e===nt?1:yc(e),sr(t,e)):[]}function pa(t){for(var e=-1,n=null==t?0:t.length,r={};++e<n;){var i=t[e];r[i[0]]=i[1]}return r}function ha(t){return t&&t.length?t[0]:nt}function da(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:yc(n);return i<0&&(i=Vl(r+i,0)),w(t,e,i)}function va(t){return(null==t?0:t.length)?ii(t,0,-1):[]}function ga(t,e){return null==t?"":jl.call(t,e)}function ma(t){var e=null==t?0:t.length;return e?t[e-1]:nt}function ya(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r;return n!==nt&&(i=yc(n),i=i<0?Vl(r+i,0):Wl(i,r-1)),e===e?X(t,e,i):x(t,k,i,!0)}function _a(t,e){return t&&t.length?zr(t,yc(e)):nt}function ba(t,e){return t&&t.length&&e&&e.length?Gr(t,e):t}function xa(t,e,n){return t&&t.length&&e&&e.length?Gr(t,e,mo(n,2)):t}function wa(t,e,n){return t&&t.length&&e&&e.length?Gr(t,e,nt,n):t}function Ca(t,e){var n=[];if(!t||!t.length)return n;var r=-1,i=[],o=t.length;for(e=mo(e,3);++r<o;){var a=t[r];e(a,r,t)&&(n.push(a),i.push(r))}return $r(t,i),n}function ka(t){return null==t?t:Yl.call(t)}function Ea(t,e,n){var r=null==t?0:t.length;return r?(n&&"number"!=typeof n&&Po(t,e,n)?(e=0,n=r):(e=null==e?0:yc(e),n=n===nt?r:yc(n)),ii(t,e,n)):[]}function Ma(t,e){return ai(t,e)}function Ta(t,e,n){return ui(t,e,mo(n,2))}function Sa(t,e){var n=null==t?0:t.length;if(n){var r=ai(t,e);if(r<n&&Vu(t[r],e))return r}return-1}function Na(t,e){return ai(t,e,!0)}function Aa(t,e,n){return ui(t,e,mo(n,2),!0)}function Pa(t,e){if(null==t?0:t.length){var n=ai(t,e,!0)-1;if(Vu(t[n],e))return n}return-1}function Oa(t){return t&&t.length?ci(t):[]}function Ia(t,e){return t&&t.length?ci(t,mo(e,2)):[]}function Da(t){var e=null==t?0:t.length;return e?ii(t,1,e):[]}function Ra(t,e,n){return t&&t.length?(e=n||e===nt?1:yc(e),ii(t,0,e<0?0:e)):[]}function La(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),e=r-e,ii(t,e<0?0:e,r)):[]}function Ua(t,e){return t&&t.length?di(t,mo(e,3),!1,!0):[]}function Fa(t,e){return t&&t.length?di(t,mo(e,3)):[]}function ja(t){return t&&t.length?fi(t):[]}function Ba(t,e){return t&&t.length?fi(t,mo(e,2)):[]}function Va(t,e){return e="function"==typeof e?e:nt,t&&t.length?fi(t,nt,e):[]}function Wa(t){if(!t||!t.length)return[];var e=0;return t=l(t,function(t){if(zu(t))return e=Vl(t.length,e),!0}),P(e,function(e){return h(t,M(e))})}function za(t,e){if(!t||!t.length)return[];var n=Wa(t);return null==e?n:h(n,function(t){return o(e,nt,t)})}function Ha(t,e){return mi(t||[],e||[],Wn)}function qa(t,e){return mi(t||[],e||[],ni)}function Ya(t){var e=n(t);return e.__chain__=!0,e}function Ka(t,e){return e(t),t}function Ga(t,e){return e(t)}function $a(){return Ya(this)}function Xa(){return new i(this.value(),this.__chain__)}function Qa(){this.__values__===nt&&(this.__values__=gc(this.value()));var t=this.__index__>=this.__values__.length;return{done:t,value:t?nt:this.__values__[this.__index__++]}}function Za(){return this}function Ja(t){for(var e,n=this;n instanceof r;){var i=Zo(n);i.__index__=0,i.__values__=nt,e?o.__wrapped__=i:e=i;var o=i;n=n.__wrapped__}return o.__wrapped__=t,e}function tu(){var t=this.__wrapped__;if(t instanceof y){var e=t;return this.__actions__.length&&(e=new y(this)),e=e.reverse(),e.__actions__.push({func:Ga,args:[ka],thisArg:nt}),new i(e,this.__chain__)}return this.thru(ka)}function eu(){return vi(this.__wrapped__,this.__actions__)}function nu(t,e,n){var r=hp(t)?s:or;return n&&Po(t,e,n)&&(e=nt),r(t,mo(e,3))}function ru(t,e){return(hp(t)?l:cr)(t,mo(e,3))}function iu(t,e){return sr(lu(t,e),1)}function ou(t,e){return sr(lu(t,e),At)}function au(t,e,n){return n=n===nt?1:yc(n),sr(lu(t,e),n)}function uu(t,e){return(hp(t)?u:ff)(t,mo(e,3))}function cu(t,e){return(hp(t)?c:pf)(t,mo(e,3))}function su(t,e,n,r){t=Wu(t)?t:$c(t),n=n&&!r?yc(n):0;var i=t.length;return n<0&&(n=Vl(i+n,0)),fc(t)?n<=i&&t.indexOf(e,n)>-1:!!i&&w(t,e,n)>-1}function lu(t,e){return(hp(t)?h:Fr)(t,mo(e,3))}function fu(t,e,n,r){return null==t?[]:(hp(e)||(e=null==e?[]:[e]),n=r?nt:n,hp(n)||(n=null==n?[]:[n]),Hr(t,e,n))}function pu(t,e,n){var r=hp(t)?v:S,i=arguments.length<3;return r(t,mo(e,4),n,i,ff)}function hu(t,e,n){var r=hp(t)?g:S,i=arguments.length<3;return r(t,mo(e,4),n,i,pf)}function du(t,e){return(hp(t)?l:cr)(t,Su(mo(e,3)))}function vu(t){return(hp(t)?Pn:ti)(t)}function gu(t,e,n){return e=(n?Po(t,e,n):e===nt)?1:yc(e),(hp(t)?On:ei)(t,e)}function mu(t){return(hp(t)?Dn:ri)(t)}function yu(t){if(null==t)return 0;if(Wu(t))return fc(t)?Q(t):t.length;var e=Cf(t);return e==Kt||e==Jt?t.size:Rr(t).length}function _u(t,e,n){var r=hp(t)?m:oi;return n&&Po(t,e,n)&&(e=nt),r(t,mo(e,3))}function bu(t,e){if("function"!=typeof e)throw new al(ot);return t=yc(t),function(){if(--t<1)return e.apply(this,arguments)}}function xu(t,e,n){return e=n?nt:e,e=t&&null==e?t.length:e,io(t,xt,nt,nt,nt,nt,e)}function wu(t,e){var n;if("function"!=typeof e)throw new al(ot);return t=yc(t),function(){return--t>0&&(n=e.apply(this,arguments)),t<=1&&(e=nt),n}}function Cu(t,e,n){e=n?nt:e;var r=io(t,mt,nt,nt,nt,nt,nt,e);return r.placeholder=Cu.placeholder,r}function ku(t,e,n){e=n?nt:e;var r=io(t,yt,nt,nt,nt,nt,nt,e);return r.placeholder=ku.placeholder,r}function Eu(t,e,n){function r(e){var n=p,r=h;return p=h=nt,y=e,v=t.apply(r,n)}function i(t){return y=t,g=Mf(u,e),_?r(t):v}function o(t){var n=t-m,r=t-y,i=e-n;return b?Wl(i,d-r):i}function a(t){var n=t-m,r=t-y;return m===nt||n>=e||n<0||b&&r>=d}function u(){var t=ep();if(a(t))return c(t);g=Mf(u,o(t))}function c(t){return g=nt,x&&p?r(t):(p=h=nt,v)}function s(){g!==nt&&yf(g),y=0,p=m=h=g=nt}function l(){return g===nt?v:c(ep())}function f(){var t=ep(),n=a(t);if(p=arguments,h=this,m=t,n){if(g===nt)return i(m);if(b)return g=Mf(u,e),r(m)}return g===nt&&(g=Mf(u,e)),v}var p,h,d,v,g,m,y=0,_=!1,b=!1,x=!0;if("function"!=typeof t)throw new al(ot);return e=bc(e)||0,tc(n)&&(_=!!n.leading,b="maxWait"in n,d=b?Vl(bc(n.maxWait)||0,e):d,x="trailing"in n?!!n.trailing:x),f.cancel=s,f.flush=l,f}function Mu(t){return io(t,Ct)}function Tu(t,e){if("function"!=typeof t||null!=e&&"function"!=typeof e)throw new al(ot);var n=function(){var r=arguments,i=e?e.apply(this,r):r[0],o=n.cache;if(o.has(i))return o.get(i);var a=t.apply(this,r);return n.cache=o.set(i,a)||o,a};return n.cache=new(Tu.Cache||an),n}function Su(t){if("function"!=typeof t)throw new al(ot);return function(){var e=arguments;switch(e.length){case 0:return!t.call(this);case 1:return!t.call(this,e[0]);case 2:return!t.call(this,e[0],e[1]);case 3:return!t.call(this,e[0],e[1],e[2])}return!t.apply(this,e)}}function Nu(t){return wu(2,t)}function Au(t,e){if("function"!=typeof t)throw new al(ot);return e=e===nt?e:yc(e),Jr(t,e)}function Pu(t,e){if("function"!=typeof t)throw new al(ot);return e=null==e?0:Vl(yc(e),0),Jr(function(n){var r=n[e],i=xi(n,0,e);return r&&d(i,r),o(t,this,i)})}function Ou(t,e,n){var r=!0,i=!0;if("function"!=typeof t)throw new al(ot);return tc(n)&&(r="leading"in n?!!n.leading:r,i="trailing"in n?!!n.trailing:i),Eu(t,e,{leading:r,maxWait:e,trailing:i})}function Iu(t){return xu(t,1)}function Du(t,e){return up(_i(e),t)}function Ru(){if(!arguments.length)return[];var t=arguments[0];return hp(t)?t:[t]}function Lu(t){return tr(t,ft)}function Uu(t,e){return e="function"==typeof e?e:nt,tr(t,ft,e)}function Fu(t){return tr(t,st|ft)}function ju(t,e){return e="function"==typeof e?e:nt,tr(t,st|ft,e)}function Bu(t,e){return null==e||nr(t,e,Lc(e))}function Vu(t,e){return t===e||t!==t&&e!==e}function Wu(t){return null!=t&&Ju(t.length)&&!Qu(t)}function zu(t){return ec(t)&&Wu(t)}function Hu(t){return!0===t||!1===t||ec(t)&&vr(t)==Vt}function qu(t){return ec(t)&&1===t.nodeType&&!sc(t)}function Yu(t){if(null==t)return!0;if(Wu(t)&&(hp(t)||"string"==typeof t||"function"==typeof t.splice||vp(t)||bp(t)||pp(t)))return!t.length;var e=Cf(t);if(e==Kt||e==Jt)return!t.size;if(Lo(t))return!Rr(t).length;for(var n in t)if(pl.call(t,n))return!1;return!0}function Ku(t,e){return Mr(t,e)}function Gu(t,e,n){n="function"==typeof n?n:nt;var r=n?n(t,e):nt;return r===nt?Mr(t,e,nt,n):!!r}function $u(t){if(!ec(t))return!1;var e=vr(t);return e==Ht||e==zt||"string"==typeof t.message&&"string"==typeof t.name&&!sc(t)}function Xu(t){return"number"==typeof t&&Fl(t)}function Qu(t){if(!tc(t))return!1;var e=vr(t);return e==qt||e==Yt||e==Bt||e==Qt}function Zu(t){return"number"==typeof t&&t==yc(t)}function Ju(t){return"number"==typeof t&&t>-1&&t%1==0&&t<=Pt}function tc(t){var e=typeof t;return null!=t&&("object"==e||"function"==e)}function ec(t){return null!=t&&"object"==typeof t}function nc(t,e){return t===e||Nr(t,e,_o(e))}function rc(t,e,n){return n="function"==typeof n?n:nt,Nr(t,e,_o(e),n)}function ic(t){return cc(t)&&t!=+t}function oc(t){if(kf(t))throw new tl(it);return Ar(t)}function ac(t){return null===t}function uc(t){return null==t}function cc(t){return"number"==typeof t||ec(t)&&vr(t)==Gt}function sc(t){if(!ec(t)||vr(t)!=Xt)return!1;var e=Cl(t);if(null===e)return!0;var n=pl.call(e,"constructor")&&e.constructor;return"function"==typeof n&&n instanceof n&&fl.call(n)==gl}function lc(t){return Zu(t)&&t>=-Pt&&t<=Pt}function fc(t){return"string"==typeof t||!hp(t)&&ec(t)&&vr(t)==te}function pc(t){return"symbol"==typeof t||ec(t)&&vr(t)==ee}function hc(t){return t===nt}function dc(t){return ec(t)&&Cf(t)==re}function vc(t){return ec(t)&&vr(t)==ie}function gc(t){if(!t)return[];if(Wu(t))return fc(t)?Z(t):Oi(t);if(Sl&&t[Sl])return z(t[Sl]());var e=Cf(t);return(e==Kt?H:e==Jt?K:$c)(t)}function mc(t){if(!t)return 0===t?t:0;if((t=bc(t))===At||t===-At){return(t<0?-1:1)*Ot}return t===t?t:0}function yc(t){var e=mc(t),n=e%1;return e===e?n?e-n:e:0}function _c(t){return t?Jn(yc(t),0,Dt):0}function bc(t){if("number"==typeof t)return t;if(pc(t))return It;if(tc(t)){var e="function"==typeof t.valueOf?t.valueOf():t;t=tc(e)?e+"":e}if("string"!=typeof t)return 0===t?t:+t;t=t.replace(Pe,"");var n=We.test(t);return n||He.test(t)?Tn(t.slice(2),n?2:8):Ve.test(t)?It:+t}function xc(t){return Ii(t,Uc(t))}function wc(t){return t?Jn(yc(t),-Pt,Pt):0===t?t:0}function Cc(t){return null==t?"":li(t)}function kc(t,e){var n=lf(t);return null==e?n:$n(n,e)}function Ec(t,e){return b(t,mo(e,3),lr)}function Mc(t,e){return b(t,mo(e,3),fr)}function Tc(t,e){return null==t?t:hf(t,mo(e,3),Uc)}function Sc(t,e){return null==t?t:df(t,mo(e,3),Uc)}function Nc(t,e){return t&&lr(t,mo(e,3))}function Ac(t,e){return t&&fr(t,mo(e,3))}function Pc(t){return null==t?[]:pr(t,Lc(t))}function Oc(t){return null==t?[]:pr(t,Uc(t))}function Ic(t,e,n){var r=null==t?nt:hr(t,e);return r===nt?n:r}function Dc(t,e){return null!=t&&ko(t,e,mr)}function Rc(t,e){return null!=t&&ko(t,e,yr)}function Lc(t){return Wu(t)?Nn(t):Rr(t)}function Uc(t){return Wu(t)?Nn(t,!0):Lr(t)}function Fc(t,e){var n={};return e=mo(e,3),lr(t,function(t,r,i){Qn(n,e(t,r,i),t)}),n}function jc(t,e){var n={};return e=mo(e,3),lr(t,function(t,r,i){Qn(n,r,e(t,r,i))}),n}function Bc(t,e){return Vc(t,Su(mo(e)))}function Vc(t,e){if(null==t)return{};var n=h(ho(t),function(t){return[t]});return e=mo(e),Yr(t,n,function(t,n){return e(t,n[0])})}function Wc(t,e,n){e=bi(e,t);var r=-1,i=e.length;for(i||(i=1,t=nt);++r<i;){var o=null==t?nt:t[$o(e[r])];o===nt&&(r=i,o=n),t=Qu(o)?o.call(t):o}return t}function zc(t,e,n){return null==t?t:ni(t,e,n)}function Hc(t,e,n,r){return r="function"==typeof r?r:nt,null==t?t:ni(t,e,n,r)}function qc(t,e,n){var r=hp(t),i=r||vp(t)||bp(t);if(e=mo(e,4),null==n){var o=t&&t.constructor;n=i?r?new o:[]:tc(t)&&Qu(o)?lf(Cl(t)):{}}return(i?u:lr)(t,function(t,r,i){return e(n,t,r,i)}),n}function Yc(t,e){return null==t||pi(t,e)}function Kc(t,e,n){return null==t?t:hi(t,e,_i(n))}function Gc(t,e,n,r){return r="function"==typeof r?r:nt,null==t?t:hi(t,e,_i(n),r)}function $c(t){return null==t?[]:D(t,Lc(t))}function Xc(t){return null==t?[]:D(t,Uc(t))}function Qc(t,e,n){return n===nt&&(n=e,e=nt),n!==nt&&(n=bc(n),n=n===n?n:0),e!==nt&&(e=bc(e),e=e===e?e:0),Jn(bc(t),e,n)}function Zc(t,e,n){return e=mc(e),n===nt?(n=e,e=0):n=mc(n),t=bc(t),_r(t,e,n)}function Jc(t,e,n){if(n&&"boolean"!=typeof n&&Po(t,e,n)&&(e=n=nt),n===nt&&("boolean"==typeof e?(n=e,e=nt):"boolean"==typeof t&&(n=t,t=nt)),t===nt&&e===nt?(t=0,e=1):(t=mc(t),e===nt?(e=t,t=0):e=mc(e)),t>e){var r=t;t=e,e=r}if(n||t%1||e%1){var i=ql();return Wl(t+i*(e-t+Mn("1e-"+((i+"").length-1))),e)}return Xr(t,e)}function ts(t){return Yp(Cc(t).toLowerCase())}function es(t){return(t=Cc(t))&&t.replace(Ye,zn).replace(hn,"")}function ns(t,e,n){t=Cc(t),e=li(e);var r=t.length;n=n===nt?r:Jn(yc(n),0,r);var i=n;return(n-=e.length)>=0&&t.slice(n,i)==e}function rs(t){return t=Cc(t),t&&we.test(t)?t.replace(be,Hn):t}function is(t){return t=Cc(t),t&&Ae.test(t)?t.replace(Ne,"\\$&"):t}function os(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;if(!e||r>=e)return t;var i=(e-r)/2;return Qi(Rl(i),n)+t+Qi(Dl(i),n)}function as(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;return e&&r<e?t+Qi(e-r,n):t}function us(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;return e&&r<e?Qi(e-r,n)+t:t}function cs(t,e,n){return n||null==e?e=0:e&&(e=+e),Hl(Cc(t).replace(Oe,""),e||0)}function ss(t,e,n){return e=(n?Po(t,e,n):e===nt)?1:yc(e),Zr(Cc(t),e)}function ls(){var t=arguments,e=Cc(t[0]);return t.length<3?e:e.replace(t[1],t[2])}function fs(t,e,n){return n&&"number"!=typeof n&&Po(t,e,n)&&(e=n=nt),(n=n===nt?Dt:n>>>0)?(t=Cc(t),t&&("string"==typeof e||null!=e&&!yp(e))&&!(e=li(e))&&V(t)?xi(Z(t),0,n):t.split(e,n)):[]}function ps(t,e,n){return t=Cc(t),n=null==n?0:Jn(yc(n),0,t.length),e=li(e),t.slice(n,n+e.length)==e}function hs(t,e,r){var i=n.templateSettings;r&&Po(t,e,r)&&(e=nt),t=Cc(t),e=Ep({},e,i,oo);var o,a,u=Ep({},e.imports,i.imports,oo),c=Lc(u),s=D(u,c),l=0,f=e.interpolate||Ke,p="__p += '",h=il((e.escape||Ke).source+"|"+f.source+"|"+(f===Ee?je:Ke).source+"|"+(e.evaluate||Ke).source+"|$","g"),d="//# sourceURL="+("sourceURL"in e?e.sourceURL:"lodash.templateSources["+ ++_n+"]")+"\n";t.replace(h,function(e,n,r,i,u,c){return r||(r=i),p+=t.slice(l,c).replace(Ge,j),n&&(o=!0,p+="' +\n__e("+n+") +\n'"),u&&(a=!0,p+="';\n"+u+";\n__p += '"),r&&(p+="' +\n((__t = ("+r+")) == null ? '' : __t) +\n'"),l=c+e.length,e}),p+="';\n";var v=e.variable;v||(p="with (obj) {\n"+p+"\n}\n"),p=(a?p.replace(ge,""):p).replace(me,"$1").replace(ye,"$1;"),p="function("+(v||"obj")+") {\n"+(v?"":"obj || (obj = {});\n")+"var __t, __p = ''"+(o?", __e = _.escape":"")+(a?", __j = Array.prototype.join;\nfunction print() { __p += __j.call(arguments, '') }\n":";\n")+p+"return __p\n}";var g=Kp(function(){return el(c,d+"return "+p).apply(nt,s)});if(g.source=p,$u(g))throw g;return g}function ds(t){return Cc(t).toLowerCase()}function vs(t){return Cc(t).toUpperCase()}function gs(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Pe,"");if(!t||!(e=li(e)))return t;var r=Z(t),i=Z(e);return xi(r,L(r,i),U(r,i)+1).join("")}function ms(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Ie,"");if(!t||!(e=li(e)))return t;var r=Z(t);return xi(r,0,U(r,Z(e))+1).join("")}function ys(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Oe,"");if(!t||!(e=li(e)))return t;var r=Z(t);return xi(r,L(r,Z(e))).join("")}function _s(t,e){var n=kt,r=Et;if(tc(e)){var i="separator"in e?e.separator:i;n="length"in e?yc(e.length):n,r="omission"in e?li(e.omission):r}t=Cc(t);var o=t.length;if(V(t)){var a=Z(t);o=a.length}if(n>=o)return t;var u=n-Q(r);if(u<1)return r;var c=a?xi(a,0,u).join(""):t.slice(0,u);if(i===nt)return c+r;if(a&&(u+=c.length-u),yp(i)){if(t.slice(u).search(i)){var s,l=c;for(i.global||(i=il(i.source,Cc(Be.exec(i))+"g")),i.lastIndex=0;s=i.exec(l);)var f=s.index;c=c.slice(0,f===nt?u:f)}}else if(t.indexOf(li(i),u)!=u){var p=c.lastIndexOf(i);p>-1&&(c=c.slice(0,p))}return c+r}function bs(t){return t=Cc(t),t&&xe.test(t)?t.replace(_e,qn):t}function xs(t,e,n){return t=Cc(t),e=n?nt:e,e===nt?W(t)?et(t):_(t):t.match(e)||[]}function ws(t){var e=null==t?0:t.length,n=mo();return t=e?h(t,function(t){if("function"!=typeof t[1])throw new al(ot);return[n(t[0]),t[1]]}):[],Jr(function(n){for(var r=-1;++r<e;){var i=t[r];if(o(i[0],this,n))return o(i[1],this,n)}})}function Cs(t){return er(tr(t,st))}function ks(t){return function(){return t}}function Es(t,e){return null==t||t!==t?e:t}function Ms(t){return t}function Ts(t){return Dr("function"==typeof t?t:tr(t,st))}function Ss(t){return jr(tr(t,st))}function Ns(t,e){return Br(t,tr(e,st))}function As(t,e,n){var r=Lc(e),i=pr(e,r);null!=n||tc(e)&&(i.length||!r.length)||(n=e,e=t,t=this,i=pr(e,Lc(e)));var o=!(tc(n)&&"chain"in n&&!n.chain),a=Qu(t);return u(i,function(n){var r=e[n];t[n]=r,a&&(t.prototype[n]=function(){var e=this.__chain__;if(o||e){var n=t(this.__wrapped__);return(n.__actions__=Oi(this.__actions__)).push({func:r,args:arguments,thisArg:t}),n.__chain__=e,n}return r.apply(t,d([this.value()],arguments))})}),t}function Ps(){return An._===this&&(An._=ml),this}function Os(){}function Is(t){return t=yc(t),Jr(function(e){return zr(e,t)})}function Ds(t){return Oo(t)?M($o(t)):Kr(t)}function Rs(t){return function(e){return null==t?nt:hr(t,e)}}function Ls(){return[]}function Us(){return!1}function Fs(){return{}}function js(){return""}function Bs(){return!0}function Vs(t,e){if((t=yc(t))<1||t>Pt)return[];var n=Dt,r=Wl(t,Dt);e=mo(e),t-=Dt;for(var i=P(r,e);++n<t;)e(n);return i}function Ws(t){return hp(t)?h(t,$o):pc(t)?[t]:Oi(Sf(Cc(t)))}function zs(t){var e=++hl;return Cc(t)+e}function Hs(t){return t&&t.length?ar(t,Ms,gr):nt}function qs(t,e){return t&&t.length?ar(t,mo(e,2),gr):nt}function Ys(t){return E(t,Ms)}function Ks(t,e){return E(t,mo(e,2))}function Gs(t){return t&&t.length?ar(t,Ms,Ur):nt}function $s(t,e){return t&&t.length?ar(t,mo(e,2),Ur):nt}function Xs(t){return t&&t.length?A(t,Ms):0}function Qs(t,e){return t&&t.length?A(t,mo(e,2)):0}e=null==e?An:Yn.defaults(An.Object(),e,Yn.pick(An,yn));var Zs=e.Array,Js=e.Date,tl=e.Error,el=e.Function,nl=e.Math,rl=e.Object,il=e.RegExp,ol=e.String,al=e.TypeError,ul=Zs.prototype,cl=el.prototype,sl=rl.prototype,ll=e["__core-js_shared__"],fl=cl.toString,pl=sl.hasOwnProperty,hl=0,dl=function(){var t=/[^.]+$/.exec(ll&&ll.keys&&ll.keys.IE_PROTO||"");return t?"Symbol(src)_1."+t:""}(),vl=sl.toString,gl=fl.call(rl),ml=An._,yl=il("^"+fl.call(pl).replace(Ne,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$"),_l=In?e.Buffer:nt,bl=e.Symbol,xl=e.Uint8Array,wl=_l?_l.allocUnsafe:nt,Cl=q(rl.getPrototypeOf,rl),kl=rl.create,El=sl.propertyIsEnumerable,Ml=ul.splice,Tl=bl?bl.isConcatSpreadable:nt,Sl=bl?bl.iterator:nt,Nl=bl?bl.toStringTag:nt,Al=function(){try{var t=bo(rl,"defineProperty");return t({},"",{}),t}catch(t){}}(),Pl=e.clearTimeout!==An.clearTimeout&&e.clearTimeout,Ol=Js&&Js.now!==An.Date.now&&Js.now,Il=e.setTimeout!==An.setTimeout&&e.setTimeout,Dl=nl.ceil,Rl=nl.floor,Ll=rl.getOwnPropertySymbols,Ul=_l?_l.isBuffer:nt,Fl=e.isFinite,jl=ul.join,Bl=q(rl.keys,rl),Vl=nl.max,Wl=nl.min,zl=Js.now,Hl=e.parseInt,ql=nl.random,Yl=ul.reverse,Kl=bo(e,"DataView"),Gl=bo(e,"Map"),$l=bo(e,"Promise"),Xl=bo(e,"Set"),Ql=bo(e,"WeakMap"),Zl=bo(rl,"create"),Jl=Ql&&new Ql,tf={},ef=Xo(Kl),nf=Xo(Gl),rf=Xo($l),of=Xo(Xl),af=Xo(Ql),uf=bl?bl.prototype:nt,cf=uf?uf.valueOf:nt,sf=uf?uf.toString:nt,lf=function(){function t(){}return function(e){if(!tc(e))return{};if(kl)return kl(e);t.prototype=e;var n=new t;return t.prototype=nt,n}}();n.templateSettings={escape:Ce,evaluate:ke,interpolate:Ee,variable:"",imports:{_:n}},n.prototype=r.prototype,n.prototype.constructor=n,i.prototype=lf(r.prototype),i.prototype.constructor=i,y.prototype=lf(r.prototype),y.prototype.constructor=y,tt.prototype.clear=Ue,tt.prototype.delete=$e,tt.prototype.get=Xe,tt.prototype.has=Qe,tt.prototype.set=Ze,Je.prototype.clear=tn,Je.prototype.delete=en,Je.prototype.get=nn,Je.prototype.has=rn,Je.prototype.set=on,an.prototype.clear=un,an.prototype.delete=cn,an.prototype.get=sn,an.prototype.has=ln,an.prototype.set=fn,dn.prototype.add=dn.prototype.push=vn,dn.prototype.has=gn,mn.prototype.clear=wn,mn.prototype.delete=Cn,mn.prototype.get=kn,mn.prototype.has=En,mn.prototype.set=Sn;var ff=Fi(lr),pf=Fi(fr,!0),hf=ji(),df=ji(!0),vf=Jl?function(t,e){return Jl.set(t,e),t}:Ms,gf=Al?function(t,e){return Al(t,"toString",{configurable:!0,enumerable:!1,value:ks(e),writable:!0})}:Ms,mf=Jr,yf=Pl||function(t){return An.clearTimeout(t)},_f=Xl&&1/K(new Xl([,-0]))[1]==At?function(t){return new Xl(t)}:Os,bf=Jl?function(t){return Jl.get(t)}:Os,xf=Ll?function(t){return null==t?[]:(t=rl(t),l(Ll(t),function(e){return El.call(t,e)}))}:Ls,wf=Ll?function(t){for(var e=[];t;)d(e,xf(t)),t=Cl(t);return e}:Ls,Cf=vr;(Kl&&Cf(new Kl(new ArrayBuffer(1)))!=ae||Gl&&Cf(new Gl)!=Kt||$l&&"[object Promise]"!=Cf($l.resolve())||Xl&&Cf(new Xl)!=Jt||Ql&&Cf(new Ql)!=re)&&(Cf=function(t){var e=vr(t),n=e==Xt?t.constructor:nt,r=n?Xo(n):"";if(r)switch(r){case ef:return ae;case nf:return Kt;case rf:return"[object Promise]";case of:return Jt;case af:return re}return e});var kf=ll?Qu:Us,Ef=Ko(vf),Mf=Il||function(t,e){return An.setTimeout(t,e)},Tf=Ko(gf),Sf=function(t){var e=Tu(t,function(t){return n.size===ut&&n.clear(),t}),n=e.cache;return e}(function(t){var e=[];return 46===t.charCodeAt(0)&&e.push(""),t.replace(Se,function(t,n,r,i){e.push(r?i.replace(Fe,"$1"):n||t)}),e}),Nf=Jr(function(t,e){return zu(t)?ir(t,sr(e,1,zu,!0)):[]}),Af=Jr(function(t,e){var n=ma(e);return zu(n)&&(n=nt),zu(t)?ir(t,sr(e,1,zu,!0),mo(n,2)):[]}),Pf=Jr(function(t,e){var n=ma(e);return zu(n)&&(n=nt),zu(t)?ir(t,sr(e,1,zu,!0),nt,n):[]}),Of=Jr(function(t){var e=h(t,yi);return e.length&&e[0]===t[0]?br(e):[]}),If=Jr(function(t){var e=ma(t),n=h(t,yi);return e===ma(n)?e=nt:n.pop(),n.length&&n[0]===t[0]?br(n,mo(e,2)):[]}),Df=Jr(function(t){var e=ma(t),n=h(t,yi);return e="function"==typeof e?e:nt,e&&n.pop(),n.length&&n[0]===t[0]?br(n,nt,e):[]}),Rf=Jr(ba),Lf=fo(function(t,e){var n=null==t?0:t.length,r=Zn(t,e);return $r(t,h(e,function(t){return Ao(t,n)?+t:t}).sort(Si)),r}),Uf=Jr(function(t){return fi(sr(t,1,zu,!0))}),Ff=Jr(function(t){var e=ma(t);return zu(e)&&(e=nt),fi(sr(t,1,zu,!0),mo(e,2))}),jf=Jr(function(t){var e=ma(t);return e="function"==typeof e?e:nt,fi(sr(t,1,zu,!0),nt,e)}),Bf=Jr(function(t,e){return zu(t)?ir(t,e):[]}),Vf=Jr(function(t){return gi(l(t,zu))}),Wf=Jr(function(t){var e=ma(t);return zu(e)&&(e=nt),gi(l(t,zu),mo(e,2))}),zf=Jr(function(t){var e=ma(t);return e="function"==typeof e?e:nt,gi(l(t,zu),nt,e)}),Hf=Jr(Wa),qf=Jr(function(t){var e=t.length,n=e>1?t[e-1]:nt;return n="function"==typeof n?(t.pop(),n):nt,za(t,n)}),Yf=fo(function(t){var e=t.length,n=e?t[0]:0,r=this.__wrapped__,o=function(e){return Zn(e,t)};return!(e>1||this.__actions__.length)&&r instanceof y&&Ao(n)?(r=r.slice(n,+n+(e?1:0)),r.__actions__.push({func:Ga,args:[o],thisArg:nt}),new i(r,this.__chain__).thru(function(t){return e&&!t.length&&t.push(nt),t})):this.thru(o)}),Kf=Li(function(t,e,n){pl.call(t,n)?++t[n]:Qn(t,n,1)}),Gf=qi(ua),$f=qi(ca),Xf=Li(function(t,e,n){pl.call(t,n)?t[n].push(e):Qn(t,n,[e])}),Qf=Jr(function(t,e,n){var r=-1,i="function"==typeof e,a=Wu(t)?Zs(t.length):[];return ff(t,function(t){a[++r]=i?o(e,t,n):wr(t,e,n)}),a}),Zf=Li(function(t,e,n){Qn(t,n,e)}),Jf=Li(function(t,e,n){t[n?0:1].push(e)},function(){return[[],[]]}),tp=Jr(function(t,e){if(null==t)return[];var n=e.length;return n>1&&Po(t,e[0],e[1])?e=[]:n>2&&Po(e[0],e[1],e[2])&&(e=[e[0]]),Hr(t,sr(e,1),[])}),ep=Ol||function(){return An.Date.now()},np=Jr(function(t,e,n){var r=dt;if(n.length){var i=Y(n,go(np));r|=_t}return io(t,r,e,n,i)}),rp=Jr(function(t,e,n){var r=dt|vt;if(n.length){var i=Y(n,go(rp));r|=_t}return io(e,r,t,n,i)}),ip=Jr(function(t,e){return rr(t,1,e)}),op=Jr(function(t,e,n){return rr(t,bc(e)||0,n)});Tu.Cache=an;var ap=mf(function(t,e){e=1==e.length&&hp(e[0])?h(e[0],I(mo())):h(sr(e,1),I(mo()));var n=e.length;return Jr(function(r){for(var i=-1,a=Wl(r.length,n);++i<a;)r[i]=e[i].call(this,r[i]);return o(t,this,r)})}),up=Jr(function(t,e){var n=Y(e,go(up));return io(t,_t,nt,e,n)}),cp=Jr(function(t,e){var n=Y(e,go(cp));return io(t,bt,nt,e,n)}),sp=fo(function(t,e){return io(t,wt,nt,nt,nt,e)}),lp=to(gr),fp=to(function(t,e){return t>=e}),pp=Cr(function(){return arguments}())?Cr:function(t){return ec(t)&&pl.call(t,"callee")&&!El.call(t,"callee")},hp=Zs.isArray,dp=Ln?I(Ln):kr,vp=Ul||Us,gp=Un?I(Un):Er,mp=Fn?I(Fn):Sr,yp=jn?I(jn):Pr,_p=Bn?I(Bn):Or,bp=Vn?I(Vn):Ir,xp=to(Ur),wp=to(function(t,e){return t<=e}),Cp=Ui(function(t,e){if(Lo(e)||Wu(e))return void Ii(e,Lc(e),t);for(var n in e)pl.call(e,n)&&Wn(t,n,e[n])}),kp=Ui(function(t,e){Ii(e,Uc(e),t)}),Ep=Ui(function(t,e,n,r){Ii(e,Uc(e),t,r)}),Mp=Ui(function(t,e,n,r){Ii(e,Lc(e),t,r)}),Tp=fo(Zn),Sp=Jr(function(t,e){t=rl(t);var n=-1,r=e.length,i=r>2?e[2]:nt;for(i&&Po(e[0],e[1],i)&&(r=1);++n<r;)for(var o=e[n],a=Uc(o),u=-1,c=a.length;++u<c;){var s=a[u],l=t[s];(l===nt||Vu(l,sl[s])&&!pl.call(t,s))&&(t[s]=o[s])}return t}),Np=Jr(function(t){return t.push(nt,ao),o(Dp,nt,t)}),Ap=Gi(function(t,e,n){null!=e&&"function"!=typeof e.toString&&(e=vl.call(e)),t[e]=n},ks(Ms)),Pp=Gi(function(t,e,n){null!=e&&"function"!=typeof e.toString&&(e=vl.call(e)),pl.call(t,e)?t[e].push(n):t[e]=[n]},mo),Op=Jr(wr),Ip=Ui(function(t,e,n){Vr(t,e,n)}),Dp=Ui(function(t,e,n,r){Vr(t,e,n,r)}),Rp=fo(function(t,e){var n={};if(null==t)return n;var r=!1;e=h(e,function(e){return e=bi(e,t),r||(r=e.length>1),e}),Ii(t,ho(t),n),r&&(n=tr(n,st|lt|ft,uo));for(var i=e.length;i--;)pi(n,e[i]);return n}),Lp=fo(function(t,e){return null==t?{}:qr(t,e)}),Up=ro(Lc),Fp=ro(Uc),jp=Wi(function(t,e,n){return e=e.toLowerCase(),t+(n?ts(e):e)}),Bp=Wi(function(t,e,n){return t+(n?"-":"")+e.toLowerCase()}),Vp=Wi(function(t,e,n){return t+(n?" ":"")+e.toLowerCase()}),Wp=Vi("toLowerCase"),zp=Wi(function(t,e,n){return t+(n?"_":"")+e.toLowerCase()}),Hp=Wi(function(t,e,n){return t+(n?" ":"")+Yp(e)}),qp=Wi(function(t,e,n){return t+(n?" ":"")+e.toUpperCase()}),Yp=Vi("toUpperCase"),Kp=Jr(function(t,e){try{return o(t,nt,e)}catch(t){return $u(t)?t:new tl(t)}}),Gp=fo(function(t,e){return u(e,function(e){e=$o(e),Qn(t,e,np(t[e],t))}),t}),$p=Yi(),Xp=Yi(!0),Qp=Jr(function(t,e){return function(n){return wr(n,t,e)}}),Zp=Jr(function(t,e){return function(n){return wr(t,n,e)}}),Jp=Xi(h),th=Xi(s),eh=Xi(m),nh=Ji(),rh=Ji(!0),ih=$i(function(t,e){return t+e},0),oh=no("ceil"),ah=$i(function(t,e){return t/e},1),uh=no("floor"),ch=$i(function(t,e){return t*e},1),sh=no("round"),lh=$i(function(t,e){return t-e},0);return n.after=bu,n.ary=xu,n.assign=Cp,n.assignIn=kp,n.assignInWith=Ep,n.assignWith=Mp,n.at=Tp,n.before=wu,n.bind=np,n.bindAll=Gp,n.bindKey=rp,n.castArray=Ru,n.chain=Ya,n.chunk=Jo,n.compact=ta,n.concat=ea,n.cond=ws,n.conforms=Cs,n.constant=ks,n.countBy=Kf,n.create=kc,n.curry=Cu,n.curryRight=ku,n.debounce=Eu,n.defaults=Sp,n.defaultsDeep=Np,n.defer=ip,n.delay=op,n.difference=Nf,n.differenceBy=Af,n.differenceWith=Pf,n.drop=na,n.dropRight=ra,n.dropRightWhile=ia,n.dropWhile=oa,n.fill=aa,n.filter=ru,n.flatMap=iu,n.flatMapDeep=ou,n.flatMapDepth=au,n.flatten=sa,n.flattenDeep=la,n.flattenDepth=fa,n.flip=Mu,n.flow=$p,n.flowRight=Xp,n.fromPairs=pa,n.functions=Pc,n.functionsIn=Oc,n.groupBy=Xf,n.initial=va,n.intersection=Of,n.intersectionBy=If,n.intersectionWith=Df,n.invert=Ap,n.invertBy=Pp,n.invokeMap=Qf,n.iteratee=Ts,n.keyBy=Zf,n.keys=Lc,n.keysIn=Uc,n.map=lu,n.mapKeys=Fc,n.mapValues=jc,n.matches=Ss,n.matchesProperty=Ns,n.memoize=Tu,n.merge=Ip,n.mergeWith=Dp,n.method=Qp,n.methodOf=Zp,n.mixin=As,n.negate=Su,n.nthArg=Is,n.omit=Rp,n.omitBy=Bc,n.once=Nu,n.orderBy=fu,n.over=Jp,n.overArgs=ap,n.overEvery=th,n.overSome=eh,n.partial=up,n.partialRight=cp,n.partition=Jf,n.pick=Lp,n.pickBy=Vc,n.property=Ds,n.propertyOf=Rs,n.pull=Rf,n.pullAll=ba,n.pullAllBy=xa,n.pullAllWith=wa,n.pullAt=Lf,n.range=nh,n.rangeRight=rh,n.rearg=sp,n.reject=du,n.remove=Ca,n.rest=Au,n.reverse=ka,n.sampleSize=gu,n.set=zc,n.setWith=Hc,n.shuffle=mu,n.slice=Ea,n.sortBy=tp,n.sortedUniq=Oa,n.sortedUniqBy=Ia,n.split=fs,n.spread=Pu,n.tail=Da,n.take=Ra,n.takeRight=La,n.takeRightWhile=Ua,n.takeWhile=Fa,n.tap=Ka,n.throttle=Ou,n.thru=Ga,n.toArray=gc,n.toPairs=Up,n.toPairsIn=Fp,n.toPath=Ws,n.toPlainObject=xc,n.transform=qc,n.unary=Iu,n.union=Uf,n.unionBy=Ff,n.unionWith=jf,n.uniq=ja,n.uniqBy=Ba,n.uniqWith=Va,n.unset=Yc,n.unzip=Wa,n.unzipWith=za,n.update=Kc,n.updateWith=Gc,n.values=$c,n.valuesIn=Xc,n.without=Bf,n.words=xs,n.wrap=Du,n.xor=Vf,n.xorBy=Wf,n.xorWith=zf,n.zip=Hf,n.zipObject=Ha,n.zipObjectDeep=qa,n.zipWith=qf,n.entries=Up,n.entriesIn=Fp,n.extend=kp,n.extendWith=Ep,As(n,n),n.add=ih,n.attempt=Kp,n.camelCase=jp,n.capitalize=ts,n.ceil=oh,n.clamp=Qc,n.clone=Lu,n.cloneDeep=Fu,n.cloneDeepWith=ju,n.cloneWith=Uu,n.conformsTo=Bu,n.deburr=es,n.defaultTo=Es,n.divide=ah,n.endsWith=ns,n.eq=Vu,n.escape=rs,n.escapeRegExp=is,n.every=nu,n.find=Gf,n.findIndex=ua,n.findKey=Ec,n.findLast=$f,n.findLastIndex=ca,n.findLastKey=Mc,n.floor=uh,n.forEach=uu,n.forEachRight=cu,n.forIn=Tc,n.forInRight=Sc,n.forOwn=Nc,n.forOwnRight=Ac,n.get=Ic,n.gt=lp,n.gte=fp,n.has=Dc,n.hasIn=Rc,n.head=ha,n.identity=Ms,n.includes=su,n.indexOf=da,n.inRange=Zc,n.invoke=Op,n.isArguments=pp,n.isArray=hp,n.isArrayBuffer=dp,n.isArrayLike=Wu,n.isArrayLikeObject=zu,n.isBoolean=Hu,n.isBuffer=vp,n.isDate=gp,n.isElement=qu,n.isEmpty=Yu,n.isEqual=Ku,n.isEqualWith=Gu,n.isError=$u,n.isFinite=Xu,n.isFunction=Qu,n.isInteger=Zu,n.isLength=Ju,n.isMap=mp,n.isMatch=nc,n.isMatchWith=rc,n.isNaN=ic,n.isNative=oc,n.isNil=uc,n.isNull=ac,n.isNumber=cc,n.isObject=tc,n.isObjectLike=ec,n.isPlainObject=sc,n.isRegExp=yp,n.isSafeInteger=lc,n.isSet=_p,n.isString=fc,n.isSymbol=pc,n.isTypedArray=bp,n.isUndefined=hc,n.isWeakMap=dc,n.isWeakSet=vc,n.join=ga,n.kebabCase=Bp,n.last=ma,n.lastIndexOf=ya,n.lowerCase=Vp,n.lowerFirst=Wp,n.lt=xp,n.lte=wp,n.max=Hs,n.maxBy=qs,n.mean=Ys,n.meanBy=Ks,n.min=Gs,n.minBy=$s,n.stubArray=Ls,n.stubFalse=Us,n.stubObject=Fs,n.stubString=js,n.stubTrue=Bs,n.multiply=ch,n.nth=_a,n.noConflict=Ps,n.noop=Os,n.now=ep,n.pad=os,n.padEnd=as,n.padStart=us,n.parseInt=cs,n.random=Jc,n.reduce=pu,n.reduceRight=hu,n.repeat=ss,n.replace=ls,n.result=Wc,n.round=sh,n.runInContext=t,n.sample=vu,n.size=yu,n.snakeCase=zp,n.some=_u,n.sortedIndex=Ma,n.sortedIndexBy=Ta,n.sortedIndexOf=Sa,n.sortedLastIndex=Na,n.sortedLastIndexBy=Aa,n.sortedLastIndexOf=Pa,n.startCase=Hp,n.startsWith=ps,n.subtract=lh,n.sum=Xs,n.sumBy=Qs,n.template=hs,n.times=Vs,n.toFinite=mc,n.toInteger=yc,n.toLength=_c,n.toLower=ds,n.toNumber=bc,n.toSafeInteger=wc,n.toString=Cc,n.toUpper=vs,n.trim=gs,n.trimEnd=ms,n.trimStart=ys,n.truncate=_s,n.unescape=bs,n.uniqueId=zs,n.upperCase=qp,n.upperFirst=Yp,n.each=uu,n.eachRight=cu,n.first=ha,As(n,function(){var t={};return lr(n,function(e,r){pl.call(n.prototype,r)||(t[r]=e)}),t}(),{chain:!1}),n.VERSION="4.17.11",u(["bind","bindKey","curry","curryRight","partial","partialRight"],function(t){n[t].placeholder=n}),u(["drop","take"],function(t,e){y.prototype[t]=function(n){n=n===nt?1:Vl(yc(n),0);var r=this.__filtered__&&!e?new y(this):this.clone();return r.__filtered__?r.__takeCount__=Wl(n,r.__takeCount__):r.__views__.push({size:Wl(n,Dt),type:t+(r.__dir__<0?"Right":"")}),r},y.prototype[t+"Right"]=function(e){return this.reverse()[t](e).reverse()}}),u(["filter","map","takeWhile"],function(t,e){var n=e+1,r=n==St||3==n;y.prototype[t]=function(t){var e=this.clone();return e.__iteratees__.push({iteratee:mo(t,3),type:n}),e.__filtered__=e.__filtered__||r,e}}),u(["head","last"],function(t,e){var n="take"+(e?"Right":"");y.prototype[t]=function(){return this[n](1).value()[0]}}),u(["initial","tail"],function(t,e){var n="drop"+(e?"":"Right");y.prototype[t]=function(){return this.__filtered__?new y(this):this[n](1)}}),y.prototype.compact=function(){return this.filter(Ms)},y.prototype.find=function(t){return this.filter(t).head()},y.prototype.findLast=function(t){return this.reverse().find(t)},y.prototype.invokeMap=Jr(function(t,e){return"function"==typeof t?new y(this):this.map(function(n){return wr(n,t,e)})}),y.prototype.reject=function(t){return this.filter(Su(mo(t)))},y.prototype.slice=function(t,e){t=yc(t);var n=this;return n.__filtered__&&(t>0||e<0)?new y(n):(t<0?n=n.takeRight(-t):t&&(n=n.drop(t)),e!==nt&&(e=yc(e),n=e<0?n.dropRight(-e):n.take(e-t)),n)},y.prototype.takeRightWhile=function(t){return this.reverse().takeWhile(t).reverse()},y.prototype.toArray=function(){return this.take(Dt)},lr(y.prototype,function(t,e){var r=/^(?:filter|find|map|reject)|While$/.test(e),o=/^(?:head|last)$/.test(e),a=n[o?"take"+("last"==e?"Right":""):e],u=o||/^find/.test(e);a&&(n.prototype[e]=function(){var e=this.__wrapped__,c=o?[1]:arguments,s=e instanceof y,l=c[0],f=s||hp(e),p=function(t){var e=a.apply(n,d([t],c));return o&&h?e[0]:e};f&&r&&"function"==typeof l&&1!=l.length&&(s=f=!1);var h=this.__chain__,v=!!this.__actions__.length,g=u&&!h,m=s&&!v;if(!u&&f){e=m?e:new y(this);var _=t.apply(e,c);return _.__actions__.push({func:Ga,args:[p],thisArg:nt}),new i(_,h)}return g&&m?t.apply(this,c):(_=this.thru(p),g?o?_.value()[0]:_.value():_)})}),u(["pop","push","shift","sort","splice","unshift"],function(t){var e=ul[t],r=/^(?:push|sort|unshift)$/.test(t)?"tap":"thru",i=/^(?:pop|shift)$/.test(t);n.prototype[t]=function(){var t=arguments;if(i&&!this.__chain__){var n=this.value();return e.apply(hp(n)?n:[],t)}return this[r](function(n){return e.apply(hp(n)?n:[],t)})}}),lr(y.prototype,function(t,e){var r=n[e];if(r){var i=r.name+"";(tf[i]||(tf[i]=[])).push({name:e,func:r})}}),tf[Ki(nt,vt).name]=[{name:"wrapper",func:nt}],y.prototype.clone=T,y.prototype.reverse=$,y.prototype.value=J,n.prototype.at=Yf,n.prototype.chain=$a,n.prototype.commit=Xa,n.prototype.next=Qa,n.prototype.plant=Ja,n.prototype.reverse=tu,n.prototype.toJSON=n.prototype.valueOf=n.prototype.value=eu,n.prototype.first=n.prototype.head,Sl&&(n.prototype[Sl]=Za),n}();An._=Yn,(i=function(){return Yn}.call(e,n,e,r))!==nt&&(r.exports=i)}).call(this)}).call(e,n(98),n(99)(t))},function(t,e,n){"use strict";var r={remove:function(t){t._reactInternalInstance=void 0},get:function(t){return t._reactInternalInstance},has:function(t){return void 0!==t._reactInternalInstance},set:function(t,e){t._reactInternalInstance=e}};t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";t.exports=n(26)},function(t,e,n){"use strict";var r=n(63);e.a=function(t){return t=n.i(r.a)(Math.abs(t)),t?t[1]:NaN}},function(t,e,n){"use strict";e.a=function(t,e){return t=+t,e-=t,function(n){return t+e*n}}},function(t,e,n){"use strict";function r(t,e){return(e-=t=+t)?function(n){return(n-t)/e}:n.i(h.a)(e)}function i(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=e?0:t>=n?1:r(t)}}}function o(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=0?e:t>=1?n:r(t)}}}function a(t,e,n,r){var i=t[0],o=t[1],a=e[0],u=e[1];return o<i?(i=n(o,i),a=r(u,a)):(i=n(i,o),a=r(a,u)),function(t){return a(i(t))}}function u(t,e,r,i){var o=Math.min(t.length,e.length)-1,a=new Array(o),u=new Array(o),c=-1;for(t[o]<t[0]&&(t=t.slice().reverse(),e=e.slice().reverse());++c<o;)a[c]=r(t[c],t[c+1]),u[c]=i(e[c],e[c+1]);return function(e){var r=n.i(l.bisect)(t,e,1,o)-1;return u[r](a[r](e))}}function c(t,e){return e.domain(t.domain()).range(t.range()).interpolate(t.interpolate()).clamp(t.clamp())}function s(t,e){function n(){return s=Math.min(g.length,m.length)>2?u:a,l=h=null,c}function c(e){return(l||(l=s(g,m,_?i(t):t,y)))(+e)}var s,l,h,g=v,m=v,y=f.b,_=!1;return c.invert=function(t){return(h||(h=s(m,g,r,_?o(e):e)))(+t)},c.domain=function(t){return arguments.length?(g=p.a.call(t,d.a),n()):g.slice()},c.range=function(t){return arguments.length?(m=p.b.call(t),n()):m.slice()},c.rangeRound=function(t){return m=p.b.call(t),y=f.c,n()},c.clamp=function(t){return arguments.length?(_=!!t,n()):_},c.interpolate=function(t){return arguments.length?(y=t,n()):y},n()}e.b=r,e.c=c,e.a=s;var l=n(7),f=n(30),p=n(16),h=n(67),d=n(126),v=[0,1]},function(t,e,n){"use strict";function r(t){return function(){var e=this.ownerDocument,n=this.namespaceURI;return n===a.b&&e.documentElement.namespaceURI===a.b?e.createElement(t):e.createElementNS(n,t)}}function i(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}var o=n(68),a=n(69);e.a=function(t){var e=n.i(o.a)(t);return(e.local?i:r)(e)}},function(t,e,n){"use strict";e.a=function(t,e){var n=t.ownerSVGElement||t;if(n.createSVGPoint){var r=n.createSVGPoint();return r.x=e.clientX,r.y=e.clientY,r=r.matrixTransform(t.getScreenCTM().inverse()),[r.x,r.y]}var i=t.getBoundingClientRect();return[e.clientX-i.left-t.clientLeft,e.clientY-i.top-t.clientTop]}},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo((2*t._x0+t._x1)/3,(2*t._y0+t._y1)/3,(t._x0+2*t._x1)/3,(t._y0+2*t._y1)/3,(t._x0+4*t._x1+e)/6,(t._y0+4*t._y1+n)/6)}function i(t){this._context=t}e.c=r,e.b=i,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN,this._point=0},lineEnd:function(){switch(this._point){case 3:r(this,this._x1,this._y1);case 2:this._context.lineTo(this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3,this._context.lineTo((5*this._x0+this._x1)/6,(5*this._y0+this._y1)/6);default:r(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new i(t)}},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo(t._x1+t._k*(t._x2-t._x0),t._y1+t._k*(t._y2-t._y0),t._x2+t._k*(t._x1-e),t._y2+t._k*(t._y1-n),t._x2,t._y2)}function i(t,e){this._context=t,this._k=(1-e)/6}e.c=r,e.b=i,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:r(this,this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2,this._x1=t,this._y1=e;break;case 2:this._point=3;default:r(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new i(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._context=t}r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;default:this._context.lineTo(t,e)}}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";e.a=function(){}},function(t,e,n){"use strict";var r={};t.exports=r},function(t,e,n){"use strict";function r(t){return"topMouseUp"===t||"topTouchEnd"===t||"topTouchCancel"===t}function i(t){return"topMouseMove"===t||"topTouchMove"===t}function o(t){return"topMouseDown"===t||"topTouchStart"===t}function a(t,e,n,r){var i=t.type||"unknown-event";t.currentTarget=m.getNodeFromInstance(r),e?v.invokeGuardedCallbackWithCatch(i,n,t):v.invokeGuardedCallback(i,n,t),t.currentTarget=null}function u(t,e){var n=t._dispatchListeners,r=t._dispatchInstances;if(Array.isArray(n))for(var i=0;i<n.length&&!t.isPropagationStopped();i++)a(t,e,n[i],r[i]);else n&&a(t,e,n,r);t._dispatchListeners=null,t._dispatchInstances=null}function c(t){var e=t._dispatchListeners,n=t._dispatchInstances;if(Array.isArray(e)){for(var r=0;r<e.length&&!t.isPropagationStopped();r++)if(e[r](t,n[r]))return n[r]}else if(e&&e(t,n))return n;return null}function s(t){var e=c(t);return t._dispatchInstances=null,t._dispatchListeners=null,e}function l(t){var e=t._dispatchListeners,n=t._dispatchInstances;Array.isArray(e)&&d("103"),t.currentTarget=e?m.getNodeFromInstance(n):null;var r=e?e(t):null;return t.currentTarget=null,t._dispatchListeners=null,t._dispatchInstances=null,r}function f(t){return!!t._dispatchListeners}var p,h,d=n(1),v=n(88),g=(n(0),n(2),{injectComponentTree:function(t){p=t},injectTreeTraversal:function(t){h=t}}),m={isEndish:r,isMoveish:i,isStartish:o,executeDirectDispatch:l,executeDispatchesInOrder:u,executeDispatchesInOrderStopAtTrue:s,hasDispatches:f,getInstanceFromNode:function(t){return p.getInstanceFromNode(t)},getNodeFromInstance:function(t){return p.getNodeFromInstance(t)},isAncestor:function(t,e){return h.isAncestor(t,e)},getLowestCommonAncestor:function(t,e){return h.getLowestCommonAncestor(t,e)},getParentInstance:function(t){return h.getParentInstance(t)},traverseTwoPhase:function(t,e,n){return h.traverseTwoPhase(t,e,n)},traverseEnterLeave:function(t,e,n,r,i){return h.traverseEnterLeave(t,e,n,r,i)},injection:g};t.exports=m},function(t,e,n){"use strict";function r(t){return Object.prototype.hasOwnProperty.call(t,v)||(t[v]=h++,f[t[v]]={}),f[t[v]]}var i,o=n(3),a=n(84),u=n(374),c=n(90),s=n(406),l=n(95),f={},p=!1,h=0,d={topAbort:"abort",topAnimationEnd:s("animationend")||"animationend",topAnimationIteration:s("animationiteration")||"animationiteration",topAnimationStart:s("animationstart")||"animationstart",topBlur:"blur",topCanPlay:"canplay",topCanPlayThrough:"canplaythrough",topChange:"change",topClick:"click",topCompositionEnd:"compositionend",topCompositionStart:"compositionstart",topCompositionUpdate:"compositionupdate",topContextMenu:"contextmenu",topCopy:"copy",topCut:"cut",topDoubleClick:"dblclick",topDrag:"drag",topDragEnd:"dragend",topDragEnter:"dragenter",topDragExit:"dragexit",topDragLeave:"dragleave",topDragOver:"dragover",topDragStart:"dragstart",topDrop:"drop",topDurationChange:"durationchange",topEmptied:"emptied",topEncrypted:"encrypted",topEnded:"ended",topError:"error",topFocus:"focus",topInput:"input",topKeyDown:"keydown",topKeyPress:"keypress",topKeyUp:"keyup",topLoadedData:"loadeddata",topLoadedMetadata:"loadedmetadata",topLoadStart:"loadstart",topMouseDown:"mousedown",topMouseMove:"mousemove",topMouseOut:"mouseout",topMouseOver:"mouseover",topMouseUp:"mouseup",topPaste:"paste",topPause:"pause",topPlay:"play",topPlaying:"playing",topProgress:"progress",topRateChange:"ratechange",topScroll:"scroll",topSeeked:"seeked",topSeeking:"seeking",topSelectionChange:"selectionchange",topStalled:"stalled",topSuspend:"suspend",topTextInput:"textInput",topTimeUpdate:"timeupdate",topTouchCancel:"touchcancel",topTouchEnd:"touchend",topTouchMove:"touchmove",topTouchStart:"touchstart",topTransitionEnd:s("transitionend")||"transitionend",topVolumeChange:"volumechange",topWaiting:"waiting",topWheel:"wheel"},v="_reactListenersID"+String(Math.random()).slice(2),g=o({},u,{ReactEventListener:null,injection:{injectReactEventListener:function(t){t.setHandleTopLevel(g.handleTopLevel),g.ReactEventListener=t}},setEnabled:function(t){g.ReactEventListener&&g.ReactEventListener.setEnabled(t)},isEnabled:function(){return!(!g.ReactEventListener||!g.ReactEventListener.isEnabled())},listenTo:function(t,e){for(var n=e,i=r(n),o=a.registrationNameDependencies[t],u=0;u<o.length;u++){var c=o[u];i.hasOwnProperty(c)&&i[c]||("topWheel"===c?l("wheel")?g.ReactEventListener.trapBubbledEvent("topWheel","wheel",n):l("mousewheel")?g.ReactEventListener.trapBubbledEvent("topWheel","mousewheel",n):g.ReactEventListener.trapBubbledEvent("topWheel","DOMMouseScroll",n):"topScroll"===c?l("scroll",!0)?g.ReactEventListener.trapCapturedEvent("topScroll","scroll",n):g.ReactEventListener.trapBubbledEvent("topScroll","scroll",g.ReactEventListener.WINDOW_HANDLE):"topFocus"===c||"topBlur"===c?(l("focus",!0)?(g.ReactEventListener.trapCapturedEvent("topFocus","focus",n),g.ReactEventListener.trapCapturedEvent("topBlur","blur",n)):l("focusin")&&(g.ReactEventListener.trapBubbledEvent("topFocus","focusin",n),g.ReactEventListener.trapBubbledEvent("topBlur","focusout",n)),i.topBlur=!0,i.topFocus=!0):d.hasOwnProperty(c)&&g.ReactEventListener.trapBubbledEvent(c,d[c],n),i[c]=!0)}},trapBubbledEvent:function(t,e,n){return g.ReactEventListener.trapBubbledEvent(t,e,n)},trapCapturedEvent:function(t,e,n){return g.ReactEventListener.trapCapturedEvent(t,e,n)},supportsEventPageXY:function(){if(!document.createEvent)return!1;var t=document.createEvent("MouseEvent");return null!=t&&"pageX"in t},ensureScrollValueMonitoring:function(){if(void 0===i&&(i=g.supportsEventPageXY()),!i&&!p){var t=c.refreshScrollValues;g.ReactEventListener.monitorScrollValue(t),p=!0}}});t.exports=g},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(90),a=n(93),u={screenX:null,screenY:null,clientX:null,clientY:null,ctrlKey:null,shiftKey:null,altKey:null,metaKey:null,getModifierState:a,button:function(t){var e=t.button;return"which"in t?e:2===e?2:4===e?1:0},buttons:null,relatedTarget:function(t){return t.relatedTarget||(t.fromElement===t.srcElement?t.toElement:t.fromElement)},pageX:function(t){return"pageX"in t?t.pageX:t.clientX+o.currentScrollLeft},pageY:function(t){return"pageY"in t?t.pageY:t.clientY+o.currentScrollTop}};i.augmentClass(r,u),t.exports=r},function(t,e,n){"use strict";var r=n(1),i=(n(0),{}),o={reinitializeTransaction:function(){this.transactionWrappers=this.getTransactionWrappers(),this.wrapperInitData?this.wrapperInitData.length=0:this.wrapperInitData=[],this._isInTransaction=!1},_isInTransaction:!1,getTransactionWrappers:null,isInTransaction:function(){return!!this._isInTransaction},perform:function(t,e,n,i,o,a,u,c){this.isInTransaction()&&r("27");var s,l;try{this._isInTransaction=!0,s=!0,this.initializeAll(0),l=t.call(e,n,i,o,a,u,c),s=!1}finally{try{if(s)try{this.closeAll(0)}catch(t){}else this.closeAll(0)}finally{this._isInTransaction=!1}}return l},initializeAll:function(t){for(var e=this.transactionWrappers,n=t;n<e.length;n++){var r=e[n];try{this.wrapperInitData[n]=i,this.wrapperInitData[n]=r.initialize?r.initialize.call(this):null}finally{if(this.wrapperInitData[n]===i)try{this.initializeAll(n+1)}catch(t){}}}},closeAll:function(t){this.isInTransaction()||r("28");for(var e=this.transactionWrappers,n=t;n<e.length;n++){var o,a=e[n],u=this.wrapperInitData[n];try{o=!0,u!==i&&a.close&&a.close.call(this,u),o=!1}finally{if(o)try{this.closeAll(n+1)}catch(t){}}}this.wrapperInitData.length=0}};t.exports=o},function(t,e,n){"use strict";function r(t){var e=""+t,n=o.exec(e);if(!n)return e;var r,i="",a=0,u=0;for(a=n.index;a<e.length;a++){switch(e.charCodeAt(a)){case 34:r="&quot;";break;case 38:r="&amp;";break;case 39:r="&#x27;";break;case 60:r="&lt;";break;case 62:r="&gt;";break;default:continue}u!==a&&(i+=e.substring(u,a)),u=a+1,i+=r}return u!==a?i+e.substring(u,a):i}function i(t){return"boolean"==typeof t||"number"==typeof t?""+t:r(t)}var o=/["'&<>]/;t.exports=i},function(t,e,n){"use strict";var r,i=n(6),o=n(83),a=/^[ \r\n\t\f]/,u=/<(!--|link|noscript|meta|script|style)[ \r\n\t\f\/>]/,c=n(91),s=c(function(t,e){if(t.namespaceURI!==o.svg||"innerHTML"in t)t.innerHTML=e;else{r=r||document.createElement("div"),r.innerHTML="<svg>"+e+"</svg>";for(var n=r.firstChild;n.firstChild;)t.appendChild(n.firstChild)}});if(i.canUseDOM){var l=document.createElement("div");l.innerHTML=" ",""===l.innerHTML&&(s=function(t,e){if(t.parentNode&&t.parentNode.replaceChild(t,t),a.test(e)||"<"===e[0]&&u.test(e)){t.innerHTML=String.fromCharCode(65279)+e;var n=t.firstChild;1===n.data.length?t.removeChild(n):n.deleteData(0,1)}else t.innerHTML=e}),l=null}t.exports=s},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default={colors:{RdBu:["rgb(255, 13, 87)","rgb(30, 136, 229)"],GnPR:["rgb(24, 196, 93)","rgb(124, 82, 255)"],CyPU:["#0099C6","#990099"],PkYg:["#DD4477","#66AA00"],DrDb:["#B82E2E","#316395"],LpLb:["#994499","#22AA99"],YlDp:["#AAAA11","#6633CC"],OrId:["#E67300","#3E0099"]},gray:"#777"}},function(t,e,n){"use strict";var r=n(28);e.a=function(t,e,n){if(null==n&&(n=r.a),i=t.length){if((e=+e)<=0||i<2)return+n(t[0],0,t);if(e>=1)return+n(t[i-1],i-1,t);var i,o=(i-1)*e,a=Math.floor(o),u=+n(t[a],a,t);return u+(+n(t[a+1],a+1,t)-u)*(o-a)}}},function(t,e,n){"use strict";function r(){}function i(t,e){var n=new r;if(t instanceof r)t.each(function(t,e){n.set(e,t)});else if(Array.isArray(t)){var i,o=-1,a=t.length;if(null==e)for(;++o<a;)n.set(o,t[o]);else for(;++o<a;)n.set(e(i=t[o],o,t),i)}else if(t)for(var u in t)n.set(u,t[u]);return n}n.d(e,"b",function(){return o});var o="$";r.prototype=i.prototype={constructor:r,has:function(t){return o+t in this},get:function(t){return this[o+t]},set:function(t,e){return this[o+t]=e,this},remove:function(t){var e=o+t;return e in this&&delete this[e]},clear:function(){for(var t in this)t[0]===o&&delete this[t]},keys:function(){var t=[];for(var e in this)e[0]===o&&t.push(e.slice(1));return t},values:function(){var t=[];for(var e in this)e[0]===o&&t.push(this[e]);return t},entries:function(){var t=[];for(var e in this)e[0]===o&&t.push({key:e.slice(1),value:this[e]});return t},size:function(){var t=0;for(var e in this)e[0]===o&&++t;return t},empty:function(){for(var t in this)if(t[0]===o)return!1;return!0},each:function(t){for(var e in this)e[0]===o&&t(this[e],e.slice(1),this)}},e.a=i},function(t,e,n){"use strict";function r(){}function i(t){var e;return t=(t+"").trim().toLowerCase(),(e=x.exec(t))?(e=parseInt(e[1],16),new s(e>>8&15|e>>4&240,e>>4&15|240&e,(15&e)<<4|15&e,1)):(e=w.exec(t))?o(parseInt(e[1],16)):(e=C.exec(t))?new s(e[1],e[2],e[3],1):(e=k.exec(t))?new s(255*e[1]/100,255*e[2]/100,255*e[3]/100,1):(e=E.exec(t))?a(e[1],e[2],e[3],e[4]):(e=M.exec(t))?a(255*e[1]/100,255*e[2]/100,255*e[3]/100,e[4]):(e=T.exec(t))?l(e[1],e[2]/100,e[3]/100,1):(e=S.exec(t))?l(e[1],e[2]/100,e[3]/100,e[4]):N.hasOwnProperty(t)?o(N[t]):"transparent"===t?new s(NaN,NaN,NaN,0):null}function o(t){return new s(t>>16&255,t>>8&255,255&t,1)}function a(t,e,n,r){return r<=0&&(t=e=n=NaN),new s(t,e,n,r)}function u(t){return t instanceof r||(t=i(t)),t?(t=t.rgb(),new s(t.r,t.g,t.b,t.opacity)):new s}function c(t,e,n,r){return 1===arguments.length?u(t):new s(t,e,n,null==r?1:r)}function s(t,e,n,r){this.r=+t,this.g=+e,this.b=+n,this.opacity=+r}function l(t,e,n,r){return r<=0?t=e=n=NaN:n<=0||n>=1?t=e=NaN:e<=0&&(t=NaN),new h(t,e,n,r)}function f(t){if(t instanceof h)return new h(t.h,t.s,t.l,t.opacity);if(t instanceof r||(t=i(t)),!t)return new h;if(t instanceof h)return t;t=t.rgb();var e=t.r/255,n=t.g/255,o=t.b/255,a=Math.min(e,n,o),u=Math.max(e,n,o),c=NaN,s=u-a,l=(u+a)/2;return s?(c=e===u?(n-o)/s+6*(n<o):n===u?(o-e)/s+2:(e-n)/s+4,s/=l<.5?u+a:2-u-a,c*=60):s=l>0&&l<1?0:c,new h(c,s,l,t.opacity)}function p(t,e,n,r){return 1===arguments.length?f(t):new h(t,e,n,null==r?1:r)}function h(t,e,n,r){this.h=+t,this.s=+e,this.l=+n,this.opacity=+r}function d(t,e,n){return 255*(t<60?e+(n-e)*t/60:t<180?n:t<240?e+(n-e)*(240-t)/60:e)}e.f=r,n.d(e,"h",function(){return g}),n.d(e,"g",function(){return m}),e.a=i,e.e=u,e.b=c,e.d=s,e.c=p;var v=n(62),g=.7,m=1/g,y="\\s*([+-]?\\d+)\\s*",_="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",b="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",x=/^#([0-9a-f]{3})$/,w=/^#([0-9a-f]{6})$/,C=new RegExp("^rgb\\("+[y,y,y]+"\\)$"),k=new RegExp("^rgb\\("+[b,b,b]+"\\)$"),E=new RegExp("^rgba\\("+[y,y,y,_]+"\\)$"),M=new RegExp("^rgba\\("+[b,b,b,_]+"\\)$"),T=new RegExp("^hsl\\("+[_,b,b]+"\\)$"),S=new RegExp("^hsla\\("+[_,b,b,_]+"\\)$"),N={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};n.i(v.a)(r,i,{displayable:function(){return this.rgb().displayable()},toString:function(){return this.rgb()+""}}),n.i(v.a)(s,c,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},rgb:function(){return this},displayable:function(){return 0<=this.r&&this.r<=255&&0<=this.g&&this.g<=255&&0<=this.b&&this.b<=255&&0<=this.opacity&&this.opacity<=1},toString:function(){var t=this.opacity;return t=isNaN(t)?1:Math.max(0,Math.min(1,t)),(1===t?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(1===t?")":", "+t+")")}})),n.i(v.a)(h,p,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new h(this.h,this.s,this.l*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new h(this.h,this.s,this.l*t,this.opacity)},rgb:function(){var t=this.h%360+360*(this.h<0),e=isNaN(t)||isNaN(this.s)?0:this.s,n=this.l,r=n+(n<.5?n:1-n)*e,i=2*n-r;return new s(d(t>=240?t-240:t+120,i,r),d(t,i,r),d(t<120?t+240:t-120,i,r),this.opacity)},displayable:function(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&0<=this.l&&this.l<=1&&0<=this.opacity&&this.opacity<=1}}))},function(t,e,n){"use strict";function r(t,e){var n=Object.create(t.prototype);for(var r in e)n[r]=e[r];return n}e.b=r,e.a=function(t,e,n){t.prototype=e.prototype=n,n.constructor=t}},function(t,e,n){"use strict";e.a=function(t,e){if((n=(t=e?t.toExponential(e-1):t.toExponential()).indexOf("e"))<0)return null;var n,r=t.slice(0,n);return[r.length>1?r[0]+r.slice(2):r,+t.slice(n+1)]}},function(t,e,n){"use strict";function r(t,e,n,r,i){var o=t*t,a=o*t;return((1-3*t+3*o-a)*e+(4-6*o+3*a)*n+(1+3*t+3*o-3*a)*r+a*i)/6}e.b=r,e.a=function(t){var e=t.length-1;return function(n){var i=n<=0?n=0:n>=1?(n=1,e-1):Math.floor(n*e),o=t[i],a=t[i+1],u=i>0?t[i-1]:2*o-a,c=i<e-1?t[i+2]:2*a-o;return r((n-i/e)*e,u,o,a,c)}}},function(t,e,n){"use strict";var r=n(10),i=n(123),o=n(118),a=n(121),u=n(43),c=n(122),s=n(124),l=n(120);e.a=function(t,e){var f,p=typeof e;return null==e||"boolean"===p?n.i(l.a)(e):("number"===p?u.a:"string"===p?(f=n.i(r.color)(e))?(e=f,i.a):s.a:e instanceof r.color?i.a:e instanceof Date?a.a:Array.isArray(e)?o.a:"function"!=typeof e.valueOf&&"function"!=typeof e.toString||isNaN(e)?c.a:u.a)(t,e)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(239);n.d(e,"scaleBand",function(){return r.a}),n.d(e,"scalePoint",function(){return r.b});var i=n(245);n.d(e,"scaleIdentity",function(){return i.a});var o=n(34);n.d(e,"scaleLinear",function(){return o.a});var a=n(246);n.d(e,"scaleLog",function(){return a.a});var u=n(127);n.d(e,"scaleOrdinal",function(){return u.a}),n.d(e,"scaleImplicit",function(){return u.b});var c=n(247);n.d(e,"scalePow",function(){return c.a}),n.d(e,"scaleSqrt",function(){return c.b});var s=n(248);n.d(e,"scaleQuantile",function(){return s.a});var l=n(249);n.d(e,"scaleQuantize",function(){return l.a});var f=n(252);n.d(e,"scaleThreshold",function(){return f.a});var p=n(128);n.d(e,"scaleTime",function(){return p.a});var h=n(254);n.d(e,"scaleUtc",function(){return h.a});var d=n(240);n.d(e,"schemeCategory10",function(){return d.a});var v=n(242);n.d(e,"schemeCategory20b",function(){return v.a});var g=n(243);n.d(e,"schemeCategory20c",function(){return g.a});var m=n(241);n.d(e,"schemeCategory20",function(){return m.a});var y=n(244);n.d(e,"interpolateCubehelixDefault",function(){return y.a});var _=n(250);n.d(e,"interpolateRainbow",function(){return _.a}),n.d(e,"interpolateWarm",function(){return _.b}),n.d(e,"interpolateCool",function(){return _.c});var b=n(255);n.d(e,"interpolateViridis",function(){return b.a}),n.d(e,"interpolateMagma",function(){return b.b}),n.d(e,"interpolateInferno",function(){return b.c}),n.d(e,"interpolatePlasma",function(){return b.d});var x=n(251);n.d(e,"scaleSequential",function(){return x.a})},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(69);e.a=function(t){var e=t+="",n=e.indexOf(":");return n>=0&&"xmlns"!==(e=t.slice(0,n))&&(t=t.slice(n+1)),r.a.hasOwnProperty(e)?{space:r.a[e],local:t}:t}},function(t,e,n){"use strict";n.d(e,"b",function(){return r});var r="http://www.w3.org/1999/xhtml";e.a={svg:"http://www.w3.org/2000/svg",xhtml:r,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"}},function(t,e,n){"use strict";function r(t,e,n){return t=i(t,e,n),function(e){var n=e.relatedTarget;n&&(n===this||8&n.compareDocumentPosition(this))||t.call(this,e)}}function i(t,e,n){return function(r){var i=l;l=r;try{t.call(this,this.__data__,e,n)}finally{l=i}}}function o(t){return t.trim().split(/^|\s+/).map(function(t){var e="",n=t.indexOf(".");return n>=0&&(e=t.slice(n+1),t=t.slice(0,n)),{type:t,name:e}})}function a(t){return function(){var e=this.__on;if(e){for(var n,r=0,i=-1,o=e.length;r<o;++r)n=e[r],t.type&&n.type!==t.type||n.name!==t.name?e[++i]=n:this.removeEventListener(n.type,n.listener,n.capture);++i?e.length=i:delete this.__on}}}function u(t,e,n){var o=s.hasOwnProperty(t.type)?r:i;return function(r,i,a){var u,c=this.__on,s=o(e,i,a);if(c)for(var l=0,f=c.length;l<f;++l)if((u=c[l]).type===t.type&&u.name===t.name)return this.removeEventListener(u.type,u.listener,u.capture),this.addEventListener(u.type,u.listener=s,u.capture=n),void(u.value=e);this.addEventListener(t.type,s,n),u={type:t.type,name:t.name,value:e,listener:s,capture:n},c?c.push(u):this.__on=[u]}}function c(t,e,n,r){var i=l;t.sourceEvent=l,l=t;try{return e.apply(n,r)}finally{l=i}}n.d(e,"a",function(){return l}),e.b=c;var s={},l=null;if("undefined"!=typeof document){"onmouseenter"in document.documentElement||(s={mouseenter:"mouseover",mouseleave:"mouseout"})}e.c=function(t,e,n){var r,i,c=o(t+""),s=c.length;{if(!(arguments.length<2)){for(l=e?u:a,null==n&&(n=!1),r=0;r<s;++r)this.each(l(c[r],e,n));return this}var l=this.node().__on;if(l)for(var f,p=0,h=l.length;p<h;++p)for(r=0,f=l[p];r<s;++r)if((i=c[r]).type===f.type&&i.name===f.name)return f.value}}},function(t,e,n){"use strict";function r(){}e.a=function(t){return null==t?r:function(){return this.querySelector(t)}}},function(t,e,n){"use strict";var r=n(70);e.a=function(){for(var t,e=r.a;t=e.sourceEvent;)e=t;return e}},function(t,e,n){"use strict";e.a=function(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}},function(t,e,n){"use strict";function r(t,e,n){var r=t._x1,i=t._y1,a=t._x2,u=t._y2;if(t._l01_a>o.a){var c=2*t._l01_2a+3*t._l01_a*t._l12_a+t._l12_2a,s=3*t._l01_a*(t._l01_a+t._l12_a);r=(r*c-t._x0*t._l12_2a+t._x2*t._l01_2a)/s,i=(i*c-t._y0*t._l12_2a+t._y2*t._l01_2a)/s}if(t._l23_a>o.a){var l=2*t._l23_2a+3*t._l23_a*t._l12_a+t._l12_2a,f=3*t._l23_a*(t._l23_a+t._l12_a);a=(a*l+t._x1*t._l23_2a-e*t._l12_2a)/f,u=(u*l+t._y1*t._l23_2a-n*t._l12_2a)/f}t._context.bezierCurveTo(r,i,a,u,t._x2,t._y2)}function i(t,e){this._context=t,this._alpha=e}e.b=r;var o=n(35),a=n(48);i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:this.point(this._x2,this._y2)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){if(t=+t,e=+e,this._point){var n=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(n*n+i*i,this._alpha))}switch(this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3;default:r(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new i(t,e):new a.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";var r=n(32),i=n(17),o=n(49),a=n(77);e.a=function(){function t(t){var i,o,a,p=t.length,h=!1;for(null==s&&(f=l(a=n.i(r.a)())),i=0;i<=p;++i)!(i<p&&c(o=t[i],i,t))===h&&((h=!h)?f.lineStart():f.lineEnd()),h&&f.point(+e(o,i,t),+u(o,i,t));if(a)return f=null,a+""||null}var e=a.a,u=a.b,c=n.i(i.a)(!0),s=null,l=o.a,f=null;return t.x=function(r){return arguments.length?(e="function"==typeof r?r:n.i(i.a)(+r),t):e},t.y=function(e){return arguments.length?(u="function"==typeof e?e:n.i(i.a)(+e),t):u},t.defined=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(!!e),t):c},t.curve=function(e){return arguments.length?(l=e,null!=s&&(f=l(s)),t):l},t.context=function(e){return arguments.length?(null==e?s=f=null:f=l(s=e),t):s},t}},function(t,e,n){"use strict";function r(t){for(var e,n=0,r=-1,i=t.length;++r<i;)(e=+t[r][1])&&(n+=e);return n}e.b=r;var i=n(37);e.a=function(t){var e=t.map(r);return n.i(i.a)(t).sort(function(t,n){return e[t]-e[n]})}},function(t,e,n){"use strict";function r(t){return t[0]}function i(t){return t[1]}e.a=r,e.b=i},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(79);n.d(e,"timeFormatDefaultLocale",function(){return r.a}),n.d(e,"timeFormat",function(){return r.b}),n.d(e,"timeParse",function(){return r.c}),n.d(e,"utcFormat",function(){return r.d}),n.d(e,"utcParse",function(){return r.e});var i=n(152);n.d(e,"timeFormatLocale",function(){return i.a});var o=n(151);n.d(e,"isoFormat",function(){return o.a});var a=n(314);n.d(e,"isoParse",function(){return a.a})},function(t,e,n){"use strict";function r(t){return i=n.i(s.a)(t),o=i.format,a=i.parse,u=i.utcFormat,c=i.utcParse,i}n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a}),n.d(e,"d",function(){return u}),n.d(e,"e",function(){return c}),e.a=r;var i,o,a,u,c,s=n(152);r({dateTime:"%x, %X",date:"%-m/%-d/%Y",time:"%-I:%M:%S %p",periods:["AM","PM"],days:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],shortDays:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],months:["January","February","March","April","May","June","July","August","September","October","November","December"],shortMonths:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})},function(t,e,n){"use strict";var r=(n(5),n(317));n.d(e,"v",function(){return r.a}),n.d(e,"p",function(){return r.a});var i=n(320);n.d(e,"u",function(){return i.a}),n.d(e,"o",function(){return i.a});var o=n(318);n.d(e,"t",function(){return o.a});var a=n(316);n.d(e,"s",function(){return a.a});var u=n(315);n.d(e,"d",function(){return u.a});var c=n(327);n.d(e,"r",function(){return c.a}),n.d(e,"f",function(){return c.a}),n.d(e,"c",function(){return c.b}),n.d(e,"g",function(){return c.c});var s=n(319);n.d(e,"q",function(){return s.a});var l=n(328);n.d(e,"e",function(){return l.a});var f=n(323);n.d(e,"n",function(){return f.a});var p=n(322);n.d(e,"m",function(){return p.a});var h=n(321);n.d(e,"b",function(){return h.a});var d=n(325);n.d(e,"l",function(){return d.a}),n.d(e,"i",function(){return d.a}),n.d(e,"a",function(){return d.b}),n.d(e,"j",function(){return d.c});var v=n(324);n.d(e,"k",function(){return v.a});var g=n(326);n.d(e,"h",function(){return g.a})},function(t,e,n){"use strict";function r(t,e){return t===e?0!==t||0!==e||1/t==1/e:t!==t&&e!==e}function i(t,e){if(r(t,e))return!0;if("object"!=typeof t||null===t||"object"!=typeof e||null===e)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(var a=0;a<n.length;a++)if(!o.call(e,n[a])||!r(t[n[a]],e[n[a]]))return!1;return!0}var o=Object.prototype.hasOwnProperty;t.exports=i},function(t,e,n){"use strict";function r(t,e){return Array.isArray(e)&&(e=e[1]),e?e.nextSibling:t.firstChild}function i(t,e,n){l.insertTreeBefore(t,e,n)}function o(t,e,n){Array.isArray(e)?u(t,e[0],e[1],n):v(t,e,n)}function a(t,e){if(Array.isArray(e)){var n=e[1];e=e[0],c(t,e,n),t.removeChild(n)}t.removeChild(e)}function u(t,e,n,r){for(var i=e;;){var o=i.nextSibling;if(v(t,i,r),i===n)break;i=o}}function c(t,e,n){for(;;){var r=e.nextSibling;if(r===n)break;t.removeChild(r)}}function s(t,e,n){var r=t.parentNode,i=t.nextSibling;i===e?n&&v(r,document.createTextNode(n),i):n?(d(i,n),c(r,i,e)):c(r,t,e)}var l=n(20),f=n(350),p=(n(4),n(9),n(91)),h=n(57),d=n(176),v=p(function(t,e,n){t.insertBefore(e,n)}),g=f.dangerouslyReplaceNodeWithMarkup,m={dangerouslyReplaceNodeWithMarkup:g,replaceDelimitedText:s,processUpdates:function(t,e){for(var n=0;n<e.length;n++){var u=e[n];switch(u.type){case"INSERT_MARKUP":i(t,u.content,r(t,u.afterNode));break;case"MOVE_EXISTING":o(t,u.fromNode,r(t,u.afterNode));break;case"SET_MARKUP":h(t,u.content);break;case"TEXT_CONTENT":d(t,u.content);break;case"REMOVE_NODE":a(t,u.fromNode)}}}};t.exports=m},function(t,e,n){"use strict";var r={html:"http://www.w3.org/1999/xhtml",mathml:"http://www.w3.org/1998/Math/MathML",svg:"http://www.w3.org/2000/svg"};t.exports=r},function(t,e,n){"use strict";function r(){if(u)for(var t in c){var e=c[t],n=u.indexOf(t);if(n>-1||a("96",t),!s.plugins[n]){e.extractEvents||a("97",t),s.plugins[n]=e;var r=e.eventTypes;for(var o in r)i(r[o],e,o)||a("98",o,t)}}}function i(t,e,n){s.eventNameDispatchConfigs.hasOwnProperty(n)&&a("99",n),s.eventNameDispatchConfigs[n]=t;var r=t.phasedRegistrationNames;if(r){for(var i in r)if(r.hasOwnProperty(i)){var u=r[i];o(u,e,n)}return!0}return!!t.registrationName&&(o(t.registrationName,e,n),!0)}function o(t,e,n){s.registrationNameModules[t]&&a("100",t),s.registrationNameModules[t]=e,s.registrationNameDependencies[t]=e.eventTypes[n].dependencies}var a=n(1),u=(n(0),null),c={},s={plugins:[],eventNameDispatchConfigs:{},registrationNameModules:{},registrationNameDependencies:{},possibleRegistrationNames:null,injectEventPluginOrder:function(t){u&&a("101"),u=Array.prototype.slice.call(t),r()},injectEventPluginsByName:function(t){var e=!1;for(var n in t)if(t.hasOwnProperty(n)){var i=t[n];c.hasOwnProperty(n)&&c[n]===i||(c[n]&&a("102",n),c[n]=i,e=!0)}e&&r()},getPluginModuleForEvent:function(t){var e=t.dispatchConfig;if(e.registrationName)return s.registrationNameModules[e.registrationName]||null;if(void 0!==e.phasedRegistrationNames){var n=e.phasedRegistrationNames;for(var r in n)if(n.hasOwnProperty(r)){var i=s.registrationNameModules[n[r]];if(i)return i}}return null},_resetEventPlugins:function(){u=null;for(var t in c)c.hasOwnProperty(t)&&delete c[t];s.plugins.length=0;var e=s.eventNameDispatchConfigs;for(var n in e)e.hasOwnProperty(n)&&delete e[n];var r=s.registrationNameModules;for(var i in r)r.hasOwnProperty(i)&&delete r[i]}};t.exports=s},function(t,e,n){"use strict";function r(t){var e={"=":"=0",":":"=2"};return"$"+(""+t).replace(/[=:]/g,function(t){return e[t]})}function i(t){var e=/(=0|=2)/g,n={"=0":"=","=2":":"};return(""+("."===t[0]&&"$"===t[1]?t.substring(2):t.substring(1))).replace(e,function(t){return n[t]})}var o={escape:r,unescape:i};t.exports=o},function(t,e,n){"use strict";function r(t){null!=t.checkedLink&&null!=t.valueLink&&u("87")}function i(t){r(t),(null!=t.value||null!=t.onChange)&&u("88")}function o(t){r(t),(null!=t.checked||null!=t.onChange)&&u("89")}function a(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}var u=n(1),c=n(380),s=n(157),l=n(26),f=s(l.isValidElement),p=(n(0),n(2),{button:!0,checkbox:!0,image:!0,hidden:!0,radio:!0,reset:!0,submit:!0}),h={value:function(t,e,n){return!t[e]||p[t.type]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`.")},checked:function(t,e,n){return!t[e]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`.")},onChange:f.func},d={},v={checkPropTypes:function(t,e,n){for(var r in h){if(h.hasOwnProperty(r))var i=h[r](e,r,t,"prop",null,c);if(i instanceof Error&&!(i.message in d)){d[i.message]=!0;a(n)}}},getValue:function(t){return t.valueLink?(i(t),t.valueLink.value):t.value},getChecked:function(t){return t.checkedLink?(o(t),t.checkedLink.value):t.checked},executeOnChange:function(t,e){return t.valueLink?(i(t),t.valueLink.requestChange(e.target.value)):t.checkedLink?(o(t),t.checkedLink.requestChange(e.target.checked)):t.onChange?t.onChange.call(void 0,e):void 0}};t.exports=v},function(t,e,n){"use strict";var r=n(1),i=(n(0),!1),o={replaceNodeWithMarkup:null,processChildrenUpdates:null,injection:{injectEnvironment:function(t){i&&r("104"),o.replaceNodeWithMarkup=t.replaceNodeWithMarkup,o.processChildrenUpdates=t.processChildrenUpdates,i=!0}}};t.exports=o},function(t,e,n){"use strict";function r(t,e,n){try{e(n)}catch(t){null===i&&(i=t)}}var i=null,o={invokeGuardedCallback:r,invokeGuardedCallbackWithCatch:r,rethrowCaughtError:function(){if(i){var t=i;throw i=null,t}}};t.exports=o},function(t,e,n){"use strict";function r(t){c.enqueueUpdate(t)}function i(t){var e=typeof t;if("object"!==e)return e;var n=t.constructor&&t.constructor.name||e,r=Object.keys(t);return r.length>0&&r.length<20?n+" (keys: "+r.join(", ")+")":n}function o(t,e){var n=u.get(t);if(!n){return null}return n}var a=n(1),u=(n(15),n(39)),c=(n(9),n(12)),s=(n(0),n(2),{isMounted:function(t){var e=u.get(t);return!!e&&!!e._renderedComponent},enqueueCallback:function(t,e,n){s.validateCallback(e,n);var i=o(t);if(!i)return null;i._pendingCallbacks?i._pendingCallbacks.push(e):i._pendingCallbacks=[e],r(i)},enqueueCallbackInternal:function(t,e){t._pendingCallbacks?t._pendingCallbacks.push(e):t._pendingCallbacks=[e],r(t)},enqueueForceUpdate:function(t){var e=o(t,"forceUpdate");e&&(e._pendingForceUpdate=!0,r(e))},enqueueReplaceState:function(t,e,n){var i=o(t,"replaceState");i&&(i._pendingStateQueue=[e],i._pendingReplaceState=!0,void 0!==n&&null!==n&&(s.validateCallback(n,"replaceState"),i._pendingCallbacks?i._pendingCallbacks.push(n):i._pendingCallbacks=[n]),r(i))},enqueueSetState:function(t,e){var n=o(t,"setState");if(n){(n._pendingStateQueue||(n._pendingStateQueue=[])).push(e),r(n)}},enqueueElementInternal:function(t,e,n){t._pendingElement=e,t._context=n,r(t)},validateCallback:function(t,e){t&&"function"!=typeof t&&a("122",e,i(t))}});t.exports=s},function(t,e,n){"use strict";var r={currentScrollLeft:0,currentScrollTop:0,refreshScrollValues:function(t){r.currentScrollLeft=t.x,r.currentScrollTop=t.y}};t.exports=r},function(t,e,n){"use strict";var r=function(t){return"undefined"!=typeof MSApp&&MSApp.execUnsafeLocalFunction?function(e,n,r,i){MSApp.execUnsafeLocalFunction(function(){return t(e,n,r,i)})}:t};t.exports=r},function(t,e,n){"use strict";function r(t){var e,n=t.keyCode;return"charCode"in t?0===(e=t.charCode)&&13===n&&(e=13):e=n,e>=32||13===e?e:0}t.exports=r},function(t,e,n){"use strict";function r(t){var e=this,n=e.nativeEvent;if(n.getModifierState)return n.getModifierState(t);var r=o[t];return!!r&&!!n[r]}function i(t){return r}var o={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};t.exports=i},function(t,e,n){"use strict";function r(t){var e=t.target||t.srcElement||window;return e.correspondingUseElement&&(e=e.correspondingUseElement),3===e.nodeType?e.parentNode:e}t.exports=r},function(t,e,n){"use strict";/**
 * Checks if an event is supported in the current execution environment.
 *
 * NOTE: This will not work correctly for non-generic events such as `change`,
 * `reset`, `load`, `error`, and `select`.
 *
 * Borrows from Modernizr.
 *
 * @param {string} eventNameSuffix Event name, e.g. "click".
 * @param {?boolean} capture Check if the capture phase is supported.
 * @return {boolean} True if the event is supported.
 * @internal
 * @license Modernizr 3.0.0pre (Custom Build) | MIT
 */
function r(t,e){if(!o.canUseDOM||e&&!("addEventListener"in document))return!1;var n="on"+t,r=n in document;if(!r){var a=document.createElement("div");a.setAttribute(n,"return;"),r="function"==typeof a[n]}return!r&&i&&"wheel"===t&&(r=document.implementation.hasFeature("Events.wheel","3.0")),r}var i,o=n(6);o.canUseDOM&&(i=document.implementation&&document.implementation.hasFeature&&!0!==document.implementation.hasFeature("","")),t.exports=r},function(t,e,n){"use strict";function r(t,e){var n=null===t||!1===t,r=null===e||!1===e;if(n||r)return n===r;var i=typeof t,o=typeof e;return"string"===i||"number"===i?"string"===o||"number"===o:"object"===o&&t.type===e.type&&t.key===e.key}t.exports=r},function(t,e,n){"use strict";var r=(n(3),n(11)),i=(n(2),r);t.exports=i},function(t,e){var n;n=function(){return this}();try{n=n||Function("return this")()||(0,eval)("this")}catch(t){"object"==typeof window&&(n=window)}t.exports=n},function(t,e){t.exports=function(t){return t.webpackPolyfill||(t.deprecate=function(){},t.paths=[],t.children||(t.children=[]),Object.defineProperty(t,"loaded",{enumerable:!0,get:function(){return t.l}}),Object.defineProperty(t,"id",{enumerable:!0,get:function(){return t.i}}),t.webpackPolyfill=1),t}},function(t,e,n){"use strict";n.d(e,"b",function(){return i}),n.d(e,"a",function(){return o});var r=Array.prototype,i=r.slice,o=r.map},function(t,e,n){"use strict";n.d(e,"b",function(){return a}),n.d(e,"c",function(){return u});var r=n(19),i=n(102),o=n.i(i.a)(r.a),a=o.right,u=o.left;e.a=a},function(t,e,n){"use strict";function r(t){return function(e,r){return n.i(i.a)(t(e),r)}}var i=n(19);e.a=function(t){return 1===t.length&&(t=r(t)),{left:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)<0?r=o+1:i=o}return r},right:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)>0?i=o:r=o+1}return r}}}},function(t,e,n){"use strict";var r=n(111);e.a=function(t,e){var i=n.i(r.a)(t,e);return i?Math.sqrt(i):i}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i,o=t.length,a=-1;if(null==e){for(;++a<o;)if(null!=(n=t[a])&&n>=n)for(r=i=n;++a<o;)null!=(n=t[a])&&(r>n&&(r=n),i<n&&(i=n))}else for(;++a<o;)if(null!=(n=e(t[a],a,t))&&n>=n)for(r=i=n;++a<o;)null!=(n=e(t[a],a,t))&&(r>n&&(r=n),i<n&&(i=n));return[r,i]}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i=t.length,o=-1;if(null==e){for(;++o<i;)if(null!=(n=t[o])&&n>=n)for(r=n;++o<i;)null!=(n=t[o])&&r>n&&(r=n)}else for(;++o<i;)if(null!=(n=e(t[o],o,t))&&n>=n)for(r=n;++o<i;)null!=(n=e(t[o],o,t))&&r>n&&(r=n);return r}},function(t,e,n){"use strict";function r(t,e){return[t,e]}e.b=r,e.a=function(t,e){null==e&&(e=r);for(var n=0,i=t.length-1,o=t[0],a=new Array(i<0?0:i);n<i;)a[n]=e(o,o=t[++n]);return a}},function(t,e,n){"use strict";e.a=function(t,e,n){t=+t,e=+e,n=(i=arguments.length)<2?(e=t,t=0,1):i<3?1:+n;for(var r=-1,i=0|Math.max(0,Math.ceil((e-t)/n)),o=new Array(i);++r<i;)o[r]=t+r*n;return o}},function(t,e,n){"use strict";e.a=function(t){return Math.ceil(Math.log(t.length)/Math.LN2)+1}},function(t,e,n){"use strict";function r(t,e,n){var r=(e-t)/Math.max(0,n),i=Math.floor(Math.log(r)/Math.LN10),c=r/Math.pow(10,i);return i>=0?(c>=o?10:c>=a?5:c>=u?2:1)*Math.pow(10,i):-Math.pow(10,-i)/(c>=o?10:c>=a?5:c>=u?2:1)}function i(t,e,n){var r=Math.abs(e-t)/Math.max(0,n),i=Math.pow(10,Math.floor(Math.log(r)/Math.LN10)),c=r/i;return c>=o?i*=10:c>=a?i*=5:c>=u&&(i*=2),e<t?-i:i}e.b=r,e.c=i;var o=Math.sqrt(50),a=Math.sqrt(10),u=Math.sqrt(2);e.a=function(t,e,n){var i,o,a,u,c=-1;if(e=+e,t=+t,n=+n,t===e&&n>0)return[t];if((i=e<t)&&(o=t,t=e,e=o),0===(u=r(t,e,n))||!isFinite(u))return[];if(u>0)for(t=Math.ceil(t/u),e=Math.floor(e/u),a=new Array(o=Math.ceil(e-t+1));++c<o;)a[c]=(t+c)*u;else for(t=Math.floor(t*u),e=Math.ceil(e*u),a=new Array(o=Math.ceil(t-e+1));++c<o;)a[c]=(t-c)/u;return i&&a.reverse(),a}},function(t,e,n){"use strict";function r(t){return t.length}var i=n(105);e.a=function(t){if(!(u=t.length))return[];for(var e=-1,o=n.i(i.a)(t,r),a=new Array(o);++e<o;)for(var u,c=-1,s=a[e]=new Array(u);++c<u;)s[c]=t[c][e];return a}},function(t,e,n){"use strict";var r=n(28);e.a=function(t,e){var i,o,a=t.length,u=0,c=-1,s=0,l=0;if(null==e)for(;++c<a;)isNaN(i=n.i(r.a)(t[c]))||(o=i-s,s+=o/++u,l+=o*(i-s));else for(;++c<a;)isNaN(i=n.i(r.a)(e(t[c],c,t)))||(o=i-s,s+=o/++u,l+=o*(i-s));if(u>1)return l/(u-1)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(209);n.d(e,"axisTop",function(){return r.a}),n.d(e,"axisRight",function(){return r.b}),n.d(e,"axisBottom",function(){return r.c}),n.d(e,"axisLeft",function(){return r.d})},function(t,e,n){"use strict";n.d(e,"b",function(){return r}),n.d(e,"a",function(){return i});var r=Math.PI/180,i=180/Math.PI},function(t,e,n){"use strict";n.d(e,"b",function(){return r});var r,i=n(63);e.a=function(t,e){var o=n.i(i.a)(t,e);if(!o)return t+"";var a=o[0],u=o[1],c=u-(r=3*Math.max(-8,Math.min(8,Math.floor(u/3))))+1,s=a.length;return c===s?a:c>s?a+new Array(c-s+1).join("0"):c>0?a.slice(0,c)+"."+a.slice(c):"0."+new Array(1-c).join("0")+n.i(i.a)(t,Math.max(0,e+c-1))[0]}},function(t,e,n){"use strict";function r(t){return new i(t)}function i(t){if(!(e=a.exec(t)))throw new Error("invalid format: "+t);var e,n=e[1]||" ",r=e[2]||">",i=e[3]||"-",u=e[4]||"",c=!!e[5],s=e[6]&&+e[6],l=!!e[7],f=e[8]&&+e[8].slice(1),p=e[9]||"";"n"===p?(l=!0,p="g"):o.a[p]||(p=""),(c||"0"===n&&"="===r)&&(c=!0,n="0",r="="),this.fill=n,this.align=r,this.sign=i,this.symbol=u,this.zero=c,this.width=s,this.comma=l,this.precision=f,this.type=p}e.a=r;var o=n(116),a=/^(?:(.)?([<>=^]))?([+\-\( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?([a-z%])?$/i;r.prototype=i.prototype,i.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(null==this.width?"":Math.max(1,0|this.width))+(this.comma?",":"")+(null==this.precision?"":"."+Math.max(0,0|this.precision))+this.type}},function(t,e,n){"use strict";var r=n(220),i=n(114),o=n(223);e.a={"":r.a,"%":function(t,e){return(100*t).toFixed(e)},b:function(t){return Math.round(t).toString(2)},c:function(t){return t+""},d:function(t){return Math.round(t).toString(10)},e:function(t,e){return t.toExponential(e)},f:function(t,e){return t.toFixed(e)},g:function(t,e){return t.toPrecision(e)},o:function(t){return Math.round(t).toString(8)},p:function(t,e){return n.i(o.a)(100*t,e)},r:o.a,s:i.a,X:function(t){return Math.round(t).toString(16).toUpperCase()},x:function(t){return Math.round(t).toString(16)}}},function(t,e,n){"use strict";var r=n(42),i=n(221),o=n(222),a=n(115),u=n(116),c=n(114),s=n(224),l=["y","z","a","f","p","n","µ","m","","k","M","G","T","P","E","Z","Y"];e.a=function(t){function e(t){function e(t){var e,n,a,u=x,s=w;if("c"===b)s=C(t)+s,t="";else{t=+t;var h=t<0;if(t=C(Math.abs(t),_),h&&0==+t&&(h=!1),u=(h?"("===o?o:"-":"-"===o||"("===o?"":o)+u,s=("s"===b?l[8+c.b/3]:"")+s+(h&&"("===o?")":""),k)for(e=-1,n=t.length;++e<n;)if(48>(a=t.charCodeAt(e))||a>57){s=(46===a?d+t.slice(e+1):t.slice(e))+s,t=t.slice(0,e);break}}y&&!f&&(t=p(t,1/0));var g=u.length+t.length+s.length,E=g<m?new Array(m-g+1).join(r):"";switch(y&&f&&(t=p(E+t,E.length?m-s.length:1/0),E=""),i){case"<":t=u+t+s+E;break;case"=":t=u+E+t+s;break;case"^":t=E.slice(0,g=E.length>>1)+u+t+s+E.slice(g);break;default:t=E+u+t+s}return v(t)}t=n.i(a.a)(t);var r=t.fill,i=t.align,o=t.sign,s=t.symbol,f=t.zero,m=t.width,y=t.comma,_=t.precision,b=t.type,x="$"===s?h[0]:"#"===s&&/[boxX]/.test(b)?"0"+b.toLowerCase():"",w="$"===s?h[1]:/[%p]/.test(b)?g:"",C=u.a[b],k=!b||/[defgprs%]/.test(b);return _=null==_?b?6:12:/[gprs]/.test(b)?Math.max(1,Math.min(21,_)):Math.max(0,Math.min(20,_)),e.toString=function(){return t+""},e}function f(t,i){var o=e((t=n.i(a.a)(t),t.type="f",t)),u=3*Math.max(-8,Math.min(8,Math.floor(n.i(r.a)(i)/3))),c=Math.pow(10,-u),s=l[8+u/3];return function(t){return o(c*t)+s}}var p=t.grouping&&t.thousands?n.i(i.a)(t.grouping,t.thousands):s.a,h=t.currency,d=t.decimal,v=t.numerals?n.i(o.a)(t.numerals):s.a,g=t.percent||"%";return{format:e,formatPrefix:f}}},function(t,e,n){"use strict";var r=n(65);e.a=function(t,e){var i,o=e?e.length:0,a=t?Math.min(o,t.length):0,u=new Array(a),c=new Array(o);for(i=0;i<a;++i)u[i]=n.i(r.a)(t[i],e[i]);for(;i<o;++i)c[i]=e[i];return function(t){for(i=0;i<a;++i)c[i]=u[i](t);return c}}},function(t,e,n){"use strict";var r=n(64);e.a=function(t){var e=t.length;return function(i){var o=Math.floor(((i%=1)<0?++i:i)*e),a=t[(o+e-1)%e],u=t[o%e],c=t[(o+1)%e],s=t[(o+2)%e];return n.i(r.b)((i-o/e)*e,a,u,c,s)}}},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";e.a=function(t,e){var n=new Date;return t=+t,e-=t,function(r){return n.setTime(t+e*r),n}}},function(t,e,n){"use strict";var r=n(65);e.a=function(t,e){var i,o={},a={};null!==t&&"object"==typeof t||(t={}),null!==e&&"object"==typeof e||(e={});for(i in e)i in t?o[i]=n.i(r.a)(t[i],e[i]):a[i]=e[i];return function(t){for(i in o)a[i]=o[i](t);return a}}},function(t,e,n){"use strict";function r(t){return function(e){var r,o,a=e.length,u=new Array(a),c=new Array(a),s=new Array(a);for(r=0;r<a;++r)o=n.i(i.rgb)(e[r]),u[r]=o.r||0,c[r]=o.g||0,s[r]=o.b||0;return u=t(u),c=t(c),s=t(s),o.opacity=1,function(t){return o.r=u(t),o.g=c(t),o.b=s(t),o+""}}}var i=n(10),o=n(64),a=n(119),u=n(31);e.a=function t(e){function r(t,e){var r=o((t=n.i(i.rgb)(t)).r,(e=n.i(i.rgb)(e)).r),a=o(t.g,e.g),c=o(t.b,e.b),s=n.i(u.a)(t.opacity,e.opacity);return function(e){return t.r=r(e),t.g=a(e),t.b=c(e),t.opacity=s(e),t+""}}var o=n.i(u.c)(e);return r.gamma=t,r}(1);r(o.a),r(a.a)},function(t,e,n){"use strict";function r(t){return function(){return t}}function i(t){return function(e){return t(e)+""}}var o=n(43),a=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,u=new RegExp(a.source,"g");e.a=function(t,e){var c,s,l,f=a.lastIndex=u.lastIndex=0,p=-1,h=[],d=[];for(t+="",e+="";(c=a.exec(t))&&(s=u.exec(e));)(l=s.index)>f&&(l=e.slice(f,l),h[p]?h[p]+=l:h[++p]=l),(c=c[0])===(s=s[0])?h[p]?h[p]+=s:h[++p]=s:(h[++p]=null,d.push({i:p,x:n.i(o.a)(c,s)})),f=u.lastIndex;return f<e.length&&(l=e.slice(f),h[p]?h[p]+=l:h[++p]=l),h.length<2?d[0]?i(d[0].x):r(e):(e=d.length,function(t){for(var n,r=0;r<e;++r)h[(n=d[r]).i]=n.x(t);return h.join("")})}},function(t,e,n){"use strict";e.a=function(t,e){t=t.slice();var n,r=0,i=t.length-1,o=t[r],a=t[i];return a<o&&(n=r,r=i,i=n,n=o,o=a,a=n),t[r]=e.floor(o),t[i]=e.ceil(a),t}},function(t,e,n){"use strict";e.a=function(t){return+t}},function(t,e,n){"use strict";function r(t){function e(e){var n=e+"",r=u.get(n);if(!r){if(s!==a)return s;u.set(n,r=c.push(e))}return t[(r-1)%t.length]}var u=n.i(i.a)(),c=[],s=a;return t=null==t?[]:o.b.call(t),e.domain=function(t){if(!arguments.length)return c.slice();c=[],u=n.i(i.a)();for(var r,o,a=-1,s=t.length;++a<s;)u.has(o=(r=t[a])+"")||u.set(o,c.push(r));return e},e.range=function(n){return arguments.length?(t=o.b.call(n),e):t.slice()},e.unknown=function(t){return arguments.length?(s=t,e):s},e.copy=function(){return r().domain(c).range(t).unknown(s)},e}n.d(e,"b",function(){return a}),e.a=r;var i=n(211),o=n(16),a={name:"implicit"}},function(t,e,n){"use strict";function r(t){return new Date(t)}function i(t){return t instanceof Date?+t:+new Date(+t)}function o(t,e,c,s,b,x,w,C,k){function E(n){return(w(n)<n?A:x(n)<n?P:b(n)<n?O:s(n)<n?I:e(n)<n?c(n)<n?D:R:t(n)<n?L:U)(n)}function M(e,r,i,o){if(null==e&&(e=10),"number"==typeof e){var u=Math.abs(i-r)/e,c=n.i(a.bisector)(function(t){return t[2]}).right(F,u);c===F.length?(o=n.i(a.tickStep)(r/_,i/_,e),e=t):c?(c=F[u/F[c-1][2]<F[c][2]/u?c-1:c],o=c[1],e=c[0]):(o=Math.max(n.i(a.tickStep)(r,i,e),1),e=C)}return null==o?e:e.every(o)}var T=n.i(f.a)(f.b,u.a),S=T.invert,N=T.domain,A=k(".%L"),P=k(":%S"),O=k("%I:%M"),I=k("%I %p"),D=k("%a %d"),R=k("%b %d"),L=k("%B"),U=k("%Y"),F=[[w,1,h],[w,5,5*h],[w,15,15*h],[w,30,30*h],[x,1,d],[x,5,5*d],[x,15,15*d],[x,30,30*d],[b,1,v],[b,3,3*v],[b,6,6*v],[b,12,12*v],[s,1,g],[s,2,2*g],[c,1,m],[e,1,y],[e,3,3*y],[t,1,_]];return T.invert=function(t){return new Date(S(t))},T.domain=function(t){return arguments.length?N(l.a.call(t,i)):N().map(r)},T.ticks=function(t,e){var n,r=N(),i=r[0],o=r[r.length-1],a=o<i;return a&&(n=i,i=o,o=n),n=M(t,i,o,e),n=n?n.range(i,o+1):[],a?n.reverse():n},T.tickFormat=function(t,e){return null==e?E:k(e)},T.nice=function(t,e){var r=N();return(t=M(t,r[0],r[r.length-1],e))?N(n.i(p.a)(r,t)):T},T.copy=function(){return n.i(f.c)(T,o(t,e,c,s,b,x,w,C,k))},T}e.b=o;var a=n(7),u=n(30),c=n(80),s=n(78),l=n(16),f=n(44),p=n(125),h=1e3,d=60*h,v=60*d,g=24*v,m=7*g,y=30*g,_=365*g;e.a=function(){return o(c.e,c.q,c.r,c.d,c.s,c.t,c.u,c.v,s.timeFormat).domain([new Date(2e3,0,1),new Date(2e3,0,2)])}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(257);n.d(e,"create",function(){return r.a});var i=n(45);n.d(e,"creator",function(){return i.a});var o=n(258);n.d(e,"local",function(){return o.a});var a=n(130);n.d(e,"matcher",function(){return a.a});var u=n(259);n.d(e,"mouse",function(){return u.a});var c=n(68);n.d(e,"namespace",function(){return c.a});var s=n(69);n.d(e,"namespaces",function(){return s.a});var l=n(46);n.d(e,"clientPoint",function(){return l.a});var f=n(131);n.d(e,"select",function(){return f.a});var p=n(260);n.d(e,"selectAll",function(){return p.a});var h=n(8);n.d(e,"selection",function(){return h.a});var d=n(71);n.d(e,"selector",function(){return d.a});var v=n(135);n.d(e,"selectorAll",function(){return v.a});var g=n(134);n.d(e,"style",function(){return g.a});var m=n(288);n.d(e,"touch",function(){return m.a});var y=n(289);n.d(e,"touches",function(){return y.a});var _=n(73);n.d(e,"window",function(){return _.a});var b=n(70);n.d(e,"event",function(){return b.a}),n.d(e,"customEvent",function(){return b.b})},function(t,e,n){"use strict";var r=function(t){return function(){return this.matches(t)}};if("undefined"!=typeof document){var i=document.documentElement;if(!i.matches){var o=i.webkitMatchesSelector||i.msMatchesSelector||i.mozMatchesSelector||i.oMatchesSelector;r=function(t){return function(){return o.call(this,t)}}}}e.a=r},function(t,e,n){"use strict";var r=n(8);e.a=function(t){return"string"==typeof t?new r.b([[document.querySelector(t)]],[document.documentElement]):new r.b([[t]],r.c)}},function(t,e,n){"use strict";function r(t,e){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=e}e.b=r;var i=n(133),o=n(8);e.a=function(){return new o.b(this._enter||this._groups.map(i.a),this._parents)},r.prototype={constructor:r,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,e){return this._parent.insertBefore(t,e)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}}},function(t,e,n){"use strict";e.a=function(t){return new Array(t.length)}},function(t,e,n){"use strict";function r(t){return function(){this.style.removeProperty(t)}}function i(t,e,n){return function(){this.style.setProperty(t,e,n)}}function o(t,e,n){return function(){var r=e.apply(this,arguments);null==r?this.style.removeProperty(t):this.style.setProperty(t,r,n)}}function a(t,e){return t.style.getPropertyValue(e)||n.i(u.a)(t).getComputedStyle(t,null).getPropertyValue(e)}e.a=a;var u=n(73);e.b=function(t,e,n){return arguments.length>1?this.each((null==e?r:"function"==typeof e?o:i)(t,e,null==n?"":n)):a(this.node(),t)}},function(t,e,n){"use strict";function r(){return[]}e.a=function(t){return null==t?r:function(){return this.querySelectorAll(t)}}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(290);n.d(e,"arc",function(){return r.a});var i=n(137);n.d(e,"area",function(){return i.a});var o=n(75);n.d(e,"line",function(){return o.a});var a=n(311);n.d(e,"pie",function(){return a.a});var u=n(291);n.d(e,"areaRadial",function(){return u.a}),n.d(e,"radialArea",function(){return u.a});var c=n(142);n.d(e,"lineRadial",function(){return c.a}),n.d(e,"radialLine",function(){return c.a});var s=n(143);n.d(e,"pointRadial",function(){return s.a});var l=n(303);n.d(e,"linkHorizontal",function(){return l.a}),n.d(e,"linkVertical",function(){return l.b}),n.d(e,"linkRadial",function(){return l.c});var f=n(313);n.d(e,"symbol",function(){return f.a}),n.d(e,"symbols",function(){return f.b});var p=n(144);n.d(e,"symbolCircle",function(){return p.a});var h=n(145);n.d(e,"symbolCross",function(){return h.a});var d=n(146);n.d(e,"symbolDiamond",function(){return d.a});var v=n(147);n.d(e,"symbolSquare",function(){return v.a});var g=n(148);n.d(e,"symbolStar",function(){return g.a});var m=n(149);n.d(e,"symbolTriangle",function(){return m.a});var y=n(150);n.d(e,"symbolWye",function(){return y.a});var _=n(292);n.d(e,"curveBasisClosed",function(){return _.a});var b=n(293);n.d(e,"curveBasisOpen",function(){return b.a});var x=n(47);n.d(e,"curveBasis",function(){return x.a});var w=n(294);n.d(e,"curveBundle",function(){return w.a});var C=n(139);n.d(e,"curveCardinalClosed",function(){return C.a});var k=n(140);n.d(e,"curveCardinalOpen",function(){return k.a});var E=n(48);n.d(e,"curveCardinal",function(){return E.a});var M=n(295);n.d(e,"curveCatmullRomClosed",function(){return M.a});var T=n(296);n.d(e,"curveCatmullRomOpen",function(){return T.a});var S=n(74);n.d(e,"curveCatmullRom",function(){return S.a});var N=n(297);n.d(e,"curveLinearClosed",function(){return N.a});var A=n(49);n.d(e,"curveLinear",function(){return A.a});var P=n(298);n.d(e,"curveMonotoneX",function(){return P.a}),n.d(e,"curveMonotoneY",function(){return P.b});var O=n(299);n.d(e,"curveNatural",function(){return O.a});var I=n(300);n.d(e,"curveStep",function(){return I.a}),n.d(e,"curveStepAfter",function(){return I.b}),n.d(e,"curveStepBefore",function(){return I.c});var D=n(312);n.d(e,"stack",function(){return D.a});var R=n(305);n.d(e,"stackOffsetExpand",function(){return R.a});var L=n(304);n.d(e,"stackOffsetDiverging",function(){return L.a});var U=n(36);n.d(e,"stackOffsetNone",function(){return U.a});var F=n(306);n.d(e,"stackOffsetSilhouette",function(){return F.a});var j=n(307);n.d(e,"stackOffsetWiggle",function(){return j.a});var B=n(76);n.d(e,"stackOrderAscending",function(){return B.a});var V=n(308);n.d(e,"stackOrderDescending",function(){return V.a});var W=n(309);n.d(e,"stackOrderInsideOut",function(){return W.a});var z=n(37);n.d(e,"stackOrderNone",function(){return z.a});var H=n(310);n.d(e,"stackOrderReverse",function(){return H.a})},function(t,e,n){"use strict";var r=n(32),i=n(17),o=n(49),a=n(75),u=n(77);e.a=function(){function t(t){var e,i,o,a,u,g=t.length,m=!1,y=new Array(g),_=new Array(g);for(null==h&&(v=d(u=n.i(r.a)())),e=0;e<=g;++e){if(!(e<g&&p(a=t[e],e,t))===m)if(m=!m)i=e,v.areaStart(),v.lineStart();else{for(v.lineEnd(),v.lineStart(),o=e-1;o>=i;--o)v.point(y[o],_[o]);v.lineEnd(),v.areaEnd()}m&&(y[e]=+c(a,e,t),_[e]=+l(a,e,t),v.point(s?+s(a,e,t):y[e],f?+f(a,e,t):_[e]))}if(u)return v=null,u+""||null}function e(){return n.i(a.a)().defined(p).curve(d).context(h)}var c=u.a,s=null,l=n.i(i.a)(0),f=u.b,p=n.i(i.a)(!0),h=null,d=o.a,v=null;return t.x=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),s=null,t):c},t.x0=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),t):c},t.x1=function(e){return arguments.length?(s=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):s},t.y=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),f=null,t):l},t.y0=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),t):l},t.y1=function(e){return arguments.length?(f=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):f},t.lineX0=t.lineY0=function(){return e().x(c).y(l)},t.lineY1=function(){return e().x(c).y(f)},t.lineX1=function(){return e().x(s).y(l)},t.defined=function(e){return arguments.length?(p="function"==typeof e?e:n.i(i.a)(!!e),t):p},t.curve=function(e){return arguments.length?(d=e,null!=h&&(v=d(h)),t):d},t.context=function(e){return arguments.length?(null==e?h=v=null:v=d(h=e),t):h},t}},function(t,e,n){"use strict";n.d(e,"a",function(){return r});var r=Array.prototype.slice},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}e.b=r;var i=n(50),o=n(48);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN,this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3),this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3),this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._x3=t,this._y3=e;break;case 1:this._point=2,this._context.moveTo(this._x4=t,this._y4=e);break;case 2:this._point=3,this._x5=t,this._y5=e;break;default:n.i(o.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}e.b=r;var i=n(48);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3,this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:n.i(i.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._curve=t}function i(t){function e(e){return new r(t(e))}return e._curve=t,e}n.d(e,"b",function(){return a}),e.a=i;var o=n(49),a=i(o.a);r.prototype={areaStart:function(){this._curve.areaStart()},areaEnd:function(){this._curve.areaEnd()},lineStart:function(){this._curve.lineStart()},lineEnd:function(){this._curve.lineEnd()},point:function(t,e){this._curve.point(e*Math.sin(t),e*-Math.cos(t))}}},function(t,e,n){"use strict";function r(t){var e=t.curve;return t.angle=t.x,delete t.x,t.radius=t.y,delete t.y,t.curve=function(t){return arguments.length?e(n.i(i.a)(t)):e()._curve},t}e.b=r;var i=n(141),o=n(75);e.a=function(){return r(n.i(o.a)().curve(i.b))}},function(t,e,n){"use strict";e.a=function(t,e){return[(e=+e)*Math.cos(t-=Math.PI/2),e*Math.sin(t)]}},function(t,e,n){"use strict";var r=n(35);e.a={draw:function(t,e){var n=Math.sqrt(e/r.b);t.moveTo(n,0),t.arc(0,0,n,0,r.c)}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e/5)/2;t.moveTo(-3*n,-n),t.lineTo(-n,-n),t.lineTo(-n,-3*n),t.lineTo(n,-3*n),t.lineTo(n,-n),t.lineTo(3*n,-n),t.lineTo(3*n,n),t.lineTo(n,n),t.lineTo(n,3*n),t.lineTo(-n,3*n),t.lineTo(-n,n),t.lineTo(-3*n,n),t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(1/3),i=2*r;e.a={draw:function(t,e){var n=Math.sqrt(e/i),o=n*r;t.moveTo(0,-n),t.lineTo(o,0),t.lineTo(0,n),t.lineTo(-o,0),t.closePath()}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e),r=-n/2;t.rect(r,r,n,n)}}},function(t,e,n){"use strict";var r=n(35),i=Math.sin(r.b/10)/Math.sin(7*r.b/10),o=Math.sin(r.c/10)*i,a=-Math.cos(r.c/10)*i;e.a={draw:function(t,e){var n=Math.sqrt(.8908130915292852*e),i=o*n,u=a*n;t.moveTo(0,-n),t.lineTo(i,u);for(var c=1;c<5;++c){var s=r.c*c/5,l=Math.cos(s),f=Math.sin(s);t.lineTo(f*n,-l*n),t.lineTo(l*i-f*u,f*i+l*u)}t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(3);e.a={draw:function(t,e){var n=-Math.sqrt(e/(3*r));t.moveTo(0,2*n),t.lineTo(-r*n,-n),t.lineTo(r*n,-n),t.closePath()}}},function(t,e,n){"use strict";var r=-.5,i=Math.sqrt(3)/2,o=1/Math.sqrt(12),a=3*(o/2+1);e.a={draw:function(t,e){var n=Math.sqrt(e/a),u=n/2,c=n*o,s=u,l=n*o+n,f=-s,p=l;t.moveTo(u,c),t.lineTo(s,l),t.lineTo(f,p),t.lineTo(r*u-i*c,i*u+r*c),t.lineTo(r*s-i*l,i*s+r*l),t.lineTo(r*f-i*p,i*f+r*p),t.lineTo(r*u+i*c,r*c-i*u),t.lineTo(r*s+i*l,r*l-i*s),t.lineTo(r*f+i*p,r*p-i*f),t.closePath()}}},function(t,e,n){"use strict";function r(t){return t.toISOString()}n.d(e,"b",function(){return o});var i=n(79),o="%Y-%m-%dT%H:%M:%S.%LZ",a=Date.prototype.toISOString?r:n.i(i.d)(o);e.a=a},function(t,e,n){"use strict";function r(t){if(0<=t.y&&t.y<100){var e=new Date(-1,t.m,t.d,t.H,t.M,t.S,t.L);return e.setFullYear(t.y),e}return new Date(t.y,t.m,t.d,t.H,t.M,t.S,t.L)}function i(t){if(0<=t.y&&t.y<100){var e=new Date(Date.UTC(-1,t.m,t.d,t.H,t.M,t.S,t.L));return e.setUTCFullYear(t.y),e}return new Date(Date.UTC(t.y,t.m,t.d,t.H,t.M,t.S,t.L))}function o(t){return{y:t,m:0,d:1,H:0,M:0,S:0,L:0}}function a(t){function e(t,e){return function(n){var r,i,o,a=[],u=-1,c=0,s=t.length;for(n instanceof Date||(n=new Date(+n));++u<s;)37===t.charCodeAt(u)&&(a.push(t.slice(c,u)),null!=(i=dt[r=t.charAt(++u)])?r=t.charAt(++u):i="e"===r?" ":"0",(o=e[r])&&(r=o(n,i)),a.push(r),c=u+1);return a.push(t.slice(c,u)),a.join("")}}function a(t,e){return function(r){var a,c,s=o(1900),l=u(s,t,r+="",0);if(l!=r.length)return null;if("Q"in s)return new Date(s.Q);if("p"in s&&(s.H=s.H%12+12*s.p),"V"in s){if(s.V<1||s.V>53)return null;"w"in s||(s.w=1),"Z"in s?(a=i(o(s.y)),c=a.getUTCDay(),a=c>4||0===c?ht.a.ceil(a):n.i(ht.a)(a),a=ht.b.offset(a,7*(s.V-1)),s.y=a.getUTCFullYear(),s.m=a.getUTCMonth(),s.d=a.getUTCDate()+(s.w+6)%7):(a=e(o(s.y)),c=a.getDay(),a=c>4||0===c?ht.c.ceil(a):n.i(ht.c)(a),a=ht.d.offset(a,7*(s.V-1)),s.y=a.getFullYear(),s.m=a.getMonth(),s.d=a.getDate()+(s.w+6)%7)}else("W"in s||"U"in s)&&("w"in s||(s.w="u"in s?s.u%7:"W"in s?1:0),c="Z"in s?i(o(s.y)).getUTCDay():e(o(s.y)).getDay(),s.m=0,s.d="W"in s?(s.w+6)%7+7*s.W-(c+5)%7:s.w+7*s.U-(c+6)%7);return"Z"in s?(s.H+=s.Z/100|0,s.M+=s.Z%100,i(s)):e(s)}}function u(t,e,n,r){for(var i,o,a=0,u=e.length,c=n.length;a<u;){if(r>=c)return-1;if(37===(i=e.charCodeAt(a++))){if(i=e.charAt(a++),!(o=Zt[i in dt?e.charAt(a++):i])||(r=o(t,n,r))<0)return-1}else if(i!=n.charCodeAt(r++))return-1}return r}function c(t,e,n){var r=Bt.exec(e.slice(n));return r?(t.p=Vt[r[0].toLowerCase()],n+r[0].length):-1}function vt(t,e,n){var r=Ht.exec(e.slice(n));return r?(t.w=qt[r[0].toLowerCase()],n+r[0].length):-1}function gt(t,e,n){var r=Wt.exec(e.slice(n));return r?(t.w=zt[r[0].toLowerCase()],n+r[0].length):-1}function mt(t,e,n){var r=Gt.exec(e.slice(n));return r?(t.m=$t[r[0].toLowerCase()],n+r[0].length):-1}function yt(t,e,n){var r=Yt.exec(e.slice(n));return r?(t.m=Kt[r[0].toLowerCase()],n+r[0].length):-1}function _t(t,e,n){return u(t,Ot,e,n)}function bt(t,e,n){return u(t,It,e,n)}function xt(t,e,n){return u(t,Dt,e,n)}function wt(t){return Ut[t.getDay()]}function Ct(t){return Lt[t.getDay()]}function kt(t){return jt[t.getMonth()]}function Et(t){return Ft[t.getMonth()]}function Mt(t){return Rt[+(t.getHours()>=12)]}function Tt(t){return Ut[t.getUTCDay()]}function St(t){return Lt[t.getUTCDay()]}function Nt(t){return jt[t.getUTCMonth()]}function At(t){return Ft[t.getUTCMonth()]}function Pt(t){return Rt[+(t.getUTCHours()>=12)]}var Ot=t.dateTime,It=t.date,Dt=t.time,Rt=t.periods,Lt=t.days,Ut=t.shortDays,Ft=t.months,jt=t.shortMonths,Bt=s(Rt),Vt=l(Rt),Wt=s(Lt),zt=l(Lt),Ht=s(Ut),qt=l(Ut),Yt=s(Ft),Kt=l(Ft),Gt=s(jt),$t=l(jt),Xt={a:wt,A:Ct,b:kt,B:Et,c:null,d:A,e:A,f:R,H:P,I:O,j:I,L:D,m:L,M:U,p:Mt,Q:ft,s:pt,S:F,u:j,U:B,V:V,w:W,W:z,x:null,X:null,y:H,Y:q,Z:Y,"%":lt},Qt={a:Tt,A:St,b:Nt,B:At,c:null,d:K,e:K,f:Z,H:G,I:$,j:X,L:Q,m:J,M:tt,p:Pt,Q:ft,s:pt,S:et,u:nt,U:rt,V:it,w:ot,W:at,x:null,X:null,y:ut,Y:ct,Z:st,"%":lt},Zt={a:vt,A:gt,b:mt,B:yt,c:_t,d:b,e:b,f:M,H:w,I:w,j:x,L:E,m:_,M:C,p:c,Q:S,s:N,S:k,u:p,U:h,V:d,w:f,W:v,x:bt,X:xt,y:m,Y:g,Z:y,"%":T};return Xt.x=e(It,Xt),Xt.X=e(Dt,Xt),Xt.c=e(Ot,Xt),Qt.x=e(It,Qt),Qt.X=e(Dt,Qt),Qt.c=e(Ot,Qt),{format:function(t){var n=e(t+="",Xt);return n.toString=function(){return t},n},parse:function(t){var e=a(t+="",r);return e.toString=function(){return t},e},utcFormat:function(t){var n=e(t+="",Qt);return n.toString=function(){return t},n},utcParse:function(t){var e=a(t,i);return e.toString=function(){return t},e}}}function u(t,e,n){var r=t<0?"-":"",i=(r?-t:t)+"",o=i.length;return r+(o<n?new Array(n-o+1).join(e)+i:i)}function c(t){return t.replace(mt,"\\$&")}function s(t){return new RegExp("^(?:"+t.map(c).join("|")+")","i")}function l(t){for(var e={},n=-1,r=t.length;++n<r;)e[t[n].toLowerCase()]=n;return e}function f(t,e,n){var r=vt.exec(e.slice(n,n+1));return r?(t.w=+r[0],n+r[0].length):-1}function p(t,e,n){var r=vt.exec(e.slice(n,n+1));return r?(t.u=+r[0],n+r[0].length):-1}function h(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.U=+r[0],n+r[0].length):-1}function d(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.V=+r[0],n+r[0].length):-1}function v(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.W=+r[0],n+r[0].length):-1}function g(t,e,n){var r=vt.exec(e.slice(n,n+4));return r?(t.y=+r[0],n+r[0].length):-1}function m(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.y=+r[0]+(+r[0]>68?1900:2e3),n+r[0].length):-1}function y(t,e,n){var r=/^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(e.slice(n,n+6));return r?(t.Z=r[1]?0:-(r[2]+(r[3]||"00")),n+r[0].length):-1}function _(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.m=r[0]-1,n+r[0].length):-1}function b(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.d=+r[0],n+r[0].length):-1}function x(t,e,n){var r=vt.exec(e.slice(n,n+3));return r?(t.m=0,t.d=+r[0],n+r[0].length):-1}function w(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.H=+r[0],n+r[0].length):-1}function C(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.M=+r[0],n+r[0].length):-1}function k(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.S=+r[0],n+r[0].length):-1}function E(t,e,n){var r=vt.exec(e.slice(n,n+3));return r?(t.L=+r[0],n+r[0].length):-1}function M(t,e,n){var r=vt.exec(e.slice(n,n+6));return r?(t.L=Math.floor(r[0]/1e3),n+r[0].length):-1}function T(t,e,n){var r=gt.exec(e.slice(n,n+1));return r?n+r[0].length:-1}function S(t,e,n){var r=vt.exec(e.slice(n));return r?(t.Q=+r[0],n+r[0].length):-1}function N(t,e,n){var r=vt.exec(e.slice(n));return r?(t.Q=1e3*+r[0],n+r[0].length):-1}function A(t,e){return u(t.getDate(),e,2)}function P(t,e){return u(t.getHours(),e,2)}function O(t,e){return u(t.getHours()%12||12,e,2)}function I(t,e){return u(1+ht.d.count(n.i(ht.e)(t),t),e,3)}function D(t,e){return u(t.getMilliseconds(),e,3)}function R(t,e){return D(t,e)+"000"}function L(t,e){return u(t.getMonth()+1,e,2)}function U(t,e){return u(t.getMinutes(),e,2)}function F(t,e){return u(t.getSeconds(),e,2)}function j(t){var e=t.getDay();return 0===e?7:e}function B(t,e){return u(ht.f.count(n.i(ht.e)(t),t),e,2)}function V(t,e){var r=t.getDay();return t=r>=4||0===r?n.i(ht.g)(t):ht.g.ceil(t),u(ht.g.count(n.i(ht.e)(t),t)+(4===n.i(ht.e)(t).getDay()),e,2)}function W(t){return t.getDay()}function z(t,e){return u(ht.c.count(n.i(ht.e)(t),t),e,2)}function H(t,e){return u(t.getFullYear()%100,e,2)}function q(t,e){return u(t.getFullYear()%1e4,e,4)}function Y(t){var e=t.getTimezoneOffset();return(e>0?"-":(e*=-1,"+"))+u(e/60|0,"0",2)+u(e%60,"0",2)}function K(t,e){return u(t.getUTCDate(),e,2)}function G(t,e){return u(t.getUTCHours(),e,2)}function $(t,e){return u(t.getUTCHours()%12||12,e,2)}function X(t,e){return u(1+ht.b.count(n.i(ht.h)(t),t),e,3)}function Q(t,e){return u(t.getUTCMilliseconds(),e,3)}function Z(t,e){return Q(t,e)+"000"}function J(t,e){return u(t.getUTCMonth()+1,e,2)}function tt(t,e){return u(t.getUTCMinutes(),e,2)}function et(t,e){return u(t.getUTCSeconds(),e,2)}function nt(t){var e=t.getUTCDay();return 0===e?7:e}function rt(t,e){return u(ht.i.count(n.i(ht.h)(t),t),e,2)}function it(t,e){var r=t.getUTCDay();return t=r>=4||0===r?n.i(ht.j)(t):ht.j.ceil(t),u(ht.j.count(n.i(ht.h)(t),t)+(4===n.i(ht.h)(t).getUTCDay()),e,2)}function ot(t){return t.getUTCDay()}function at(t,e){return u(ht.a.count(n.i(ht.h)(t),t),e,2)}function ut(t,e){return u(t.getUTCFullYear()%100,e,2)}function ct(t,e){return u(t.getUTCFullYear()%1e4,e,4)}function st(){return"+0000"}function lt(){return"%"}function ft(t){return+t}function pt(t){return Math.floor(+t/1e3)}e.a=a;var ht=n(80),dt={"-":"",_:" ",0:"0"},vt=/^\s*\d+/,gt=/^%/,mt=/[\\^$*+?|[\]().{}]/g},function(t,e,n){"use strict";var r=n(11),i={listen:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!1),{remove:function(){t.removeEventListener(e,n,!1)}}):t.attachEvent?(t.attachEvent("on"+e,n),{remove:function(){t.detachEvent("on"+e,n)}}):void 0},capture:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!0),{remove:function(){t.removeEventListener(e,n,!0)}}):{remove:r}},registerDefault:function(){}};t.exports=i},function(t,e,n){"use strict";function r(t){try{t.focus()}catch(t){}}t.exports=r},function(t,e,n){"use strict";function r(t){if(void 0===(t=t||("undefined"!=typeof document?document:void 0)))return null;try{return t.activeElement||t.body}catch(e){return t.body}}t.exports=r},function(t,e){function n(){throw new Error("setTimeout has not been defined")}function r(){throw new Error("clearTimeout has not been defined")}function i(t){if(l===setTimeout)return setTimeout(t,0);if((l===n||!l)&&setTimeout)return l=setTimeout,setTimeout(t,0);try{return l(t,0)}catch(e){try{return l.call(null,t,0)}catch(e){return l.call(this,t,0)}}}function o(t){if(f===clearTimeout)return clearTimeout(t);if((f===r||!f)&&clearTimeout)return f=clearTimeout,clearTimeout(t);try{return f(t)}catch(e){try{return f.call(null,t)}catch(e){return f.call(this,t)}}}function a(){v&&h&&(v=!1,h.length?d=h.concat(d):g=-1,d.length&&u())}function u(){if(!v){var t=i(a);v=!0;for(var e=d.length;e;){for(h=d,d=[];++g<e;)h&&h[g].run();g=-1,e=d.length}h=null,v=!1,o(t)}}function c(t,e){this.fun=t,this.array=e}function s(){}var l,f,p=t.exports={};!function(){try{l="function"==typeof setTimeout?setTimeout:n}catch(t){l=n}try{f="function"==typeof clearTimeout?clearTimeout:r}catch(t){f=r}}();var h,d=[],v=!1,g=-1;p.nextTick=function(t){var e=new Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)e[n-1]=arguments[n];d.push(new c(t,e)),1!==d.length||v||i(u)},c.prototype.run=function(){this.fun.apply(null,this.array)},p.title="browser",p.browser=!0,p.env={},p.argv=[],p.version="",p.versions={},p.on=s,p.addListener=s,p.once=s,p.off=s,p.removeListener=s,p.removeAllListeners=s,p.emit=s,p.prependListener=s,p.prependOnceListener=s,p.listeners=function(t){return[]},p.binding=function(t){throw new Error("process.binding is not supported")},p.cwd=function(){return"/"},p.chdir=function(t){throw new Error("process.chdir is not supported")},p.umask=function(){return 0}},function(t,e,n){"use strict";var r=n(343);t.exports=function(t){return r(t,!1)}},function(t,e,n){"use strict";function r(t,e){return t+e.charAt(0).toUpperCase()+e.substring(1)}var i={animationIterationCount:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},o=["Webkit","ms","Moz","O"];Object.keys(i).forEach(function(t){o.forEach(function(e){i[r(e,t)]=i[t]})});var a={background:{backgroundAttachment:!0,backgroundColor:!0,backgroundImage:!0,backgroundPositionX:!0,backgroundPositionY:!0,backgroundRepeat:!0},backgroundPosition:{backgroundPositionX:!0,backgroundPositionY:!0},border:{borderWidth:!0,borderStyle:!0,borderColor:!0},borderBottom:{borderBottomWidth:!0,borderBottomStyle:!0,borderBottomColor:!0},borderLeft:{borderLeftWidth:!0,borderLeftStyle:!0,borderLeftColor:!0},borderRight:{borderRightWidth:!0,borderRightStyle:!0,borderRightColor:!0},borderTop:{borderTopWidth:!0,borderTopStyle:!0,borderTopColor:!0},font:{fontStyle:!0,fontVariant:!0,fontWeight:!0,fontSize:!0,lineHeight:!0,fontFamily:!0},outline:{outlineWidth:!0,outlineStyle:!0,outlineColor:!0}},u={isUnitlessNumber:i,shorthandPropertyExpansions:a};t.exports=u},function(t,e,n){"use strict";function r(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}var i=n(1),o=n(18),a=(n(0),function(){function t(e){r(this,t),this._callbacks=null,this._contexts=null,this._arg=e}return t.prototype.enqueue=function(t,e){this._callbacks=this._callbacks||[],this._callbacks.push(t),this._contexts=this._contexts||[],this._contexts.push(e)},t.prototype.notifyAll=function(){var t=this._callbacks,e=this._contexts,n=this._arg;if(t&&e){t.length!==e.length&&i("24"),this._callbacks=null,this._contexts=null;for(var r=0;r<t.length;r++)t[r].call(e[r],n);t.length=0,e.length=0}},t.prototype.checkpoint=function(){return this._callbacks?this._callbacks.length:0},t.prototype.rollback=function(t){this._callbacks&&this._contexts&&(this._callbacks.length=t,this._contexts.length=t)},t.prototype.reset=function(){this._callbacks=null,this._contexts=null},t.prototype.destructor=function(){this.reset()},t}());t.exports=o.addPoolingTo(a)},function(t,e,n){"use strict";function r(t){return!!s.hasOwnProperty(t)||!c.hasOwnProperty(t)&&(u.test(t)?(s[t]=!0,!0):(c[t]=!0,!1))}function i(t,e){return null==e||t.hasBooleanValue&&!e||t.hasNumericValue&&isNaN(e)||t.hasPositiveNumericValue&&e<1||t.hasOverloadedBooleanValue&&!1===e}var o=n(21),a=(n(4),n(9),n(407)),u=(n(2),new RegExp("^["+o.ATTRIBUTE_NAME_START_CHAR+"]["+o.ATTRIBUTE_NAME_CHAR+"]*$")),c={},s={},l={createMarkupForID:function(t){return o.ID_ATTRIBUTE_NAME+"="+a(t)},setAttributeForID:function(t,e){t.setAttribute(o.ID_ATTRIBUTE_NAME,e)},createMarkupForRoot:function(){return o.ROOT_ATTRIBUTE_NAME+'=""'},setAttributeForRoot:function(t){t.setAttribute(o.ROOT_ATTRIBUTE_NAME,"")},createMarkupForProperty:function(t,e){var n=o.properties.hasOwnProperty(t)?o.properties[t]:null;if(n){if(i(n,e))return"";var r=n.attributeName;return n.hasBooleanValue||n.hasOverloadedBooleanValue&&!0===e?r+'=""':r+"="+a(e)}return o.isCustomAttribute(t)?null==e?"":t+"="+a(e):null},createMarkupForCustomAttribute:function(t,e){return r(t)&&null!=e?t+"="+a(e):""},setValueForProperty:function(t,e,n){var r=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(r){var a=r.mutationMethod;if(a)a(t,n);else{if(i(r,n))return void this.deleteValueForProperty(t,e);if(r.mustUseProperty)t[r.propertyName]=n;else{var u=r.attributeName,c=r.attributeNamespace;c?t.setAttributeNS(c,u,""+n):r.hasBooleanValue||r.hasOverloadedBooleanValue&&!0===n?t.setAttribute(u,""):t.setAttribute(u,""+n)}}}else if(o.isCustomAttribute(e))return void l.setValueForAttribute(t,e,n)},setValueForAttribute:function(t,e,n){if(r(e)){null==n?t.removeAttribute(e):t.setAttribute(e,""+n)}},deleteValueForAttribute:function(t,e){t.removeAttribute(e)},deleteValueForProperty:function(t,e){var n=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(n){var r=n.mutationMethod;if(r)r(t,void 0);else if(n.mustUseProperty){var i=n.propertyName;n.hasBooleanValue?t[i]=!1:t[i]=""}else t.removeAttribute(n.attributeName)}else o.isCustomAttribute(e)&&t.removeAttribute(e)}};t.exports=l},function(t,e,n){"use strict";var r={hasCachedChildNodes:1};t.exports=r},function(t,e,n){"use strict";function r(){if(this._rootNodeID&&this._wrapperState.pendingUpdate){this._wrapperState.pendingUpdate=!1;var t=this._currentElement.props,e=u.getValue(t);null!=e&&i(this,Boolean(t.multiple),e)}}function i(t,e,n){var r,i,o=c.getNodeFromInstance(t).options;if(e){for(r={},i=0;i<n.length;i++)r[""+n[i]]=!0;for(i=0;i<o.length;i++){var a=r.hasOwnProperty(o[i].value);o[i].selected!==a&&(o[i].selected=a)}}else{for(r=""+n,i=0;i<o.length;i++)if(o[i].value===r)return void(o[i].selected=!0);o.length&&(o[0].selected=!0)}}function o(t){var e=this._currentElement.props,n=u.executeOnChange(e,t);return this._rootNodeID&&(this._wrapperState.pendingUpdate=!0),s.asap(r,this),n}var a=n(3),u=n(86),c=n(4),s=n(12),l=(n(2),!1),f={getHostProps:function(t,e){return a({},e,{onChange:t._wrapperState.onChange,value:void 0})},mountWrapper:function(t,e){var n=u.getValue(e);t._wrapperState={pendingUpdate:!1,initialValue:null!=n?n:e.defaultValue,listeners:null,onChange:o.bind(t),wasMultiple:Boolean(e.multiple)},void 0===e.value||void 0===e.defaultValue||l||(l=!0)},getSelectValueContext:function(t){return t._wrapperState.initialValue},postUpdateWrapper:function(t){var e=t._currentElement.props;t._wrapperState.initialValue=void 0;var n=t._wrapperState.wasMultiple;t._wrapperState.wasMultiple=Boolean(e.multiple);var r=u.getValue(e);null!=r?(t._wrapperState.pendingUpdate=!1,i(t,Boolean(e.multiple),r)):n!==Boolean(e.multiple)&&(null!=e.defaultValue?i(t,Boolean(e.multiple),e.defaultValue):i(t,Boolean(e.multiple),e.multiple?[]:""))}};t.exports=f},function(t,e,n){"use strict";var r,i={injectEmptyComponentFactory:function(t){r=t}},o={create:function(t){return r(t)}};o.injection=i,t.exports=o},function(t,e,n){"use strict";var r={logTopLevelRenders:!1};t.exports=r},function(t,e,n){"use strict";function r(t){return u||a("111",t.type),new u(t)}function i(t){return new c(t)}function o(t){return t instanceof c}var a=n(1),u=(n(0),null),c=null,s={injectGenericComponentClass:function(t){u=t},injectTextComponentClass:function(t){c=t}},l={createInternalComponent:r,createInstanceForText:i,isTextComponent:o,injection:s};t.exports=l},function(t,e,n){"use strict";function r(t){return o(document.documentElement,t)}var i=n(367),o=n(331),a=n(154),u=n(155),c={hasSelectionCapabilities:function(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&("input"===e&&"text"===t.type||"textarea"===e||"true"===t.contentEditable)},getSelectionInformation:function(){var t=u();return{focusedElem:t,selectionRange:c.hasSelectionCapabilities(t)?c.getSelection(t):null}},restoreSelection:function(t){var e=u(),n=t.focusedElem,i=t.selectionRange;e!==n&&r(n)&&(c.hasSelectionCapabilities(n)&&c.setSelection(n,i),a(n))},getSelection:function(t){var e;if("selectionStart"in t)e={start:t.selectionStart,end:t.selectionEnd};else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var n=document.selection.createRange();n.parentElement()===t&&(e={start:-n.moveStart("character",-t.value.length),end:-n.moveEnd("character",-t.value.length)})}else e=i.getOffsets(t);return e||{start:0,end:0}},setSelection:function(t,e){var n=e.start,r=e.end;if(void 0===r&&(r=n),"selectionStart"in t)t.selectionStart=n,t.selectionEnd=Math.min(r,t.value.length);else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var o=t.createTextRange();o.collapse(!0),o.moveStart("character",n),o.moveEnd("character",r-n),o.select()}else i.setOffsets(t,e)}};t.exports=c},function(t,e,n){"use strict";function r(t,e){for(var n=Math.min(t.length,e.length),r=0;r<n;r++)if(t.charAt(r)!==e.charAt(r))return r;return t.length===e.length?-1:n}function i(t){return t?t.nodeType===D?t.documentElement:t.firstChild:null}function o(t){return t.getAttribute&&t.getAttribute(P)||""}function a(t,e,n,r,i){var o;if(x.logTopLevelRenders){var a=t._currentElement.props.child,u=a.type;o="React mount: "+("string"==typeof u?u:u.displayName||u.name),console.time(o)}var c=k.mountComponent(t,n,null,_(t,e),i,0);o&&console.timeEnd(o),t._renderedComponent._topLevelWrapper=t,j._mountImageIntoNode(c,e,t,r,n)}function u(t,e,n,r){var i=M.ReactReconcileTransaction.getPooled(!n&&b.useCreateElement);i.perform(a,null,t,e,i,n,r),M.ReactReconcileTransaction.release(i)}function c(t,e,n){for(k.unmountComponent(t,n),e.nodeType===D&&(e=e.documentElement);e.lastChild;)e.removeChild(e.lastChild)}function s(t){var e=i(t);if(e){var n=y.getInstanceFromNode(e);return!(!n||!n._hostParent)}}function l(t){return!(!t||t.nodeType!==I&&t.nodeType!==D&&t.nodeType!==R)}function f(t){var e=i(t),n=e&&y.getInstanceFromNode(e);return n&&!n._hostParent?n:null}function p(t){var e=f(t);return e?e._hostContainerInfo._topLevelWrapper:null}var h=n(1),d=n(20),v=n(21),g=n(26),m=n(53),y=(n(15),n(4)),_=n(361),b=n(363),x=n(164),w=n(39),C=(n(9),n(377)),k=n(24),E=n(89),M=n(12),T=n(51),S=n(174),N=(n(0),n(57)),A=n(96),P=(n(2),v.ID_ATTRIBUTE_NAME),O=v.ROOT_ATTRIBUTE_NAME,I=1,D=9,R=11,L={},U=1,F=function(){this.rootID=U++};F.prototype.isReactComponent={},F.prototype.render=function(){return this.props.child},F.isReactTopLevelWrapper=!0;var j={TopLevelWrapper:F,_instancesByReactRootID:L,scrollMonitor:function(t,e){e()},_updateRootComponent:function(t,e,n,r,i){return j.scrollMonitor(r,function(){E.enqueueElementInternal(t,e,n),i&&E.enqueueCallbackInternal(t,i)}),t},_renderNewRootComponent:function(t,e,n,r){l(e)||h("37"),m.ensureScrollValueMonitoring();var i=S(t,!1);M.batchedUpdates(u,i,e,n,r);var o=i._instance.rootID;return L[o]=i,i},renderSubtreeIntoContainer:function(t,e,n,r){return null!=t&&w.has(t)||h("38"),j._renderSubtreeIntoContainer(t,e,n,r)},_renderSubtreeIntoContainer:function(t,e,n,r){E.validateCallback(r,"ReactDOM.render"),g.isValidElement(e)||h("39","string"==typeof e?" Instead of passing a string like 'div', pass React.createElement('div') or <div />.":"function"==typeof e?" Instead of passing a class like Foo, pass React.createElement(Foo) or <Foo />.":null!=e&&void 0!==e.props?" This may be caused by unintentionally loading two independent copies of React.":"");var a,u=g.createElement(F,{child:e});if(t){var c=w.get(t);a=c._processChildContext(c._context)}else a=T;var l=p(n);if(l){var f=l._currentElement,d=f.props.child;if(A(d,e)){var v=l._renderedComponent.getPublicInstance(),m=r&&function(){r.call(v)};return j._updateRootComponent(l,u,a,n,m),v}j.unmountComponentAtNode(n)}var y=i(n),_=y&&!!o(y),b=s(n),x=_&&!l&&!b,C=j._renderNewRootComponent(u,n,x,a)._renderedComponent.getPublicInstance();return r&&r.call(C),C},render:function(t,e,n){return j._renderSubtreeIntoContainer(null,t,e,n)},unmountComponentAtNode:function(t){l(t)||h("40");var e=p(t);if(!e){s(t),1===t.nodeType&&t.hasAttribute(O);return!1}return delete L[e._instance.rootID],M.batchedUpdates(c,e,t,!1),!0},_mountImageIntoNode:function(t,e,n,o,a){if(l(e)||h("41"),o){var u=i(e);if(C.canReuseMarkup(t,u))return void y.precacheNode(n,u);var c=u.getAttribute(C.CHECKSUM_ATTR_NAME);u.removeAttribute(C.CHECKSUM_ATTR_NAME);var s=u.outerHTML;u.setAttribute(C.CHECKSUM_ATTR_NAME,c);var f=t,p=r(f,s),v=" (client) "+f.substring(p-20,p+20)+"\n (server) "+s.substring(p-20,p+20);e.nodeType===D&&h("42",v)}if(e.nodeType===D&&h("43"),a.useCreateElement){for(;e.lastChild;)e.removeChild(e.lastChild);d.insertTreeBefore(e,t,null)}else N(e,t),y.precacheNode(n,e.firstChild)}};t.exports=j},function(t,e,n){"use strict";var r=n(1),i=n(26),o=(n(0),{HOST:0,COMPOSITE:1,EMPTY:2,getType:function(t){return null===t||!1===t?o.EMPTY:i.isValidElement(t)?"function"==typeof t.type?o.COMPOSITE:o.HOST:void r("26",t)}});t.exports=o},function(t,e,n){"use strict";function r(t,e){return null==e&&i("30"),null==t?e:Array.isArray(t)?Array.isArray(e)?(t.push.apply(t,e),t):(t.push(e),t):Array.isArray(e)?[t].concat(e):[t,e]}var i=n(1);n(0);t.exports=r},function(t,e,n){"use strict";function r(t,e,n){Array.isArray(t)?t.forEach(e,n):t&&e.call(n,t)}t.exports=r},function(t,e,n){"use strict";function r(t){for(var e;(e=t._renderedNodeType)===i.COMPOSITE;)t=t._renderedComponent;return e===i.HOST?t._renderedComponent:e===i.EMPTY?null:void 0}var i=n(168);t.exports=r},function(t,e,n){"use strict";function r(){return!o&&i.canUseDOM&&(o="textContent"in document.documentElement?"textContent":"innerText"),o}var i=n(6),o=null;t.exports=r},function(t,e,n){"use strict";function r(t){var e=t.type,n=t.nodeName;return n&&"input"===n.toLowerCase()&&("checkbox"===e||"radio"===e)}function i(t){return t._wrapperState.valueTracker}function o(t,e){t._wrapperState.valueTracker=e}function a(t){t._wrapperState.valueTracker=null}function u(t){var e;return t&&(e=r(t)?""+t.checked:t.value),e}var c=n(4),s={_getTrackerFromNode:function(t){return i(c.getInstanceFromNode(t))},track:function(t){if(!i(t)){var e=c.getNodeFromInstance(t),n=r(e)?"checked":"value",u=Object.getOwnPropertyDescriptor(e.constructor.prototype,n),s=""+e[n];e.hasOwnProperty(n)||"function"!=typeof u.get||"function"!=typeof u.set||(Object.defineProperty(e,n,{enumerable:u.enumerable,configurable:!0,get:function(){return u.get.call(this)},set:function(t){s=""+t,u.set.call(this,t)}}),o(t,{getValue:function(){return s},setValue:function(t){s=""+t},stopTracking:function(){a(t),delete e[n]}}))}},updateValueIfChanged:function(t){if(!t)return!1;var e=i(t);if(!e)return s.track(t),!0;var n=e.getValue(),r=u(c.getNodeFromInstance(t));return r!==n&&(e.setValue(r),!0)},stopTracking:function(t){var e=i(t);e&&e.stopTracking()}};t.exports=s},function(t,e,n){"use strict";function r(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}function i(t){return"function"==typeof t&&void 0!==t.prototype&&"function"==typeof t.prototype.mountComponent&&"function"==typeof t.prototype.receiveComponent}function o(t,e){var n;if(null===t||!1===t)n=s.create(o);else if("object"==typeof t){var u=t,c=u.type;if("function"!=typeof c&&"string"!=typeof c){var p="";p+=r(u._owner),a("130",null==c?c:typeof c,p)}"string"==typeof u.type?n=l.createInternalComponent(u):i(u.type)?(n=new u.type(u),n.getHostNode||(n.getHostNode=n.getNativeNode)):n=new f(u)}else"string"==typeof t||"number"==typeof t?n=l.createInstanceForText(t):a("131",typeof t);return n._mountIndex=0,n._mountImage=null,n}var a=n(1),u=n(3),c=n(358),s=n(163),l=n(165),f=(n(420),n(0),n(2),function(t){this.construct(t)});u(f.prototype,c,{_instantiateReactComponent:o}),t.exports=o},function(t,e,n){"use strict";function r(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return"input"===e?!!i[t.type]:"textarea"===e}var i={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};t.exports=r},function(t,e,n){"use strict";var r=n(6),i=n(56),o=n(57),a=function(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&3===n.nodeType)return void(n.nodeValue=e)}t.textContent=e};r.canUseDOM&&("textContent"in document.documentElement||(a=function(t,e){if(3===t.nodeType)return void(t.nodeValue=e);o(t,i(e))})),t.exports=a},function(t,e,n){"use strict";function r(t,e){return t&&"object"==typeof t&&null!=t.key?s.escape(t.key):e.toString(36)}function i(t,e,n,o){var p=typeof t;if("undefined"!==p&&"boolean"!==p||(t=null),null===t||"string"===p||"number"===p||"object"===p&&t.$$typeof===u)return n(o,t,""===e?l+r(t,0):e),1;var h,d,v=0,g=""===e?l:e+f;if(Array.isArray(t))for(var m=0;m<t.length;m++)h=t[m],d=g+r(h,m),v+=i(h,d,n,o);else{var y=c(t);if(y){var _,b=y.call(t);if(y!==t.entries)for(var x=0;!(_=b.next()).done;)h=_.value,d=g+r(h,x++),v+=i(h,d,n,o);else for(;!(_=b.next()).done;){var w=_.value;w&&(h=w[1],d=g+s.escape(w[0])+f+r(h,0),v+=i(h,d,n,o))}}else if("object"===p){var C="",k=String(t);a("31","[object Object]"===k?"object with keys {"+Object.keys(t).join(", ")+"}":k,C)}}return v}function o(t,e,n){return null==t?0:i(t,"",e,n)}var a=n(1),u=(n(15),n(373)),c=n(404),s=(n(0),n(85)),l=(n(2),"."),f=":";t.exports=o},function(t,e,n){"use strict";function r(t,e,n){this.props=t,this.context=e,this.refs=s,this.updater=n||c}function i(t,e,n){this.props=t,this.context=e,this.refs=s,this.updater=n||c}function o(){}var a=n(40),u=n(3),c=n(181),s=(n(182),n(51));n(0),n(421);r.prototype.isReactComponent={},r.prototype.setState=function(t,e){"object"!=typeof t&&"function"!=typeof t&&null!=t&&a("85"),this.updater.enqueueSetState(this,t),e&&this.updater.enqueueCallback(this,e,"setState")},r.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this),t&&this.updater.enqueueCallback(this,t,"forceUpdate")};o.prototype=r.prototype,i.prototype=new o,i.prototype.constructor=i,u(i.prototype,r.prototype),i.prototype.isPureReactComponent=!0,t.exports={Component:r,PureComponent:i}},function(t,e,n){"use strict";function r(t){var e=Function.prototype.toString,n=Object.prototype.hasOwnProperty,r=RegExp("^"+e.call(n).replace(/[\\^$.*+?()[\]{}|]/g,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$");try{var i=e.call(t);return r.test(i)}catch(t){return!1}}function i(t){var e=s(t);if(e){var n=e.childIDs;l(t),n.forEach(i)}}function o(t,e,n){return"\n    in "+(t||"Unknown")+(e?" (at "+e.fileName.replace(/^.*[\\\/]/,"")+":"+e.lineNumber+")":n?" (created by "+n+")":"")}function a(t){return null==t?"#empty":"string"==typeof t||"number"==typeof t?"#text":"string"==typeof t.type?t.type:t.type.displayName||t.type.name||"Unknown"}function u(t){var e,n=E.getDisplayName(t),r=E.getElement(t),i=E.getOwnerID(t);return i&&(e=E.getDisplayName(i)),o(n,r&&r._source,e)}var c,s,l,f,p,h,d,v=n(40),g=n(15),m=(n(0),n(2),"function"==typeof Array.from&&"function"==typeof Map&&r(Map)&&null!=Map.prototype&&"function"==typeof Map.prototype.keys&&r(Map.prototype.keys)&&"function"==typeof Set&&r(Set)&&null!=Set.prototype&&"function"==typeof Set.prototype.keys&&r(Set.prototype.keys));if(m){var y=new Map,_=new Set;c=function(t,e){y.set(t,e)},s=function(t){return y.get(t)},l=function(t){y.delete(t)},f=function(){return Array.from(y.keys())},p=function(t){_.add(t)},h=function(t){_.delete(t)},d=function(){return Array.from(_.keys())}}else{var b={},x={},w=function(t){return"."+t},C=function(t){return parseInt(t.substr(1),10)};c=function(t,e){var n=w(t);b[n]=e},s=function(t){var e=w(t);return b[e]},l=function(t){var e=w(t);delete b[e]},f=function(){return Object.keys(b).map(C)},p=function(t){var e=w(t);x[e]=!0},h=function(t){var e=w(t);delete x[e]},d=function(){return Object.keys(x).map(C)}}var k=[],E={onSetChildren:function(t,e){var n=s(t);n||v("144"),n.childIDs=e;for(var r=0;r<e.length;r++){var i=e[r],o=s(i);o||v("140"),null==o.childIDs&&"object"==typeof o.element&&null!=o.element&&v("141"),o.isMounted||v("71"),null==o.parentID&&(o.parentID=t),o.parentID!==t&&v("142",i,o.parentID,t)}},onBeforeMountComponent:function(t,e,n){c(t,{element:e,parentID:n,text:null,childIDs:[],isMounted:!1,updateCount:0})},onBeforeUpdateComponent:function(t,e){var n=s(t);n&&n.isMounted&&(n.element=e)},onMountComponent:function(t){var e=s(t);e||v("144"),e.isMounted=!0,0===e.parentID&&p(t)},onUpdateComponent:function(t){var e=s(t);e&&e.isMounted&&e.updateCount++},onUnmountComponent:function(t){var e=s(t);if(e){e.isMounted=!1;0===e.parentID&&h(t)}k.push(t)},purgeUnmountedComponents:function(){if(!E._preventPurging){for(var t=0;t<k.length;t++){i(k[t])}k.length=0}},isMounted:function(t){var e=s(t);return!!e&&e.isMounted},getCurrentStackAddendum:function(t){var e="";if(t){var n=a(t),r=t._owner;e+=o(n,t._source,r&&r.getName())}var i=g.current,u=i&&i._debugID;return e+=E.getStackAddendumByID(u)},getStackAddendumByID:function(t){for(var e="";t;)e+=u(t),t=E.getParentID(t);return e},getChildIDs:function(t){var e=s(t);return e?e.childIDs:[]},getDisplayName:function(t){var e=E.getElement(t);return e?a(e):null},getElement:function(t){var e=s(t);return e?e.element:null},getOwnerID:function(t){var e=E.getElement(t);return e&&e._owner?e._owner._debugID:null},getParentID:function(t){var e=s(t);return e?e.parentID:null},getSource:function(t){var e=s(t),n=e?e.element:null;return null!=n?n._source:null},getText:function(t){var e=E.getElement(t);return"string"==typeof e?e:"number"==typeof e?""+e:null},getUpdateCount:function(t){var e=s(t);return e?e.updateCount:0},getRootIDs:d,getRegisteredIDs:f,pushNonStandardWarningStack:function(t,e){if("function"==typeof console.reactStack){var n=[],r=g.current,i=r&&r._debugID;try{for(t&&n.push({name:i?E.getDisplayName(i):null,fileName:e?e.fileName:null,lineNumber:e?e.lineNumber:null});i;){var o=E.getElement(i),a=E.getParentID(i),u=E.getOwnerID(i),c=u?E.getDisplayName(u):null,s=o&&o._source;n.push({name:c,fileName:s?s.fileName:null,lineNumber:s?s.lineNumber:null}),i=a}}catch(t){}console.reactStack(n)}},popNonStandardWarningStack:function(){"function"==typeof console.reactStackEnd&&console.reactStackEnd()}};t.exports=E},function(t,e,n){"use strict";var r="function"==typeof Symbol&&Symbol.for&&Symbol.for("react.element")||60103;t.exports=r},function(t,e,n){"use strict";var r=(n(2),{isMounted:function(t){return!1},enqueueCallback:function(t,e){},enqueueForceUpdate:function(t){},enqueueReplaceState:function(t,e){},enqueueSetState:function(t,e){}});t.exports=r},function(t,e,n){"use strict";var r=!1;t.exports=r},,function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},c=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),s=n(41),l=r(s),f=n(129),p=n(66),h=(n(7),n(29)),d=n(78),v=n(112),g=n(136),m=n(10),y=n(38),_=n(58),b=r(_),x=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return window.lastAdditiveForceArrayVisualizer=t,t.topOffset=28,t.leftOffset=80,t.height=350,t.effectFormat=(0,h.format)(".2"),t.redraw=(0,y.debounce)(function(){return t.draw()},200),t}return a(e,t),c(e,[{key:"componentDidMount",value:function(){var t=this;this.mainGroup=this.svg.append("g"),this.onTopGroup=this.svg.append("g"),this.xaxisElement=this.onTopGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-array-xaxis"),this.yaxisElement=this.onTopGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-array-yaxis"),this.hoverGroup1=this.svg.append("g"),this.hoverGroup2=this.svg.append("g"),this.baseValueTitle=this.svg.append("text"),this.hoverLine=this.svg.append("line"),this.hoverxOutline=this.svg.append("text").attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#fff").attr("stroke","#fff").attr("stroke-width","6").attr("font-size","12px"),this.hoverx=this.svg.append("text").attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#000").attr("font-size","12px"),this.hoverxTitle=this.svg.append("text").attr("text-anchor","middle").attr("opacity",.6).attr("font-size","12px"),this.hoveryOutline=this.svg.append("text").attr("text-anchor","end").attr("font-weight","bold").attr("fill","#fff").attr("stroke","#fff").attr("stroke-width","6").attr("font-size","12px"),this.hovery=this.svg.append("text").attr("text-anchor","end").attr("font-weight","bold").attr("fill","#000").attr("font-size","12px"),this.xlabel=this.wrapper.select(".additive-force-array-xlabel"),this.ylabel=this.wrapper.select(".additive-force-array-ylabel");var e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in b.default.colors?e=b.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=b.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),this.colors=e.map(function(t){return(0,m.hsl)(t)}),this.brighterColors=[1.45,1.6].map(function(e,n){return t.colors[n].brighter(e)});var n=(0,h.format)(",.4");if(null!=this.props.ordering_keys&&null!=this.props.ordering_keys_time_format){var r=function(t){return"object"==(void 0===t?"undefined":u(t))?this.formatTime(t):n(t)};this.parseTime=(0,d.timeParse)(this.props.ordering_keys_time_format),this.formatTime=(0,d.timeFormat)(this.props.ordering_keys_time_format),this.xtickFormat=r}else this.parseTime=null,this.formatTime=null,this.xtickFormat=n;this.xscale=(0,p.scaleLinear)(),this.xaxis=(0,v.axisBottom)().scale(this.xscale).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.xtickFormat(e)}).tickPadding(-18),this.ytickFormat=n,this.yscale=(0,p.scaleLinear)(),this.yaxis=(0,v.axisLeft)().scale(this.yscale).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.ytickFormat(t.invLinkFunction(e))}).tickPadding(2),this.xlabel.node().onchange=function(){return t.internalDraw()},this.ylabel.node().onchange=function(){return t.internalDraw()},this.svg.on("mousemove",function(e){return t.mouseMoved(e)}),this.svg.on("click",function(e){return alert("This original index of the sample you clicked is "+t.nearestExpIndex)}),this.svg.on("mouseout",function(e){return t.mouseOut(e)}),window.addEventListener("resize",this.redraw),window.setTimeout(this.redraw,50)}},{key:"componentDidUpdate",value:function(){this.draw()}},{key:"mouseOut",value:function(){this.hoverLine.attr("display","none"),this.hoverx.attr("display","none"),this.hoverxOutline.attr("display","none"),this.hoverxTitle.attr("display","none"),this.hovery.attr("display","none"),this.hoveryOutline.attr("display","none"),this.hoverGroup1.attr("display","none"),this.hoverGroup2.attr("display","none")}},{key:"mouseMoved",value:function(t){var e=this,n=void 0,r=void 0;this.hoverLine.attr("display",""),this.hoverx.attr("display",""),this.hoverxOutline.attr("display",""),this.hoverxTitle.attr("display",""),this.hovery.attr("display",""),this.hoveryOutline.attr("display",""),this.hoverGroup1.attr("display",""),this.hoverGroup2.attr("display","");var i=(0,f.mouse)(this.svg.node())[0];if(this.props.explanations){for(n=0;n<this.props.explanations.length;++n)(!r||Math.abs(r.xmapScaled-i)>Math.abs(this.props.explanations[n].xmapScaled-i))&&(r=this.props.explanations[n],this.nearestExpIndex=n);this.hoverLine.attr("x1",r.xmapScaled).attr("x2",r.xmapScaled).attr("y1",0+this.topOffset).attr("y2",this.height),this.hoverx.attr("x",r.xmapScaled).attr("y",this.topOffset-5).text(this.xtickFormat(r.xmap)),this.hoverxOutline.attr("x",r.xmapScaled).attr("y",this.topOffset-5).text(this.xtickFormat(r.xmap)),this.hoverxTitle.attr("x",r.xmapScaled).attr("y",this.topOffset-18).text(r.count>1?r.count+" averaged samples":""),this.hovery.attr("x",this.leftOffset-6).attr("y",r.joinPointy).text(this.ytickFormat(this.invLinkFunction(r.joinPoint))),this.hoveryOutline.attr("x",this.leftOffset-6).attr("y",r.joinPointy).text(this.ytickFormat(this.invLinkFunction(r.joinPoint)));for(var o=(this.props.featureNames.length,[]),a=void 0,u=void 0,c=this.currPosOrderedFeatures.length-1;c>=0;--c){var s=this.currPosOrderedFeatures[c],l=r.features[s];u=5+(l.posyTop+l.posyBottom)/2,(!a||u-a>=15)&&l.posyTop-l.posyBottom>=6&&(o.push(l),a=u)}var p=[];a=void 0;var h=!0,d=!1,v=void 0;try{for(var g,m=this.currNegOrderedFeatures[Symbol.iterator]();!(h=(g=m.next()).done);h=!0){var y=g.value,_=r.features[y];u=5+(_.negyTop+_.negyBottom)/2,(!a||a-u>=15)&&_.negyTop-_.negyBottom>=6&&(p.push(_),a=u)}}catch(t){d=!0,v=t}finally{try{!h&&m.return&&m.return()}finally{if(d)throw v}}var b=function(t){var n="";return null!==t.value&&void 0!==t.value&&(n=" = "+(isNaN(t.value)?t.value:e.ytickFormat(t.value))),r.count>1?"mean("+e.props.featureNames[t.ind]+")"+n:e.props.featureNames[t.ind]+n},x=this.hoverGroup1.selectAll(".pos-values").data(o);x.enter().append("text").attr("class","pos-values").merge(x).attr("x",r.xmapScaled+5).attr("y",function(t){return 4+(t.posyTop+t.posyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").attr("opacity",1).text(b),x.exit().remove();var w=this.hoverGroup2.selectAll(".pos-values").data(o);w.enter().append("text").attr("class","pos-values").merge(w).attr("x",r.xmapScaled+5).attr("y",function(t){return 4+(t.posyTop+t.posyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("fill",this.colors[0]).text(b),w.exit().remove();var C=this.hoverGroup1.selectAll(".neg-values").data(p);C.enter().append("text").attr("class","neg-values").merge(C).attr("x",r.xmapScaled+5).attr("y",function(t){return 4+(t.negyTop+t.negyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").attr("opacity",1).text(b),C.exit().remove();var k=this.hoverGroup2.selectAll(".neg-values").data(p);k.enter().append("text").attr("class","neg-values").merge(k).attr("x",r.xmapScaled+5).attr("y",function(t){return 4+(t.negyTop+t.negyBottom)/2}).attr("text-anchor","start").attr("font-size",12).attr("fill",this.colors[1]).text(b),k.exit().remove()}}},{key:"draw",value:function(){var t=this;if(this.props.explanations&&0!==this.props.explanations.length){(0,y.each)(this.props.explanations,function(t,e){return t.origInd=e});var e={},n={},r={},i=!0,o=!1,a=void 0;try{for(var u,c=this.props.explanations[Symbol.iterator]();!(i=(u=c.next()).done);i=!0){var s=u.value;for(var l in s.features)void 0===e[l]&&(e[l]=0,n[l]=0,r[l]=0),s.features[l].effect>0?e[l]+=s.features[l].effect:n[l]-=s.features[l].effect,null!==s.features[l].value&&void 0!==s.features[l].value&&(r[l]+=1)}}catch(t){o=!0,a=t}finally{try{!i&&c.return&&c.return()}finally{if(o)throw a}}this.usedFeatures=(0,y.sortBy)((0,y.keys)(e),function(t){return-(e[t]+n[t])}),console.log("found ",this.usedFeatures.length," used features"),this.posOrderedFeatures=(0,y.sortBy)(this.usedFeatures,function(t){return e[t]}),this.negOrderedFeatures=(0,y.sortBy)(this.usedFeatures,function(t){return-n[t]}),this.singleValueFeatures=(0,y.filter)(this.usedFeatures,function(t){return r[t]>0});var f=["sample order by similarity","sample order by output value","original sample ordering"].concat(this.singleValueFeatures.map(function(e){return t.props.featureNames[e]}));null!=this.props.ordering_keys&&f.unshift("sample order by key");var p=this.xlabel.selectAll("option").data(f);p.enter().append("option").merge(p).attr("value",function(t){return t}).text(function(t){return t}),p.exit().remove();var h=this.props.outNames[0]?this.props.outNames[0]:"model output value";f=(0,y.map)(this.usedFeatures,function(e){return[t.props.featureNames[e],t.props.featureNames[e]+" effects"]}),f.unshift(["model output value",h]);var d=this.ylabel.selectAll("option").data(f);d.enter().append("option").merge(d).attr("value",function(t){return t[0]}).text(function(t){return t[1]}),d.exit().remove(),this.ylabel.style("top",(this.height-10-this.topOffset)/2+this.topOffset+"px").style("left",10-this.ylabel.node().offsetWidth/2+"px"),this.internalDraw()}}},{key:"internalDraw",value:function(){var t=this,e=!0,n=!1,r=void 0;try{for(var i,o=this.props.explanations[Symbol.iterator]();!(e=(i=o.next()).done);e=!0){var a=i.value,u=!0,c=!1,s=void 0;try{for(var l,f=this.usedFeatures[Symbol.iterator]();!(u=(l=f.next()).done);u=!0){var h=l.value;a.features.hasOwnProperty(h)||(a.features[h]={effect:0,value:0}),a.features[h].ind=h}}catch(t){c=!0,s=t}finally{try{!u&&f.return&&f.return()}finally{if(c)throw s}}}}catch(t){n=!0,r=t}finally{try{!e&&o.return&&o.return()}finally{if(n)throw r}}var d=void 0,v=this.xlabel.node().value,m="sample order by key"===v&&null!=this.props.ordering_keys_time_format;if(this.xscale=m?(0,p.scaleTime)():(0,p.scaleLinear)(),this.xaxis.scale(this.xscale),"sample order by similarity"===v)d=(0,y.sortBy)(this.props.explanations,function(t){return t.simIndex}),(0,y.each)(d,function(t,e){return t.xmap=e});else if("sample order by output value"===v)d=(0,y.sortBy)(this.props.explanations,function(t){return-t.outValue}),(0,y.each)(d,function(t,e){return t.xmap=e});else if("original sample ordering"===v)d=(0,y.sortBy)(this.props.explanations,function(t){return t.origInd}),(0,y.each)(d,function(t,e){return t.xmap=e});else if("sample order by key"===v)d=this.props.explanations,m?(0,y.each)(d,function(e,n){return e.xmap=t.parseTime(t.props.ordering_keys[n])}):(0,y.each)(d,function(e,n){return e.xmap=t.props.ordering_keys[n]}),d=(0,y.sortBy)(d,function(t){return t.xmap});else{var _=(0,y.findKey)(this.props.featureNames,function(t){return t===v});(0,y.each)(this.props.explanations,function(t,e){return t.xmap=t.features[_].value});var b=(0,y.sortBy)(this.props.explanations,function(t){return t.xmap}),x=(0,y.map)(b,function(t){return t.xmap});if("string"==typeof x[0])return void alert("Ordering by category names is not yet supported.");var w=(0,y.min)(x),C=(0,y.max)(x),k=(C-w)/100;d=[];for(var E=void 0,M=void 0,T=0;T<b.length;++T){var S=b[T];if(E&&!M&&S.xmap-E.xmap<=k||M&&S.xmap-M.xmap<=k){M||(M=(0,y.cloneDeep)(E),M.count=1);var N=!0,A=!1,P=void 0;try{for(var O,I=this.usedFeatures[Symbol.iterator]();!(N=(O=I.next()).done);N=!0){var D=O.value;M.features[D].effect+=S.features[D].effect,M.features[D].value+=S.features[D].value}}catch(t){A=!0,P=t}finally{try{!N&&I.return&&I.return()}finally{if(A)throw P}}M.count+=1}else if(E)if(M){var R=!0,L=!1,U=void 0;try{for(var F,j=this.usedFeatures[Symbol.iterator]();!(R=(F=j.next()).done);R=!0){var B=F.value;M.features[B].effect/=M.count,M.features[B].value/=M.count}}catch(t){L=!0,U=t}finally{try{!R&&j.return&&j.return()}finally{if(L)throw U}}d.push(M),M=void 0}else d.push(E);E=S}E.xmap-d[d.length-1].xmap>k&&d.push(E)}this.currUsedFeatures=this.usedFeatures,this.currPosOrderedFeatures=this.posOrderedFeatures,this.currNegOrderedFeatures=this.negOrderedFeatures;var V=this.ylabel.node().value;if("model output value"!==V){d=(0,y.cloneDeep)(d);for(var W=(0,y.findKey)(this.props.featureNames,function(t){return t===V}),z=0;z<d.length;++z){var H=d[z].features[W];d[z].features={},d[z].features[W]=H}this.currUsedFeatures=[W],this.currPosOrderedFeatures=[W],this.currNegOrderedFeatures=[W]}this.currExplanations=d,"identity"===this.props.link?this.invLinkFunction=function(e){return t.props.baseValue+e}:"logit"===this.props.link?this.invLinkFunction=function(e){return 1/(1+Math.exp(-(t.props.baseValue+e)))}:console.log("ERROR: Unrecognized link function: ",this.props.link),this.predValues=(0,y.map)(d,function(t){return(0,y.sum)((0,y.map)(t.features,function(t){return t.effect}))});var q=this.wrapper.node().offsetWidth;if(0==q)return setTimeout(function(){return t.draw(d)},500);this.svg.style("height",this.height+"px"),this.svg.style("width",q+"px");var Y=(0,y.map)(d,function(t){return t.xmap});this.xscale.domain([(0,y.min)(Y),(0,y.max)(Y)]).range([this.leftOffset,q]).clamp(!0),this.xaxisElement.attr("transform","translate(0,"+this.topOffset+")").call(this.xaxis);for(var K=0;K<this.currExplanations.length;++K)this.currExplanations[K].xmapScaled=this.xscale(this.currExplanations[K].xmap);for(var G=d.length,$=0,X=0;X<G;++X){var Q=d[X].features,Z=(0,y.sum)((0,y.map)((0,y.filter)(Q,function(t){return t.effect>0}),function(t){return t.effect}))||0,J=(0,y.sum)((0,y.map)((0,y.filter)(Q,function(t){return t.effect<0}),function(t){return-t.effect}))||0;$=Math.max($,2.2*Math.max(Z,J))}this.yscale.domain([-$/2,$/2]).range([this.height-10,this.topOffset]),this.yaxisElement.attr("transform","translate("+this.leftOffset+",0)").call(this.yaxis);for(var tt=0;tt<G;++tt){var et=d[tt].features,nt=((0,y.sum)((0,y.map)(et,function(t){return Math.abs(t.effect)})),(0,y.sum)((0,y.map)((0,y.filter)(et,function(t){return t.effect<0}),function(t){return-t.effect}))||0),rt=-nt,it=void 0,ot=!0,at=!1,ut=void 0;try{for(var ct,st=this.currPosOrderedFeatures[Symbol.iterator]();!(ot=(ct=st.next()).done);ot=!0)it=ct.value,et[it].posyTop=this.yscale(rt),et[it].effect>0&&(rt+=et[it].effect),et[it].posyBottom=this.yscale(rt),et[it].ind=it}catch(t){at=!0,ut=t}finally{try{!ot&&st.return&&st.return()}finally{if(at)throw ut}}var lt=rt,ft=!0,pt=!1,ht=void 0;try{for(var dt,vt=this.currNegOrderedFeatures[Symbol.iterator]();!(ft=(dt=vt.next()).done);ft=!0)it=dt.value,et[it].negyTop=this.yscale(rt),et[it].effect<0&&(rt-=et[it].effect),et[it].negyBottom=this.yscale(rt)}catch(t){pt=!0,ht=t}finally{try{!ft&&vt.return&&vt.return()}finally{if(pt)throw ht}}d[tt].joinPoint=lt,d[tt].joinPointy=this.yscale(lt)}var gt=(0,g.line)().x(function(t){return t[0]}).y(function(t){return t[1]}),mt=this.mainGroup.selectAll(".force-bar-array-area-pos").data(this.currUsedFeatures);mt.enter().append("path").attr("class","force-bar-array-area-pos").merge(mt).attr("d",function(t){var e=(0,y.map)((0,y.range)(G),function(e){return[d[e].xmapScaled,d[e].features[t].posyTop]}),n=(0,y.map)((0,y.rangeRight)(G),function(e){return[d[e].xmapScaled,d[e].features[t].posyBottom]});return gt(e.concat(n))}).attr("fill",this.colors[0]),mt.exit().remove();var yt=this.mainGroup.selectAll(".force-bar-array-area-neg").data(this.currUsedFeatures);yt.enter().append("path").attr("class","force-bar-array-area-neg").merge(yt).attr("d",function(t){var e=(0,y.map)((0,y.range)(G),function(e){return[d[e].xmapScaled,d[e].features[t].negyTop]}),n=(0,y.map)((0,y.rangeRight)(G),function(e){return[d[e].xmapScaled,d[e].features[t].negyBottom]});return gt(e.concat(n))}).attr("fill",this.colors[1]),yt.exit().remove();var _t=this.mainGroup.selectAll(".force-bar-array-divider-pos").data(this.currUsedFeatures);_t.enter().append("path").attr("class","force-bar-array-divider-pos").merge(_t).attr("d",function(t){var e=(0,y.map)((0,y.range)(G),function(e){return[d[e].xmapScaled,d[e].features[t].posyBottom]});return gt(e)}).attr("fill","none").attr("stroke-width",1).attr("stroke",function(e){return t.colors[0].brighter(1.2)}),_t.exit().remove();var bt=this.mainGroup.selectAll(".force-bar-array-divider-neg").data(this.currUsedFeatures);bt.enter().append("path").attr("class","force-bar-array-divider-neg").merge(bt).attr("d",function(t){var e=(0,y.map)((0,y.range)(G),function(e){return[d[e].xmapScaled,d[e].features[t].negyTop]});return gt(e)}).attr("fill","none").attr("stroke-width",1).attr("stroke",function(e){return t.colors[1].brighter(1.5)}),bt.exit().remove();for(var xt=function(t,e,n,r,i){var o=void 0,a=void 0;"pos"===i?(o=t[n].features[e].posyBottom,a=t[n].features[e].posyTop):(o=t[n].features[e].negyBottom,a=t[n].features[e].negyTop);for(var u=void 0,c=void 0,s=n+1;s<=r;++s)"pos"===i?(u=t[s].features[e].posyBottom,c=t[s].features[e].posyTop):(u=t[s].features[e].negyBottom,c=t[s].features[e].negyTop),u>o&&(o=u),c<a&&(a=c);return{top:o,bottom:a}},wt=[],Ct=["pos","neg"],kt=0;kt<Ct.length;kt++){var Et=Ct[kt],Mt=!0,Tt=!1,St=void 0;try{for(var Nt,At=this.currUsedFeatures[Symbol.iterator]();!(Mt=(Nt=At.next()).done);Mt=!0)for(var Pt=Nt.value,Ot=0,It=0,Dt=0,Rt={top:0,bottom:0},Lt=void 0;It<G-1;){for(;Dt<100&&It<G-1;)++It,Dt=d[It].xmapScaled-d[Ot].xmapScaled;for(Rt=xt(d,Pt,Ot,It,Et);Rt.bottom-Rt.top<20&&Ot<It;)++Ot,Rt=xt(d,Pt,Ot,It,Et);if(Dt=d[It].xmapScaled-d[Ot].xmapScaled,Rt.bottom-Rt.top>=20&&Dt>=100){for(;It<G-1;){if(++It,Lt=xt(d,Pt,Ot,It,Et),!(Lt.bottom-Lt.top>20)){--It;break}Rt=Lt}Dt=d[It].xmapScaled-d[Ot].xmapScaled,wt.push([(d[It].xmapScaled+d[Ot].xmapScaled)/2,(Rt.top+Rt.bottom)/2,this.props.featureNames[Pt]]);var Ut=d[It].xmapScaled;for(Ot=It;Ut+100>d[Ot].xmapScaled&&Ot<G-1;)++Ot;It=Ot}}}catch(t){Tt=!0,St=t}finally{try{!Mt&&At.return&&At.return()}finally{if(Tt)throw St}}}var Ft=this.onTopGroup.selectAll(".force-bar-array-flabels").data(wt);Ft.enter().append("text").attr("class","force-bar-array-flabels").merge(Ft).attr("x",function(t){return t[0]}).attr("y",function(t){return t[1]+4}).text(function(t){return t[2]}),Ft.exit().remove()}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.redraw)}},{key:"render",value:function(){var t=this;return l.default.createElement("div",{ref:function(e){return t.wrapper=(0,f.select)(e)},style:{textAlign:"center"}},l.default.createElement("style",{dangerouslySetInnerHTML:{__html:"\n          .force-bar-array-wrapper {\n            text-align: center;\n          }\n          .force-bar-array-xaxis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-array-xaxis .domain {\n            opacity: 0;\n          }\n          .force-bar-array-xaxis paths {\n            display: none;\n          }\n          .force-bar-array-yaxis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-array-yaxis paths {\n            display: none;\n          }\n          .tick line {\n            stroke: #000;\n            stroke-width: 1px;\n            opacity: 0.4;\n          }\n          .tick text {\n            fill: #000;\n            opacity: 0.5;\n            font-size: 12px;\n            padding: 0px;\n          }\n          .force-bar-array-flabels {\n            font-size: 12px;\n            fill: #fff;\n            text-anchor: middle;\n          }\n          .additive-force-array-xlabel {\n            background: none;\n            border: 1px solid #ccc;\n            opacity: 0.5;\n            margin-bottom: 0px;\n            font-size: 12px;\n            font-family: arial;\n            margin-left: 80px;\n            max-width: 300px;\n          }\n          .additive-force-array-xlabel:focus {\n            outline: none;\n          }\n          .additive-force-array-ylabel {\n            position: relative;\n            top: 0px;\n            left: 0px;\n            transform: rotate(-90deg);\n            background: none;\n            border: 1px solid #ccc;\n            opacity: 0.5;\n            margin-bottom: 0px;\n            font-size: 12px;\n            font-family: arial;\n            max-width: 150px;\n          }\n          .additive-force-array-ylabel:focus {\n            outline: none;\n          }\n          .additive-force-array-hoverLine {\n            stroke-width: 1px;\n            stroke: #fff;\n            opacity: 1;\n          }"}}),l.default.createElement("select",{className:"additive-force-array-xlabel"}),l.default.createElement("div",{style:{height:"0px",textAlign:"left"}},l.default.createElement("select",{className:"additive-force-array-ylabel"})),l.default.createElement("svg",{ref:function(e){return t.svg=(0,f.select)(e)},style:{userSelect:"none",display:"block",fontFamily:"arial",sansSerif:!0}}))}}]),e}(l.default.Component);x.defaultProps={plot_cmap:"RdBu",ordering_keys:null,ordering_keys_time_format:null},e.default=x},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),c=n(41),s=r(c),l=n(129),f=n(66),p=(n(7),n(29)),h=n(112),d=n(136),v=n(10),g=n(38),m=n(58),y=r(m),b=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return window.lastAdditiveForceVisualizer=t,t.effectFormat=(0,p.format)(".2"),t.redraw=(0,g.debounce)(function(){return t.draw()},200),t}return a(e,t),u(e,[{key:"componentDidMount",value:function(){var t=this;this.mainGroup=this.svg.append("g"),this.axisElement=this.mainGroup.append("g").attr("transform","translate(0,35)").attr("class","force-bar-axis"),this.onTopGroup=this.svg.append("g"),this.baseValueTitle=this.svg.append("text"),this.joinPointLine=this.svg.append("line"),this.joinPointLabelOutline=this.svg.append("text"),this.joinPointLabel=this.svg.append("text"),this.joinPointTitleLeft=this.svg.append("text"),this.joinPointTitleLeftArrow=this.svg.append("text"),this.joinPointTitle=this.svg.append("text"),this.joinPointTitleRightArrow=this.svg.append("text"),this.joinPointTitleRight=this.svg.append("text"),this.hoverLabelBacking=this.svg.append("text").attr("x",10).attr("y",20).attr("text-anchor","middle").attr("font-size",12).attr("stroke","#fff").attr("fill","#fff").attr("stroke-width","4").attr("stroke-linejoin","round").text("").on("mouseover",function(e){t.hoverLabel.attr("opacity",1),t.hoverLabelBacking.attr("opacity",1)}).on("mouseout",function(e){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)}),this.hoverLabel=this.svg.append("text").attr("x",10).attr("y",20).attr("text-anchor","middle").attr("font-size",12).attr("fill","#0f0").text("").on("mouseover",function(e){t.hoverLabel.attr("opacity",1),t.hoverLabelBacking.attr("opacity",1)}).on("mouseout",function(e){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)});var e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in y.default.colors?e=y.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=y.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),this.colors=e.map(function(t){return(0,v.hsl)(t)}),this.brighterColors=[1.45,1.6].map(function(e,n){return t.colors[n].brighter(e)}),this.colors.map(function(e,n){var r=t.svg.append("linearGradient").attr("id","linear-grad-"+n).attr("x1","0%").attr("y1","0%").attr("x2","0%").attr("y2","100%");r.append("stop").attr("offset","0%").attr("stop-color",e).attr("stop-opacity",.6),r.append("stop").attr("offset","100%").attr("stop-color",e).attr("stop-opacity",0);var i=t.svg.append("linearGradient").attr("id","linear-backgrad-"+n).attr("x1","0%").attr("y1","0%").attr("x2","0%").attr("y2","100%");i.append("stop").attr("offset","0%").attr("stop-color",e).attr("stop-opacity",.5),i.append("stop").attr("offset","100%").attr("stop-color",e).attr("stop-opacity",0)}),this.tickFormat=(0,p.format)(",.4"),this.scaleCentered=(0,f.scaleLinear)(),this.axis=(0,h.axisBottom)().scale(this.scaleCentered).tickSizeInner(4).tickSizeOuter(0).tickFormat(function(e){return t.tickFormat(t.invLinkFunction(e))}).tickPadding(-18),window.addEventListener("resize",this.redraw),window.setTimeout(this.redraw,50)}},{key:"componentDidUpdate",value:function(){this.draw()}},{key:"draw",value:function(){var t=this;(0,g.each)(this.props.featureNames,function(e,n){t.props.features[n]&&(t.props.features[n].name=e)}),"identity"===this.props.link?this.invLinkFunction=function(e){return t.props.baseValue+e}:"logit"===this.props.link?this.invLinkFunction=function(e){return 1/(1+Math.exp(-(t.props.baseValue+e)))}:console.log("ERROR: Unrecognized link function: ",this.props.link);var e=this.svg.node().parentNode.offsetWidth;if(0==e)return setTimeout(function(){return t.draw(t.props)},500);this.svg.style("height","150px"),this.svg.style("width",e+"px");var n=(0,g.sortBy)(this.props.features,function(t){return-1/(t.effect+1e-10)}),r=(0,g.sum)((0,g.map)(n,function(t){return Math.abs(t.effect)})),i=(0,g.sum)((0,g.map)((0,g.filter)(n,function(t){return t.effect>0}),function(t){return t.effect}))||0,o=(0,g.sum)((0,g.map)((0,g.filter)(n,function(t){return t.effect<0}),function(t){return-t.effect}))||0;this.domainSize=3*Math.max(i,o);var a=(0,f.scaleLinear)().domain([0,this.domainSize]).range([0,e]),u=e/2-a(o);this.scaleCentered.domain([-this.domainSize/2,this.domainSize/2]).range([0,e]).clamp(!0),this.axisElement.attr("transform","translate(0,50)").call(this.axis);var c=0,s=void 0,l=void 0,h=void 0;for(s=0;s<n.length;++s)n[s].x=c,n[s].effect<0&&void 0===l&&(l=c,h=s),c+=Math.abs(n[s].effect);void 0===l&&(l=c,h=s);var v=(0,d.line)().x(function(t){return t[0]}).y(function(t){return t[1]}),m=function(e){return void 0!==e.value&&null!==e.value&&""!==e.value?e.name+" = "+(isNaN(e.value)?e.value:t.tickFormat(e.value)):e.name};n=this.props.hideBars?[]:n;var y=this.mainGroup.selectAll(".force-bar-blocks").data(n);y.enter().append("path").attr("class","force-bar-blocks").merge(y).attr("d",function(t,e){var n=a(t.x)+u,r=a(Math.abs(t.effect)),i=t.effect<0?-4:4,o=i;return e===h&&(i=0),e===h-1&&(o=0),v([[n,56],[n+r,56],[n+r+o,64.5],[n+r,73],[n,73],[n+i,64.5]])}).attr("fill",function(e){return e.effect>0?t.colors[0]:t.colors[1]}).on("mouseover",function(e){if(a(Math.abs(e.effect))<a(r)/50||a(Math.abs(e.effect))<10){var n=a(e.x)+u,i=a(Math.abs(e.effect));t.hoverLabel.attr("opacity",1).attr("x",n+i/2).attr("y",50.5).attr("fill",e.effect>0?t.colors[0]:t.colors[1]).text(m(e)),t.hoverLabelBacking.attr("opacity",1).attr("x",n+i/2).attr("y",50.5).text(m(e))}}).on("mouseout",function(e){t.hoverLabel.attr("opacity",0),t.hoverLabelBacking.attr("opacity",0)}),y.exit().remove();var b=_.filter(n,function(t){return a(Math.abs(t.effect))>a(r)/50&&a(Math.abs(t.effect))>10}),x=this.onTopGroup.selectAll(".force-bar-labels").data(b);if(x.exit().remove(),x=x.enter().append("text").attr("class","force-bar-labels").attr("font-size","12px").attr("y",function(t){return 98}).merge(x).text(function(e){return void 0!==e.value&&null!==e.value&&""!==e.value?e.name+" = "+(isNaN(e.value)?e.value:t.tickFormat(e.value)):e.name}).attr("fill",function(e){return e.effect>0?t.colors[0]:t.colors[1]}).attr("stroke",function(t,e){return t.textWidth=Math.max(this.getComputedTextLength(),a(Math.abs(t.effect))-10),t.innerTextWidth=this.getComputedTextLength(),"none"}),this.filteredData=b,n.length>0){c=l+a.invert(5);for(var w=h;w<n.length;++w)n[w].textx=c,c+=a.invert(n[w].textWidth+10);c=l-a.invert(5);for(var C=h-1;C>=0;--C)n[C].textx=c,c-=a.invert(n[C].textWidth+10)}x.attr("x",function(t){return a(t.textx)+u+(t.effect>0?-t.textWidth/2:t.textWidth/2)}).attr("text-anchor","middle"),b=(0,g.filter)(b,function(n){return a(n.textx)+u>t.props.labelMargin&&a(n.textx)+u<e-t.props.labelMargin}),this.filteredData2=b;var k=b.slice(),E=(0,g.findIndex)(n,b[0])-1;E>=0&&k.unshift(n[E]);var M=this.mainGroup.selectAll(".force-bar-labelBacking").data(b);M.enter().append("path").attr("class","force-bar-labelBacking").attr("stroke","none").attr("opacity",.2).merge(M).attr("d",function(t){return v([[a(t.x)+a(Math.abs(t.effect))+u,73],[(t.effect>0?a(t.textx):a(t.textx)+t.textWidth)+u+5,83],[(t.effect>0?a(t.textx):a(t.textx)+t.textWidth)+u+5,104],[(t.effect>0?a(t.textx)-t.textWidth:a(t.textx))+u-5,104],[(t.effect>0?a(t.textx)-t.textWidth:a(t.textx))+u-5,83],[a(t.x)+u,73]])}).attr("fill",function(t){return"url(#linear-backgrad-"+(t.effect>0?0:1)+")"}),M.exit().remove();var T=this.mainGroup.selectAll(".force-bar-labelDividers").data(b.slice(0,-1));T.enter().append("rect").attr("class","force-bar-labelDividers").attr("height","21px").attr("width","1px").attr("y",83).merge(T).attr("x",function(t){return(t.effect>0?a(t.textx):a(t.textx)+t.textWidth)+u+4.5}).attr("fill",function(t){return"url(#linear-grad-"+(t.effect>0?0:1)+")"}),T.exit().remove();var S=this.mainGroup.selectAll(".force-bar-labelLinks").data(b.slice(0,-1));S.enter().append("line").attr("class","force-bar-labelLinks").attr("y1",73).attr("y2",83).attr("stroke-opacity",.5).attr("stroke-width",1).merge(S).attr("x1",function(t){return a(t.x)+a(Math.abs(t.effect))+u}).attr("x2",function(t){return(t.effect>0?a(t.textx):a(t.textx)+t.textWidth)+u+5}).attr("stroke",function(e){return e.effect>0?t.colors[0]:t.colors[1]}),S.exit().remove();var N=this.mainGroup.selectAll(".force-bar-blockDividers").data(n.slice(0,-1));N.enter().append("path").attr("class","force-bar-blockDividers").attr("stroke-width",2).attr("fill","none").merge(N).attr("d",function(t){var e=a(t.x)+a(Math.abs(t.effect))+u;return v([[e,56],[e+(t.effect<0?-4:4),64.5],[e,73]])}).attr("stroke",function(e,n){return h===n+1||Math.abs(e.effect)<1e-8?"#rgba(0,0,0,0)":e.effect>0?t.brighterColors[0]:t.brighterColors[1]}),N.exit().remove(),this.joinPointLine.attr("x1",a(l)+u).attr("x2",a(l)+u).attr("y1",50).attr("y2",56).attr("stroke","#F2F2F2").attr("stroke-width",1).attr("opacity",1),this.joinPointLabelOutline.attr("x",a(l)+u).attr("y",45).attr("color","#fff").attr("text-anchor","middle").attr("font-weight","bold").attr("stroke","#fff").attr("stroke-width",6).text((0,p.format)(",.2f")(this.invLinkFunction(l-o))).attr("opacity",1),console.log("joinPoint",l,u,50,o),this.joinPointLabel.attr("x",a(l)+u).attr("y",45).attr("text-anchor","middle").attr("font-weight","bold").attr("fill","#000").text((0,p.format)(",.2f")(this.invLinkFunction(l-o))).attr("opacity",1),this.joinPointTitle.attr("x",a(l)+u).attr("y",28).attr("text-anchor","middle").attr("font-size","12").attr("fill","#000").text(this.props.outNames[0]).attr("opacity",.5),this.props.hideBars||(this.joinPointTitleLeft.attr("x",a(l)+u-16).attr("y",12).attr("text-anchor","end").attr("font-size","13").attr("fill",this.colors[0]).text("higher").attr("opacity",1),this.joinPointTitleRight.attr("x",a(l)+u+16).attr("y",12).attr("text-anchor","start").attr("font-size","13").attr("fill",this.colors[1]).text("lower").attr("opacity",1),this.joinPointTitleLeftArrow.attr("x",a(l)+u+7).attr("y",8).attr("text-anchor","end").attr("font-size","13").attr("fill",this.colors[0]).text("→").attr("opacity",1),this.joinPointTitleRightArrow.attr("x",a(l)+u-7).attr("y",14).attr("text-anchor","start").attr("font-size","13").attr("fill",this.colors[1]).text("←").attr("opacity",1)),this.props.hideBaseValueLabel||this.baseValueTitle.attr("x",this.scaleCentered(0)).attr("y",28).attr("text-anchor","middle").attr("font-size","12").attr("fill","#000").text("base value").attr("opacity",.5)}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.redraw)}},{key:"render",value:function(){var t=this;return s.default.createElement("svg",{ref:function(e){return t.svg=(0,l.select)(e)},style:{userSelect:"none",display:"block",fontFamily:"arial",sansSerif:!0}},s.default.createElement("style",{dangerouslySetInnerHTML:{__html:"\n          .force-bar-axis path {\n            fill: none;\n            opacity: 0.4;\n          }\n          .force-bar-axis paths {\n            display: none;\n          }\n          .tick line {\n            stroke: #000;\n            stroke-width: 1px;\n            opacity: 0.4;\n          }\n          .tick text {\n            fill: #000;\n            opacity: 0.5;\n            font-size: 12px;\n            padding: 0px;\n          }"}}))}}]),e}(s.default.Component);b.defaultProps={plot_cmap:"RdBu"},e.default=b},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}function i(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}function o(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!=typeof e&&"function"!=typeof e?t:e}function a(t,e){if("function"!=typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}Object.defineProperty(e,"__esModule",{value:!0});var u=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),c=n(41),s=r(c),l=n(66),f=(n(7),n(29)),p=n(38),h=n(58),d=r(h),v=function(t){function e(){i(this,e);var t=o(this,(e.__proto__||Object.getPrototypeOf(e)).call(this));return t.width=100,window.lastSimpleListInstance=t,t.effectFormat=(0,f.format)(".2"),t}return a(e,t),u(e,[{key:"render",value:function(){var t=this,e=void 0;"string"==typeof this.props.plot_cmap?this.props.plot_cmap in d.default.colors?e=d.default.colors[this.props.plot_cmap]:(console.log("Invalid color map name, reverting to default."),e=d.default.colors.RdBu):Array.isArray(this.props.plot_cmap)&&(e=this.props.plot_cmap),console.log(this.props.features,this.props.features),this.scale=(0,l.scaleLinear)().domain([0,(0,p.max)((0,p.map)(this.props.features,function(t){return Math.abs(t.effect)}))]).range([0,this.width]);var n=(0,p.reverse)((0,p.sortBy)(Object.keys(this.props.features),function(e){return Math.abs(t.props.features[e].effect)})),r=n.map(function(n){var r=t.props.features[n],i=t.props.featureNames[n],o={width:t.scale(Math.abs(r.effect)),height:"20px",background:r.effect<0?e[0]:e[1],display:"inline-block"},a=void 0,u=void 0,c={lineHeight:"20px",display:"inline-block",width:t.width+40,verticalAlign:"top",marginRight:"5px",textAlign:"right"},l={lineHeight:"20px",display:"inline-block",width:t.width+40,verticalAlign:"top",marginLeft:"5px"};return r.effect<0?(u=s.default.createElement("span",{style:l},i),c.width=40+t.width-t.scale(Math.abs(r.effect)),c.textAlign="right",c.color="#999",c.fontSize="13px",a=s.default.createElement("span",{style:c},t.effectFormat(r.effect))):(c.textAlign="right",a=s.default.createElement("span",{style:c},i),l.width=40,l.textAlign="left",l.color="#999",l.fontSize="13px",u=s.default.createElement("span",{style:l},t.effectFormat(r.effect))),s.default.createElement("div",{key:n,style:{marginTop:"2px"}},a,s.default.createElement("div",{style:o}),u)});return s.default.createElement("span",null,r)}}]),e}(s.default.Component);v.defaultProps={plot_cmap:"RdBu"},e.default=v},function(t,e,n){"use strict";t.exports=n(359)},function(t,e,n){var r=(n(0),n(411)),i=!1;t.exports=function(t){t=t||{};var e=t.shouldRejectClick||r;i=!0,n(22).injection.injectEventPluginsByName({TapEventPlugin:n(409)(e)})}},function(t,e,n){"use strict";function r(t){return t&&t.__esModule?t:{default:t}}var i=n(41),o=r(i),a=n(187),u=r(a),c=n(188),s=r(c),l=n(186),f=r(l),p=n(185),h=r(p),d=n(184),v=r(d);(0,s.default)(),window.SHAP={SimpleListVisualizer:f.default,AdditiveForceVisualizer:h.default,AdditiveForceArrayVisualizer:v.default,React:o.default,ReactDom:u.default}},,function(t,e,n){"use strict";function r(t){return t}function i(t,e,n){function i(t,e){var n=y.hasOwnProperty(e)?y[e]:null;C.hasOwnProperty(e)&&u("OVERRIDE_BASE"===n,"ReactClassInterface: You are attempting to override `%s` from your class specification. Ensure that your method names do not overlap with React methods.",e),t&&u("DEFINE_MANY"===n||"DEFINE_MANY_MERGED"===n,"ReactClassInterface: You are attempting to define `%s` on your component more than once. This conflict may be due to a mixin.",e)}function s(t,n){if(n){u("function"!=typeof n,"ReactClass: You're attempting to use a component class or function as a mixin. Instead, just use a regular object."),u(!e(n),"ReactClass: You're attempting to use a component as a mixin. Instead, just use a regular object.");var r=t.prototype,o=r.__reactAutoBindPairs;n.hasOwnProperty(c)&&b.mixins(t,n.mixins);for(var a in n)if(n.hasOwnProperty(a)&&a!==c){var s=n[a],l=r.hasOwnProperty(a);if(i(l,a),b.hasOwnProperty(a))b[a](t,s);else{var f=y.hasOwnProperty(a),d="function"==typeof s,v=d&&!f&&!l&&!1!==n.autobind;if(v)o.push(a,s),r[a]=s;else if(l){var g=y[a];u(f&&("DEFINE_MANY_MERGED"===g||"DEFINE_MANY"===g),"ReactClass: Unexpected spec policy %s for key %s when mixing in component specs.",g,a),"DEFINE_MANY_MERGED"===g?r[a]=p(r[a],s):"DEFINE_MANY"===g&&(r[a]=h(r[a],s))}else r[a]=s}}}else;}function l(t,e){if(e)for(var n in e){var r=e[n];if(e.hasOwnProperty(n)){var i=n in b;u(!i,'ReactClass: You are attempting to define a reserved property, `%s`, that shouldn\'t be on the "statics" key. Define it as an instance property instead; it will still be accessible on the constructor.',n);var o=n in t;if(o){var a=_.hasOwnProperty(n)?_[n]:null;return u("DEFINE_MANY_MERGED"===a,"ReactClass: You are attempting to define `%s` on your component more than once. This conflict may be due to a mixin.",n),void(t[n]=p(t[n],r))}t[n]=r}}}function f(t,e){u(t&&e&&"object"==typeof t&&"object"==typeof e,"mergeIntoWithNoDuplicateKeys(): Cannot merge non-objects.");for(var n in e)e.hasOwnProperty(n)&&(u(void 0===t[n],"mergeIntoWithNoDuplicateKeys(): Tried to merge two objects with the same key: `%s`. This conflict may be due to a mixin; in particular, this may be caused by two getInitialState() or getDefaultProps() methods returning objects with clashing keys.",n),t[n]=e[n]);return t}function p(t,e){return function(){var n=t.apply(this,arguments),r=e.apply(this,arguments);if(null==n)return r;if(null==r)return n;var i={};return f(i,n),f(i,r),i}}function h(t,e){return function(){t.apply(this,arguments),e.apply(this,arguments)}}function d(t,e){var n=e.bind(t);return n}function v(t){for(var e=t.__reactAutoBindPairs,n=0;n<e.length;n+=2){var r=e[n],i=e[n+1];t[r]=d(t,i)}}function g(t){var e=r(function(t,r,i){this.__reactAutoBindPairs.length&&v(this),this.props=t,this.context=r,this.refs=a,this.updater=i||n,this.state=null;var o=this.getInitialState?this.getInitialState():null;u("object"==typeof o&&!Array.isArray(o),"%s.getInitialState(): must return an object or null",e.displayName||"ReactCompositeComponent"),this.state=o});e.prototype=new k,e.prototype.constructor=e,e.prototype.__reactAutoBindPairs=[],m.forEach(s.bind(null,e)),s(e,x),s(e,t),s(e,w),e.getDefaultProps&&(e.defaultProps=e.getDefaultProps()),u(e.prototype.render,"createClass(...): Class specification must implement a `render` method.");for(var i in y)e.prototype[i]||(e.prototype[i]=null);return e}var m=[],y={mixins:"DEFINE_MANY",statics:"DEFINE_MANY",propTypes:"DEFINE_MANY",contextTypes:"DEFINE_MANY",childContextTypes:"DEFINE_MANY",getDefaultProps:"DEFINE_MANY_MERGED",getInitialState:"DEFINE_MANY_MERGED",getChildContext:"DEFINE_MANY_MERGED",render:"DEFINE_ONCE",componentWillMount:"DEFINE_MANY",componentDidMount:"DEFINE_MANY",componentWillReceiveProps:"DEFINE_MANY",shouldComponentUpdate:"DEFINE_ONCE",componentWillUpdate:"DEFINE_MANY",componentDidUpdate:"DEFINE_MANY",componentWillUnmount:"DEFINE_MANY",UNSAFE_componentWillMount:"DEFINE_MANY",UNSAFE_componentWillReceiveProps:"DEFINE_MANY",UNSAFE_componentWillUpdate:"DEFINE_MANY",updateComponent:"OVERRIDE_BASE"},_={getDerivedStateFromProps:"DEFINE_MANY_MERGED"},b={displayName:function(t,e){t.displayName=e},mixins:function(t,e){if(e)for(var n=0;n<e.length;n++)s(t,e[n])},childContextTypes:function(t,e){t.childContextTypes=o({},t.childContextTypes,e)},contextTypes:function(t,e){t.contextTypes=o({},t.contextTypes,e)},getDefaultProps:function(t,e){t.getDefaultProps?t.getDefaultProps=p(t.getDefaultProps,e):t.getDefaultProps=e},propTypes:function(t,e){t.propTypes=o({},t.propTypes,e)},statics:function(t,e){l(t,e)},autobind:function(){}},x={componentDidMount:function(){this.__isMounted=!0}},w={componentWillUnmount:function(){this.__isMounted=!1}},C={replaceState:function(t,e){this.updater.enqueueReplaceState(this,t,e)},isMounted:function(){return!!this.__isMounted}},k=function(){};return o(k.prototype,t.prototype,C),g}var o=n(3),a=n(51),u=n(0),c="mixins";t.exports=i},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(106);e.a=function(t,e,n){var i,o,a,u,c=t.length,s=e.length,l=new Array(c*s);for(null==n&&(n=r.b),i=a=0;i<c;++i)for(u=t[i],o=0;o<s;++o,++a)l[a]=n(u,e[o]);return l}},function(t,e,n){"use strict";e.a=function(t,e){return e<t?-1:e>t?1:e>=t?0:NaN}},function(t,e,n){"use strict";var r=n(100),i=n(101),o=n(192),a=n(104),u=n(196),c=n(107),s=n(109),l=n(108);e.a=function(){function t(t){var r,o,a=t.length,u=new Array(a);for(r=0;r<a;++r)u[r]=e(t[r],r,t);var l=f(u),h=l[0],d=l[1],v=p(u,h,d);Array.isArray(v)||(v=n.i(s.c)(h,d,v),v=n.i(c.a)(Math.ceil(h/v)*v,Math.floor(d/v)*v,v));for(var g=v.length;v[0]<=h;)v.shift(),--g;for(;v[g-1]>d;)v.pop(),--g;var m,y=new Array(g+1);for(r=0;r<=g;++r)m=y[r]=[],m.x0=r>0?v[r-1]:h,m.x1=r<g?v[r]:d;for(r=0;r<a;++r)o=u[r],h<=o&&o<=d&&y[n.i(i.a)(v,o,0,g)].push(t[r]);return y}var e=u.a,f=a.a,p=l.a;return t.value=function(r){return arguments.length?(e="function"==typeof r?r:n.i(o.a)(r),t):e},t.domain=function(e){return arguments.length?(f="function"==typeof e?e:n.i(o.a)([e[0],e[1]]),t):f},t.thresholds=function(e){return arguments.length?(p="function"==typeof e?e:Array.isArray(e)?n.i(o.a)(r.b.call(e)):n.i(o.a)(e),t):p},t}},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i=t.length,o=-1;if(null==e){for(;++o<i;)if(null!=(n=t[o])&&n>=n)for(r=n;++o<i;)null!=(n=t[o])&&n>r&&(r=n)}else for(;++o<i;)if(null!=(n=e(t[o],o,t))&&n>=n)for(r=n;++o<i;)null!=(n=e(t[o],o,t))&&n>r&&(r=n);return r}},function(t,e,n){"use strict";var r=n(28);e.a=function(t,e){var i,o=t.length,a=o,u=-1,c=0;if(null==e)for(;++u<o;)isNaN(i=n.i(r.a)(t[u]))?--a:c+=i;else for(;++u<o;)isNaN(i=n.i(r.a)(e(t[u],u,t)))?--a:c+=i;if(a)return c/a}},function(t,e,n){"use strict";var r=n(19),i=n(28),o=n(59);e.a=function(t,e){var a,u=t.length,c=-1,s=[];if(null==e)for(;++c<u;)isNaN(a=n.i(i.a)(t[c]))||s.push(a);else for(;++c<u;)isNaN(a=n.i(i.a)(e(t[c],c,t)))||s.push(a);return n.i(o.a)(s.sort(r.a),.5)}},function(t,e,n){"use strict";e.a=function(t){for(var e,n,r,i=t.length,o=-1,a=0;++o<i;)a+=t[o].length;for(n=new Array(a);--i>=0;)for(r=t[i],e=r.length;--e>=0;)n[--a]=r[e];return n}},function(t,e,n){"use strict";e.a=function(t,e){for(var n=e.length,r=new Array(n);n--;)r[n]=t[e[n]];return r}},function(t,e,n){"use strict";var r=n(19);e.a=function(t,e){if(n=t.length){var n,i,o=0,a=0,u=t[a];for(null==e&&(e=r.a);++o<n;)(e(i=t[o],u)<0||0!==e(u,u))&&(u=i,a=o);return 0===e(u,u)?a:void 0}}},function(t,e,n){"use strict";e.a=function(t,e,n){for(var r,i,o=(null==n?t.length:n)-(e=null==e?0:+e);o;)i=Math.random()*o--|0,r=t[o+e],t[o+e]=t[i+e],t[i+e]=r;return t}},function(t,e,n){"use strict";e.a=function(t,e){var n,r=t.length,i=-1,o=0;if(null==e)for(;++i<r;)(n=+t[i])&&(o+=n);else for(;++i<r;)(n=+e(t[i],i,t))&&(o+=n);return o}},function(t,e,n){"use strict";var r=n(100),i=n(19),o=n(28),a=n(59);e.a=function(t,e,u){return t=r.a.call(t,o.a).sort(i.a),Math.ceil((u-e)/(2*(n.i(a.a)(t,.75)-n.i(a.a)(t,.25))*Math.pow(t.length,-1/3)))}},function(t,e,n){"use strict";var r=n(103);e.a=function(t,e,i){return Math.ceil((i-e)/(3.5*n.i(r.a)(t)*Math.pow(t.length,-1/3)))}},function(t,e,n){"use strict";var r=n(110);e.a=function(){return n.i(r.a)(arguments)}},function(t,e,n){"use strict";n.d(e,"a",function(){return r});var r=Array.prototype.slice},function(t,e,n){"use strict";function r(t){return"translate("+(t+.5)+",0)"}function i(t){return"translate(0,"+(t+.5)+")"}function o(t){return function(e){return+t(e)}}function a(t){var e=Math.max(0,t.bandwidth()-1)/2;return t.round()&&(e=Math.round(e)),function(n){return+t(n)+e}}function u(){return!this.__axis}function c(t,e){function n(n){var r=null==s?e.ticks?e.ticks.apply(e,c):e.domain():s,i=null==l?e.tickFormat?e.tickFormat.apply(e,c):d.a:l,h=Math.max(f,0)+b,k=e.range(),E=+k[0]+.5,M=+k[k.length-1]+.5,T=(e.bandwidth?a:o)(e.copy()),S=n.selection?n.selection():n,N=S.selectAll(".domain").data([null]),A=S.selectAll(".tick").data(r,e).order(),P=A.exit(),O=A.enter().append("g").attr("class","tick"),I=A.select("line"),D=A.select("text");N=N.merge(N.enter().insert("path",".tick").attr("class","domain").attr("stroke","#000")),A=A.merge(O),I=I.merge(O.append("line").attr("stroke","#000").attr(w+"2",x*f)),D=D.merge(O.append("text").attr("fill","#000").attr(w,x*h).attr("dy",t===v?"0em":t===m?"0.71em":"0.32em")),n!==S&&(N=N.transition(n),A=A.transition(n),I=I.transition(n),D=D.transition(n),P=P.transition(n).attr("opacity",_).attr("transform",function(t){return isFinite(t=T(t))?C(t):this.getAttribute("transform")}),O.attr("opacity",_).attr("transform",function(t){var e=this.parentNode.__axis;return C(e&&isFinite(e=e(t))?e:T(t))})),P.remove(),N.attr("d",t===y||t==g?"M"+x*p+","+E+"H0.5V"+M+"H"+x*p:"M"+E+","+x*p+"V0.5H"+M+"V"+x*p),A.attr("opacity",1).attr("transform",function(t){return C(T(t))}),I.attr(w+"2",x*f),D.attr(w,x*h).text(i),S.filter(u).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===g?"start":t===y?"end":"middle"),S.each(function(){this.__axis=T})}var c=[],s=null,l=null,f=6,p=6,b=3,x=t===v||t===y?-1:1,w=t===y||t===g?"x":"y",C=t===v||t===m?r:i;return n.scale=function(t){return arguments.length?(e=t,n):e},n.ticks=function(){return c=h.a.call(arguments),n},n.tickArguments=function(t){return arguments.length?(c=null==t?[]:h.a.call(t),n):c.slice()},n.tickValues=function(t){return arguments.length?(s=null==t?null:h.a.call(t),n):s&&s.slice()},n.tickFormat=function(t){return arguments.length?(l=t,n):l},n.tickSize=function(t){return arguments.length?(f=p=+t,n):f},n.tickSizeInner=function(t){return arguments.length?(f=+t,n):f},n.tickSizeOuter=function(t){return arguments.length?(p=+t,n):p},n.tickPadding=function(t){return arguments.length?(b=+t,n):b},n}function s(t){return c(v,t)}function l(t){return c(g,t)}function f(t){return c(m,t)}function p(t){return c(y,t)}e.a=s,e.b=l,e.c=f,e.d=p;var h=n(208),d=n(210),v=1,g=2,m=3,y=4,_=1e-6},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";var r=(n(214),n(215),n(60));n.d(e,"a",function(){return r.a});n(213),n(216),n(212)},function(t,e,n){"use strict"},function(t,e,n){"use strict"},function(t,e,n){"use strict";n(60)},function(t,e,n){"use strict";function r(){}function i(t,e){var n=new r;if(t instanceof r)t.each(function(t){n.add(t)});else if(t){var i=-1,o=t.length;if(null==e)for(;++i<o;)n.add(t[i]);else for(;++i<o;)n.add(e(t[i],i,t))}return n}var o=n(60),a=o.a.prototype;r.prototype=i.prototype={constructor:r,has:a.has,add:function(t){return t+="",this[o.b+t]=t,this},remove:a.remove,clear:a.clear,values:a.keys,size:a.size,empty:a.empty,each:a.each}},function(t,e,n){"use strict"},function(t,e,n){"use strict";function r(t){if(t instanceof o)return new o(t.h,t.s,t.l,t.opacity);t instanceof u.d||(t=n.i(u.e)(t));var e=t.r/255,r=t.g/255,i=t.b/255,a=(g*i+d*e-v*r)/(g+d-v),s=i-a,l=(h*(r-a)-f*s)/p,m=Math.sqrt(l*l+s*s)/(h*a*(1-a)),y=m?Math.atan2(l,s)*c.a-120:NaN;return new o(y<0?y+360:y,m,a,t.opacity)}function i(t,e,n,i){return 1===arguments.length?r(t):new o(t,e,n,null==i?1:i)}function o(t,e,n,r){this.h=+t,this.s=+e,this.l=+n,this.opacity=+r}e.a=i;var a=n(62),u=n(61),c=n(113),s=-.14861,l=1.78277,f=-.29227,p=-.90649,h=1.97294,d=h*p,v=h*l,g=l*f-p*s;n.i(a.a)(o,i,n.i(a.b)(u.f,{brighter:function(t){return t=null==t?u.g:Math.pow(u.g,t),new o(this.h,this.s,this.l*t,this.opacity)},darker:function(t){return t=null==t?u.h:Math.pow(u.h,t),new o(this.h,this.s,this.l*t,this.opacity)},rgb:function(){var t=isNaN(this.h)?0:(this.h+120)*c.b,e=+this.l,n=isNaN(this.s)?0:this.s*e*(1-e),r=Math.cos(t),i=Math.sin(t);return new u.d(255*(e+n*(s*r+l*i)),255*(e+n*(f*r+p*i)),255*(e+n*(h*r)),this.opacity)}}))},function(t,e,n){"use strict";function r(t){if(t instanceof o)return new o(t.l,t.a,t.b,t.opacity);if(t instanceof p){var e=t.h*v.b;return new o(t.l,Math.cos(e)*t.c,Math.sin(e)*t.c,t.opacity)}t instanceof d.d||(t=n.i(d.e)(t));var r=s(t.r),i=s(t.g),u=s(t.b),c=a((.4124564*r+.3575761*i+.1804375*u)/g),l=a((.2126729*r+.7151522*i+.072175*u)/m);return new o(116*l-16,500*(c-l),200*(l-a((.0193339*r+.119192*i+.9503041*u)/y)),t.opacity)}function i(t,e,n,i){return 1===arguments.length?r(t):new o(t,e,n,null==i?1:i)}function o(t,e,n,r){this.l=+t,this.a=+e,this.b=+n,this.opacity=+r}function a(t){return t>w?Math.pow(t,1/3):t/x+_}function u(t){return t>b?t*t*t:x*(t-_)}function c(t){return 255*(t<=.0031308?12.92*t:1.055*Math.pow(t,1/2.4)-.055)}function s(t){return(t/=255)<=.04045?t/12.92:Math.pow((t+.055)/1.055,2.4)}function l(t){if(t instanceof p)return new p(t.h,t.c,t.l,t.opacity);t instanceof o||(t=r(t));var e=Math.atan2(t.b,t.a)*v.a;return new p(e<0?e+360:e,Math.sqrt(t.a*t.a+t.b*t.b),t.l,t.opacity)}function f(t,e,n,r){return 1===arguments.length?l(t):new p(t,e,n,null==r?1:r)}function p(t,e,n,r){this.h=+t,this.c=+e,this.l=+n,this.opacity=+r}e.a=i,e.b=f;var h=n(62),d=n(61),v=n(113),g=.95047,m=1,y=1.08883,_=4/29,b=6/29,x=3*b*b,w=b*b*b;n.i(h.a)(o,i,n.i(h.b)(d.f,{brighter:function(t){return new o(this.l+18*(null==t?1:t),this.a,this.b,this.opacity)},darker:function(t){return new o(this.l-18*(null==t?1:t),this.a,this.b,this.opacity)},rgb:function(){var t=(this.l+16)/116,e=isNaN(this.a)?t:t+this.a/500,n=isNaN(this.b)?t:t-this.b/200;return t=m*u(t),e=g*u(e),n=y*u(n),new d.d(c(3.2404542*e-1.5371385*t-.4985314*n),c(-.969266*e+1.8760108*t+.041556*n),c(.0556434*e-.2040259*t+1.0572252*n),this.opacity)}})),n.i(h.a)(p,f,n.i(h.b)(d.f,{brighter:function(t){return new p(this.h,this.c,this.l+18*(null==t?1:t),this.opacity)},darker:function(t){return new p(this.h,this.c,this.l-18*(null==t?1:t),this.opacity)},rgb:function(){return r(this).rgb()}}))},function(t,e,n){"use strict";function r(t){return i=n.i(u.a)(t),o=i.format,a=i.formatPrefix,i}n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a}),e.a=r;var i,o,a,u=n(117);r({decimal:".",thousands:",",grouping:[3],currency:["$",""]})},function(t,e,n){"use strict";e.a=function(t,e){t=t.toPrecision(e);t:for(var n,r=t.length,i=1,o=-1;i<r;++i)switch(t[i]){case".":o=n=i;break;case"0":0===o&&(o=i),n=i;break;case"e":break t;default:o>0&&(o=0)}return o>0?t.slice(0,o)+t.slice(n+1):t}},function(t,e,n){"use strict";e.a=function(t,e){return function(n,r){for(var i=n.length,o=[],a=0,u=t[0],c=0;i>0&&u>0&&(c+u+1>r&&(u=Math.max(1,r-c)),o.push(n.substring(i-=u,i+u)),!((c+=u+1)>r));)u=t[a=(a+1)%t.length];return o.reverse().join(e)}}},function(t,e,n){"use strict";e.a=function(t){return function(e){return e.replace(/[0-9]/g,function(e){return t[+e]})}}},function(t,e,n){"use strict";var r=n(63);e.a=function(t,e){var i=n.i(r.a)(t,e);if(!i)return t+"";var o=i[0],a=i[1];return a<0?"0."+new Array(-a).join("0")+o:o.length>a+1?o.slice(0,a+1)+"."+o.slice(a+1):o+new Array(a-o.length+2).join("0")}},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";var r=n(42);e.a=function(t){return Math.max(0,-n.i(r.a)(Math.abs(t)))}},function(t,e,n){"use strict";var r=n(42);e.a=function(t,e){return Math.max(0,3*Math.max(-8,Math.min(8,Math.floor(n.i(r.a)(e)/3)))-n.i(r.a)(Math.abs(t)))}},function(t,e,n){"use strict";var r=n(42);e.a=function(t,e){return t=Math.abs(t),e=Math.abs(e)-t,Math.max(0,n.i(r.a)(e)-n.i(r.a)(t))+1}},function(t,e,n){"use strict";function r(t){return function e(r){function a(e,a){var u=t((e=n.i(i.cubehelix)(e)).h,(a=n.i(i.cubehelix)(a)).h),c=n.i(o.a)(e.s,a.s),s=n.i(o.a)(e.l,a.l),l=n.i(o.a)(e.opacity,a.opacity);return function(t){return e.h=u(t),e.s=c(t),e.l=s(Math.pow(t,r)),e.opacity=l(t),e+""}}return r=+r,a.gamma=e,a}(1)}n.d(e,"a",function(){return a});var i=n(10),o=n(31),a=(r(o.b),r(o.a))},function(t,e,n){"use strict";function r(t){return function(e,r){var a=t((e=n.i(i.hcl)(e)).h,(r=n.i(i.hcl)(r)).h),u=n.i(o.a)(e.c,r.c),c=n.i(o.a)(e.l,r.l),s=n.i(o.a)(e.opacity,r.opacity);return function(t){return e.h=a(t),e.c=u(t),e.l=c(t),e.opacity=s(t),e+""}}}var i=n(10),o=n(31);r(o.b),r(o.a)},function(t,e,n){"use strict";function r(t){return function(e,r){var a=t((e=n.i(i.hsl)(e)).h,(r=n.i(i.hsl)(r)).h),u=n.i(o.a)(e.s,r.s),c=n.i(o.a)(e.l,r.l),s=n.i(o.a)(e.opacity,r.opacity);return function(t){return e.h=a(t),e.s=u(t),e.l=c(t),e.opacity=s(t),e+""}}}var i=n(10),o=n(31);r(o.b),r(o.a)},function(t,e,n){"use strict";n(10),n(31)},function(t,e,n){"use strict"},function(t,e,n){"use strict";e.a=function(t,e){return t=+t,e-=t,function(n){return Math.round(t+e*n)}}},function(t,e,n){"use strict";n.d(e,"a",function(){return i});var r=180/Math.PI,i={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1};e.b=function(t,e,n,i,o,a){var u,c,s;return(u=Math.sqrt(t*t+e*e))&&(t/=u,e/=u),(s=t*n+e*i)&&(n-=t*s,i-=e*s),(c=Math.sqrt(n*n+i*i))&&(n/=c,i/=c,s/=c),t*i<e*n&&(t=-t,e=-e,s=-s,u=-u),{translateX:o,translateY:a,rotate:Math.atan2(e,t)*r,skewX:Math.atan(s)*r,scaleX:u,scaleY:c}}},function(t,e,n){"use strict";function r(t,e,r,o){function a(t){return t.length?t.pop()+" ":""}function u(t,o,a,u,c,s){if(t!==a||o!==u){var l=c.push("translate(",null,e,null,r);s.push({i:l-4,x:n.i(i.a)(t,a)},{i:l-2,x:n.i(i.a)(o,u)})}else(a||u)&&c.push("translate("+a+e+u+r)}function c(t,e,r,u){t!==e?(t-e>180?e+=360:e-t>180&&(t+=360),u.push({i:r.push(a(r)+"rotate(",null,o)-2,x:n.i(i.a)(t,e)})):e&&r.push(a(r)+"rotate("+e+o)}function s(t,e,r,u){t!==e?u.push({i:r.push(a(r)+"skewX(",null,o)-2,x:n.i(i.a)(t,e)}):e&&r.push(a(r)+"skewX("+e+o)}function l(t,e,r,o,u,c){if(t!==r||e!==o){var s=u.push(a(u)+"scale(",null,",",null,")");c.push({i:s-4,x:n.i(i.a)(t,r)},{i:s-2,x:n.i(i.a)(e,o)})}else 1===r&&1===o||u.push(a(u)+"scale("+r+","+o+")")}return function(e,n){var r=[],i=[];return e=t(e),n=t(n),u(e.translateX,e.translateY,n.translateX,n.translateY,r,i),c(e.rotate,n.rotate,r,i),s(e.skewX,n.skewX,r,i),l(e.scaleX,e.scaleY,n.scaleX,n.scaleY,r,i),e=n=null,function(t){for(var e,n=-1,o=i.length;++n<o;)r[(e=i[n]).i]=e.x(t);return r.join("")}}}var i=n(43),o=n(236);r(o.a,"px, ","px)","deg)"),r(o.b,", ",")",")")},function(t,e,n){"use strict";function r(t){return"none"===t?s.a:(o||(o=document.createElement("DIV"),a=document.documentElement,u=document.defaultView),o.style.transform=t,t=u.getComputedStyle(a.appendChild(o),null).getPropertyValue("transform"),a.removeChild(o),t=t.slice(7,-1).split(","),n.i(s.b)(+t[0],+t[1],+t[2],+t[3],+t[4],+t[5]))}function i(t){return null==t?s.a:(c||(c=document.createElementNS("http://www.w3.org/2000/svg","g")),c.setAttribute("transform",t),(t=c.transform.baseVal.consolidate())?(t=t.matrix,n.i(s.b)(t.a,t.b,t.c,t.d,t.e,t.f)):s.a)}e.a=r,e.b=i;var o,a,u,c,s=n(234)},function(t,e,n){"use strict";Math.SQRT2},function(t,e,n){"use strict";function r(){this._x0=this._y0=this._x1=this._y1=null,this._=""}function i(){return new r}var o=Math.PI,a=2*o,u=a-1e-6;r.prototype=i.prototype={constructor:r,moveTo:function(t,e){this._+="M"+(this._x0=this._x1=+t)+","+(this._y0=this._y1=+e)},closePath:function(){null!==this._x1&&(this._x1=this._x0,this._y1=this._y0,this._+="Z")},lineTo:function(t,e){this._+="L"+(this._x1=+t)+","+(this._y1=+e)},quadraticCurveTo:function(t,e,n,r){this._+="Q"+ +t+","+ +e+","+(this._x1=+n)+","+(this._y1=+r)},bezierCurveTo:function(t,e,n,r,i,o){this._+="C"+ +t+","+ +e+","+ +n+","+ +r+","+(this._x1=+i)+","+(this._y1=+o)},arcTo:function(t,e,n,r,i){t=+t,e=+e,n=+n,r=+r,i=+i;var a=this._x1,u=this._y1,c=n-t,s=r-e,l=a-t,f=u-e,p=l*l+f*f;if(i<0)throw new Error("negative radius: "+i);if(null===this._x1)this._+="M"+(this._x1=t)+","+(this._y1=e);else if(p>1e-6)if(Math.abs(f*c-s*l)>1e-6&&i){var h=n-a,d=r-u,v=c*c+s*s,g=h*h+d*d,m=Math.sqrt(v),y=Math.sqrt(p),_=i*Math.tan((o-Math.acos((v+p-g)/(2*m*y)))/2),b=_/y,x=_/m;Math.abs(b-1)>1e-6&&(this._+="L"+(t+b*l)+","+(e+b*f)),this._+="A"+i+","+i+",0,0,"+ +(f*h>l*d)+","+(this._x1=t+x*c)+","+(this._y1=e+x*s)}else this._+="L"+(this._x1=t)+","+(this._y1=e);else;},arc:function(t,e,n,r,i,c){t=+t,e=+e,n=+n;var s=n*Math.cos(r),l=n*Math.sin(r),f=t+s,p=e+l,h=1^c,d=c?r-i:i-r;if(n<0)throw new Error("negative radius: "+n);null===this._x1?this._+="M"+f+","+p:(Math.abs(this._x1-f)>1e-6||Math.abs(this._y1-p)>1e-6)&&(this._+="L"+f+","+p),n&&(d<0&&(d=d%a+a),d>u?this._+="A"+n+","+n+",0,1,"+h+","+(t-s)+","+(e-l)+"A"+n+","+n+",0,1,"+h+","+(this._x1=f)+","+(this._y1=p):d>1e-6&&(this._+="A"+n+","+n+",0,"+ +(d>=o)+","+h+","+(this._x1=t+n*Math.cos(i))+","+(this._y1=e+n*Math.sin(i))))},rect:function(t,e,n,r){this._+="M"+(this._x0=this._x1=+t)+","+(this._y0=this._y1=+e)+"h"+ +n+"v"+ +r+"h"+-n+"Z"},toString:function(){return this._}},e.a=i},function(t,e,n){"use strict";function r(){function t(){var t=c().length,r=l[1]<l[0],o=l[r-0],u=l[1-r];e=(u-o)/Math.max(1,t-p+2*h),f&&(e=Math.floor(e)),o+=(u-o-e*(t-p))*d,i=e*(1-p),f&&(o=Math.round(o),i=Math.round(i));var v=n.i(a.range)(t).map(function(t){return o+e*t});return s(r?v.reverse():v)}var e,i,o=n.i(u.a)().unknown(void 0),c=o.domain,s=o.range,l=[0,1],f=!1,p=0,h=0,d=.5;return delete o.unknown,o.domain=function(e){return arguments.length?(c(e),t()):c()},o.range=function(e){return arguments.length?(l=[+e[0],+e[1]],t()):l.slice()},o.rangeRound=function(e){return l=[+e[0],+e[1]],f=!0,t()},o.bandwidth=function(){return i},o.step=function(){return e},o.round=function(e){return arguments.length?(f=!!e,t()):f},o.padding=function(e){return arguments.length?(p=h=Math.max(0,Math.min(1,e)),t()):p},o.paddingInner=function(e){return arguments.length?(p=Math.max(0,Math.min(1,e)),t()):p},o.paddingOuter=function(e){return arguments.length?(h=Math.max(0,Math.min(1,e)),t()):h},o.align=function(e){return arguments.length?(d=Math.max(0,Math.min(1,e)),t()):d},o.copy=function(){return r().domain(c()).range(l).round(f).paddingInner(p).paddingOuter(h).align(d)},t()}function i(t){var e=t.copy;return t.padding=t.paddingOuter,delete t.paddingInner,delete t.paddingOuter,t.copy=function(){return i(e())},t}function o(){return i(r().paddingInner(1))}e.a=r,e.b=o;var a=n(7),u=n(127)},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("1f77b4aec7e8ff7f0effbb782ca02c98df8ad62728ff98969467bdc5b0d58c564bc49c94e377c2f7b6d27f7f7fc7c7c7bcbd22dbdb8d17becf9edae5")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("393b795254a36b6ecf9c9ede6379398ca252b5cf6bcedb9c8c6d31bd9e39e7ba52e7cb94843c39ad494ad6616be7969c7b4173a55194ce6dbdde9ed6")},function(t,e,n){"use strict";var r=n(33);e.a=n.i(r.a)("3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9")},function(t,e,n){"use strict";var r=n(10),i=n(30);e.a=n.i(i.d)(n.i(r.cubehelix)(300,.5,0),n.i(r.cubehelix)(-240,.5,1))},function(t,e,n){"use strict";function r(){function t(t){return+t}var e=[0,1];return t.invert=t,t.domain=t.range=function(n){return arguments.length?(e=i.a.call(n,a.a),t):e.slice()},t.copy=function(){return r().domain(e)},n.i(o.b)(t)}e.a=r;var i=n(16),o=n(34),a=n(126)},function(t,e,n){"use strict";function r(t,e){return(e=Math.log(e/t))?function(n){return Math.log(n/t)/e}:n.i(p.a)(e)}function i(t,e){return t<0?function(n){return-Math.pow(-e,n)*Math.pow(-t,1-n)}:function(n){return Math.pow(e,n)*Math.pow(t,1-n)}}function o(t){return isFinite(t)?+("1e"+t):t<0?0:t}function a(t){return 10===t?o:t===Math.E?Math.exp:function(e){return Math.pow(t,e)}}function u(t){return t===Math.E?Math.log:10===t&&Math.log10||2===t&&Math.log2||(t=Math.log(t),function(e){return Math.log(e)/t})}function c(t){return function(e){return-t(-e)}}function s(){function t(){return v=u(p),g=a(p),o()[0]<0&&(v=c(v),g=c(g)),e}var e=n.i(d.a)(r,i).domain([1,10]),o=e.domain,p=10,v=u(10),g=a(10);return e.base=function(e){return arguments.length?(p=+e,t()):p},e.domain=function(e){return arguments.length?(o(e),t()):o()},e.ticks=function(t){var e,r=o(),i=r[0],a=r[r.length-1];(e=a<i)&&(f=i,i=a,a=f);var u,c,s,f=v(i),h=v(a),d=null==t?10:+t,m=[];if(!(p%1)&&h-f<d){if(f=Math.round(f)-1,h=Math.round(h)+1,i>0){for(;f<h;++f)for(c=1,u=g(f);c<p;++c)if(!((s=u*c)<i)){if(s>a)break;m.push(s)}}else for(;f<h;++f)for(c=p-1,u=g(f);c>=1;--c)if(!((s=u*c)<i)){if(s>a)break;m.push(s)}}else m=n.i(l.ticks)(f,h,Math.min(h-f,d)).map(g);return e?m.reverse():m},e.tickFormat=function(t,r){if(null==r&&(r=10===p?".0e":","),"function"!=typeof r&&(r=n.i(f.format)(r)),t===1/0)return r;null==t&&(t=10);var i=Math.max(1,p*t/e.ticks().length);return function(t){var e=t/g(Math.round(v(t)));return e*p<p-.5&&(e*=p),e<=i?r(t):""}},e.nice=function(){return o(n.i(h.a)(o(),{floor:function(t){return g(Math.floor(v(t)))},ceil:function(t){return g(Math.ceil(v(t)))}}))},e.copy=function(){return n.i(d.c)(e,s().base(p))},e}e.a=s;var l=n(7),f=n(29),p=n(67),h=n(125),d=n(44)},function(t,e,n){"use strict";function r(t,e){return t<0?-Math.pow(-t,e):Math.pow(t,e)}function i(){function t(t,e){return(e=r(e,o)-(t=r(t,o)))?function(n){return(r(n,o)-t)/e}:n.i(a.a)(e)}function e(t,e){return e=r(e,o)-(t=r(t,o)),function(n){return r(t+e*n,1/o)}}var o=1,s=n.i(c.a)(t,e),l=s.domain;return s.exponent=function(t){return arguments.length?(o=+t,l(l())):o},s.copy=function(){return n.i(c.c)(s,i().exponent(o))},n.i(u.b)(s)}function o(){return i().exponent(.5)}e.a=i,e.b=o;var a=n(67),u=n(34),c=n(44)},function(t,e,n){"use strict";function r(){function t(){var t=0,r=Math.max(1,u.length);for(c=new Array(r-1);++t<r;)c[t-1]=n.i(i.quantile)(a,t/r);return e}function e(t){if(!isNaN(t=+t))return u[n.i(i.bisect)(c,t)]}var a=[],u=[],c=[];return e.invertExtent=function(t){var e=u.indexOf(t);return e<0?[NaN,NaN]:[e>0?c[e-1]:a[0],e<c.length?c[e]:a[a.length-1]]},e.domain=function(e){if(!arguments.length)return a.slice();a=[];for(var n,r=0,o=e.length;r<o;++r)null==(n=e[r])||isNaN(n=+n)||a.push(n);return a.sort(i.ascending),t()},e.range=function(e){return arguments.length?(u=o.b.call(e),t()):u.slice()},e.quantiles=function(){return c.slice()},e.copy=function(){return r().domain(a).range(u)},e}e.a=r;var i=n(7),o=n(16)},function(t,e,n){"use strict";function r(){function t(t){if(t<=t)return f[n.i(i.bisect)(l,t,0,s)]}function e(){var e=-1;for(l=new Array(s);++e<s;)l[e]=((e+1)*c-(e-s)*u)/(s+1);return t}var u=0,c=1,s=1,l=[.5],f=[0,1];return t.domain=function(t){return arguments.length?(u=+t[0],c=+t[1],e()):[u,c]},t.range=function(t){return arguments.length?(s=(f=o.b.call(t)).length-1,e()):f.slice()},t.invertExtent=function(t){var e=f.indexOf(t);return e<0?[NaN,NaN]:e<1?[u,l[0]]:e>=s?[l[s-1],c]:[l[e-1],l[e]]},t.copy=function(){return r().domain([u,c]).range(f)},n.i(a.b)(t)}e.a=r;var i=n(7),o=n(16),a=n(34)},function(t,e,n){"use strict";n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a});var r=n(10),i=n(30),o=n.i(i.d)(n.i(r.cubehelix)(-100,.75,.35),n.i(r.cubehelix)(80,1.5,.8)),a=n.i(i.d)(n.i(r.cubehelix)(260,.75,.35),n.i(r.cubehelix)(80,1.5,.8)),u=n.i(r.cubehelix)();e.a=function(t){(t<0||t>1)&&(t-=Math.floor(t));var e=Math.abs(t-.5);return u.h=360*t-100,u.s=1.5-1.5*e,u.l=.8-.9*e,u+""}},function(t,e,n){"use strict";function r(t){function e(e){var n=(e-o)/(a-o);return t(u?Math.max(0,Math.min(1,n)):n)}var o=0,a=1,u=!1;return e.domain=function(t){return arguments.length?(o=+t[0],a=+t[1],e):[o,a]},e.clamp=function(t){return arguments.length?(u=!!t,e):u},e.interpolator=function(n){return arguments.length?(t=n,e):t},e.copy=function(){return r(t).domain([o,a]).clamp(u)},n.i(i.b)(e)}e.a=r;var i=n(34)},function(t,e,n){"use strict";function r(){function t(t){if(t<=t)return a[n.i(i.bisect)(e,t,0,u)]}var e=[.5],a=[0,1],u=1;return t.domain=function(n){return arguments.length?(e=o.b.call(n),u=Math.min(e.length,a.length-1),t):e.slice()},t.range=function(n){return arguments.length?(a=o.b.call(n),u=Math.min(e.length,a.length-1),t):a.slice()},t.invertExtent=function(t){var n=a.indexOf(t);return[e[n-1],e[n]]},t.copy=function(){return r().domain(e).range(a)},t}e.a=r;var i=n(7),o=n(16)},function(t,e,n){"use strict";var r=n(7),i=n(29);e.a=function(t,e,o){var a,u=t[0],c=t[t.length-1],s=n.i(r.tickStep)(u,c,null==e?10:e);switch(o=n.i(i.formatSpecifier)(null==o?",f":o),o.type){case"s":var l=Math.max(Math.abs(u),Math.abs(c));return null!=o.precision||isNaN(a=n.i(i.precisionPrefix)(s,l))||(o.precision=a),n.i(i.formatPrefix)(o,l);case"":case"e":case"g":case"p":case"r":null!=o.precision||isNaN(a=n.i(i.precisionRound)(s,Math.max(Math.abs(u),Math.abs(c))))||(o.precision=a-("e"===o.type));break;case"f":case"%":null!=o.precision||isNaN(a=n.i(i.precisionFixed)(s))||(o.precision=a-2*("%"===o.type))}return n.i(i.format)(o)}},function(t,e,n){"use strict";var r=n(128),i=n(78),o=n(80);e.a=function(){return n.i(r.b)(o.h,o.k,o.l,o.b,o.m,o.n,o.o,o.p,i.utcFormat).domain([Date.UTC(2e3,0,1),Date.UTC(2e3,0,2)])}},function(t,e,n){"use strict";function r(t){var e=t.length;return function(n){return t[Math.max(0,Math.min(e-1,Math.floor(n*e)))]}}n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a}),n.d(e,"d",function(){return u});var i=n(33);e.a=r(n.i(i.a)("44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"));var o=r(n.i(i.a)("00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf")),a=r(n.i(i.a)("00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4")),u=r(n.i(i.a)("0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921"))},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(45),i=n(131);e.a=function(t){return n.i(i.a)(n.i(r.a)(t).call(document.documentElement))}},function(t,e,n){"use strict";function r(){return new i}function i(){this._="@"+(++o).toString(36)}e.a=r;var o=0;i.prototype=r.prototype={constructor:i,get:function(t){for(var e=this._;!(e in t);)if(!(t=t.parentNode))return;return t[e]},set:function(t,e){return t[this._]=e},remove:function(t){return this._ in t&&delete t[this._]},toString:function(){return this._}}},function(t,e,n){"use strict";var r=n(72),i=n(46);e.a=function(t){var e=n.i(r.a)();return e.changedTouches&&(e=e.changedTouches[0]),n.i(i.a)(t,e)}},function(t,e,n){"use strict";var r=n(8);e.a=function(t){return"string"==typeof t?new r.b([document.querySelectorAll(t)],[document.documentElement]):new r.b([null==t?[]:t],r.c)}},function(t,e,n){"use strict";var r=n(45);e.a=function(t){var e="function"==typeof t?t:n.i(r.a)(t);return this.select(function(){return this.appendChild(e.apply(this,arguments))})}},function(t,e,n){"use strict";function r(t){return function(){this.removeAttribute(t)}}function i(t){return function(){this.removeAttributeNS(t.space,t.local)}}function o(t,e){return function(){this.setAttribute(t,e)}}function a(t,e){return function(){this.setAttributeNS(t.space,t.local,e)}}function u(t,e){return function(){var n=e.apply(this,arguments);null==n?this.removeAttribute(t):this.setAttribute(t,n)}}function c(t,e){return function(){var n=e.apply(this,arguments);null==n?this.removeAttributeNS(t.space,t.local):this.setAttributeNS(t.space,t.local,n)}}var s=n(68);e.a=function(t,e){var l=n.i(s.a)(t);if(arguments.length<2){var f=this.node();return l.local?f.getAttributeNS(l.space,l.local):f.getAttribute(l)}return this.each((null==e?l.local?i:r:"function"==typeof e?l.local?c:u:l.local?a:o)(l,e))}},function(t,e,n){"use strict";e.a=function(){var t=arguments[0];return arguments[0]=this,t.apply(null,arguments),this}},function(t,e,n){"use strict";function r(t){return t.trim().split(/^|\s+/)}function i(t){return t.classList||new o(t)}function o(t){this._node=t,this._names=r(t.getAttribute("class")||"")}function a(t,e){for(var n=i(t),r=-1,o=e.length;++r<o;)n.add(e[r])}function u(t,e){for(var n=i(t),r=-1,o=e.length;++r<o;)n.remove(e[r])}function c(t){return function(){a(this,t)}}function s(t){return function(){u(this,t)}}function l(t,e){return function(){(e.apply(this,arguments)?a:u)(this,t)}}o.prototype={add:function(t){this._names.indexOf(t)<0&&(this._names.push(t),this._node.setAttribute("class",this._names.join(" ")))},remove:function(t){var e=this._names.indexOf(t);e>=0&&(this._names.splice(e,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(t){return this._names.indexOf(t)>=0}},e.a=function(t,e){var n=r(t+"");if(arguments.length<2){for(var o=i(this.node()),a=-1,u=n.length;++a<u;)if(!o.contains(n[a]))return!1;return!0}return this.each(("function"==typeof e?l:e?c:s)(n,e))}},function(t,e,n){"use strict";function r(){return this.parentNode.insertBefore(this.cloneNode(!1),this.nextSibling)}function i(){return this.parentNode.insertBefore(this.cloneNode(!0),this.nextSibling)}e.a=function(t){return this.select(t?i:r)}},function(t,e,n){"use strict";function r(t,e,n,r,i,o){for(var u,c=0,s=e.length,l=o.length;c<l;++c)(u=e[c])?(u.__data__=o[c],r[c]=u):n[c]=new a.b(t,o[c]);for(;c<s;++c)(u=e[c])&&(i[c]=u)}function i(t,e,n,r,i,o,u){var s,l,f,p={},h=e.length,d=o.length,v=new Array(h);for(s=0;s<h;++s)(l=e[s])&&(v[s]=f=c+u.call(l,l.__data__,s,e),f in p?i[s]=l:p[f]=l);for(s=0;s<d;++s)f=c+u.call(t,o[s],s,o),(l=p[f])?(r[s]=l,l.__data__=o[s],p[f]=null):n[s]=new a.b(t,o[s]);for(s=0;s<h;++s)(l=e[s])&&p[v[s]]===l&&(i[s]=l)}var o=n(8),a=n(132),u=n(256),c="$";e.a=function(t,e){if(!t)return y=new Array(this.size()),d=-1,this.each(function(t){y[++d]=t}),y;var a=e?i:r,c=this._parents,s=this._groups;"function"!=typeof t&&(t=n.i(u.a)(t));for(var l=s.length,f=new Array(l),p=new Array(l),h=new Array(l),d=0;d<l;++d){var v=c[d],g=s[d],m=g.length,y=t.call(v,v&&v.__data__,d,c),_=y.length,b=p[d]=new Array(_),x=f[d]=new Array(_);a(v,g,b,x,h[d]=new Array(m),y,e);for(var w,C,k=0,E=0;k<_;++k)if(w=b[k]){for(k>=E&&(E=k+1);!(C=x[E])&&++E<_;);w._next=C||null}}return f=new o.b(f,c),f._enter=p,f._exit=h,f}},function(t,e,n){"use strict";e.a=function(t){return arguments.length?this.property("__data__",t):this.node().__data__}},function(t,e,n){"use strict";function r(t,e,r){var i=n.i(a.a)(t),o=i.CustomEvent;"function"==typeof o?o=new o(e,r):(o=i.document.createEvent("Event"),r?(o.initEvent(e,r.bubbles,r.cancelable),o.detail=r.detail):o.initEvent(e,!1,!1)),t.dispatchEvent(o)}function i(t,e){return function(){return r(this,t,e)}}function o(t,e){return function(){return r(this,t,e.apply(this,arguments))}}var a=n(73);e.a=function(t,e){return this.each(("function"==typeof e?o:i)(t,e))}},function(t,e,n){"use strict";e.a=function(t){for(var e=this._groups,n=0,r=e.length;n<r;++n)for(var i,o=e[n],a=0,u=o.length;a<u;++a)(i=o[a])&&t.call(i,i.__data__,a,o);return this}},function(t,e,n){"use strict";e.a=function(){return!this.node()}},function(t,e,n){"use strict";var r=n(133),i=n(8);e.a=function(){return new i.b(this._exit||this._groups.map(r.a),this._parents)}},function(t,e,n){"use strict";var r=n(8),i=n(130);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=new Array(o),u=0;u<o;++u)for(var c,s=e[u],l=s.length,f=a[u]=[],p=0;p<l;++p)(c=s[p])&&t.call(c,c.__data__,p,s)&&f.push(c);return new r.b(a,this._parents)}},function(t,e,n){"use strict";function r(){this.innerHTML=""}function i(t){return function(){this.innerHTML=t}}function o(t){return function(){var e=t.apply(this,arguments);this.innerHTML=null==e?"":e}}e.a=function(t){return arguments.length?this.each(null==t?r:("function"==typeof t?o:i)(t)):this.node().innerHTML}},function(t,e,n){"use strict";function r(){return null}var i=n(45),o=n(71);e.a=function(t,e){var a="function"==typeof t?t:n.i(i.a)(t),u=null==e?r:"function"==typeof e?e:n.i(o.a)(e);return this.select(function(){return this.insertBefore(a.apply(this,arguments),u.apply(this,arguments)||null)})}},function(t,e,n){"use strict";function r(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";var r=n(8);e.a=function(t){for(var e=this._groups,n=t._groups,i=e.length,o=n.length,a=Math.min(i,o),u=new Array(i),c=0;c<a;++c)for(var s,l=e[c],f=n[c],p=l.length,h=u[c]=new Array(p),d=0;d<p;++d)(s=l[d]||f[d])&&(h[d]=s);for(;c<i;++c)u[c]=e[c];return new r.b(u,this._parents)}},function(t,e,n){"use strict";e.a=function(){for(var t=this._groups,e=0,n=t.length;e<n;++e)for(var r=t[e],i=0,o=r.length;i<o;++i){var a=r[i];if(a)return a}return null}},function(t,e,n){"use strict";e.a=function(){var t=new Array(this.size()),e=-1;return this.each(function(){t[++e]=this}),t}},function(t,e,n){"use strict";e.a=function(){for(var t=this._groups,e=-1,n=t.length;++e<n;)for(var r,i=t[e],o=i.length-1,a=i[o];--o>=0;)(r=i[o])&&(a&&a!==r.nextSibling&&a.parentNode.insertBefore(r,a),a=r);return this}},function(t,e,n){"use strict";function r(t){return function(){delete this[t]}}function i(t,e){return function(){this[t]=e}}function o(t,e){return function(){var n=e.apply(this,arguments);null==n?delete this[t]:this[t]=n}}e.a=function(t,e){return arguments.length>1?this.each((null==e?r:"function"==typeof e?o:i)(t,e)):this.node()[t]}},function(t,e,n){"use strict";function r(){this.nextSibling&&this.parentNode.appendChild(this)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";function r(){var t=this.parentNode;t&&t.removeChild(this)}e.a=function(){return this.each(r)}},function(t,e,n){"use strict";var r=n(8),i=n(71);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=new Array(o),u=0;u<o;++u)for(var c,s,l=e[u],f=l.length,p=a[u]=new Array(f),h=0;h<f;++h)(c=l[h])&&(s=t.call(c,c.__data__,h,l))&&("__data__"in c&&(s.__data__=c.__data__),p[h]=s);return new r.b(a,this._parents)}},function(t,e,n){"use strict";var r=n(8),i=n(135);e.a=function(t){"function"!=typeof t&&(t=n.i(i.a)(t));for(var e=this._groups,o=e.length,a=[],u=[],c=0;c<o;++c)for(var s,l=e[c],f=l.length,p=0;p<f;++p)(s=l[p])&&(a.push(t.call(s,s.__data__,p,l)),u.push(s));return new r.b(a,u)}},function(t,e,n){"use strict";e.a=function(){var t=0;return this.each(function(){++t}),t}},function(t,e,n){"use strict";function r(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}var i=n(8);e.a=function(t){function e(e,n){return e&&n?t(e.__data__,n.__data__):!e-!n}t||(t=r);for(var n=this._groups,o=n.length,a=new Array(o),u=0;u<o;++u){for(var c,s=n[u],l=s.length,f=a[u]=new Array(l),p=0;p<l;++p)(c=s[p])&&(f[p]=c);f.sort(e)}return new i.b(a,this._parents).order()}},function(t,e,n){"use strict";function r(){this.textContent=""}function i(t){return function(){this.textContent=t}}function o(t){return function(){var e=t.apply(this,arguments);this.textContent=null==e?"":e}}e.a=function(t){return arguments.length?this.each(null==t?r:("function"==typeof t?o:i)(t)):this.node().textContent}},function(t,e,n){"use strict";var r=n(72),i=n(46);e.a=function(t,e,o){arguments.length<3&&(o=e,e=n.i(r.a)().changedTouches);for(var a,u=0,c=e?e.length:0;u<c;++u)if((a=e[u]).identifier===o)return n.i(i.a)(t,a);return null}},function(t,e,n){"use strict";var r=n(72),i=n(46);e.a=function(t,e){null==e&&(e=n.i(r.a)().touches);for(var o=0,a=e?e.length:0,u=new Array(a);o<a;++o)u[o]=n.i(i.a)(t,e[o]);return u}},function(t,e,n){"use strict";function r(t){return t.innerRadius}function i(t){return t.outerRadius}function o(t){return t.startAngle}function a(t){return t.endAngle}function u(t){return t&&t.padAngle}function c(t,e,n,r,i,o,a,u){var c=n-t,s=r-e,l=a-i,f=u-o,p=(l*(e-o)-f*(t-i))/(f*c-l*s);return[t+p*c,e+p*s]}function s(t,e,r,i,o,a,u){var c=t-r,s=e-i,l=(u?a:-a)/n.i(p.d)(c*c+s*s),f=l*s,h=-l*c,d=t+f,v=e+h,g=r+f,m=i+h,y=(d+g)/2,_=(v+m)/2,b=g-d,x=m-v,w=b*b+x*x,C=o-a,k=d*m-g*v,E=(x<0?-1:1)*n.i(p.d)(n.i(p.e)(0,C*C*w-k*k)),M=(k*x-b*E)/w,T=(-k*b-x*E)/w,S=(k*x+b*E)/w,N=(-k*b+x*E)/w,A=M-y,P=T-_,O=S-y,I=N-_;return A*A+P*P>O*O+I*I&&(M=S,T=N),{cx:M,cy:T,x01:-f,y01:-h,x11:M*(o/C-1),y11:T*(o/C-1)}}var l=n(32),f=n(17),p=n(35);e.a=function(){function t(){var t,r,i=+e.apply(this,arguments),o=+h.apply(this,arguments),a=g.apply(this,arguments)-p.f,u=m.apply(this,arguments)-p.f,f=n.i(p.g)(u-a),b=u>a;if(_||(_=t=n.i(l.a)()),o<i&&(r=o,o=i,i=r),o>p.a)if(f>p.c-p.a)_.moveTo(o*n.i(p.h)(a),o*n.i(p.i)(a)),_.arc(0,0,o,a,u,!b),i>p.a&&(_.moveTo(i*n.i(p.h)(u),i*n.i(p.i)(u)),_.arc(0,0,i,u,a,b));else{var x,w,C=a,k=u,E=a,M=u,T=f,S=f,N=y.apply(this,arguments)/2,A=N>p.a&&(v?+v.apply(this,arguments):n.i(p.d)(i*i+o*o)),P=n.i(p.j)(n.i(p.g)(o-i)/2,+d.apply(this,arguments)),O=P,I=P;if(A>p.a){var D=n.i(p.k)(A/i*n.i(p.i)(N)),R=n.i(p.k)(A/o*n.i(p.i)(N));(T-=2*D)>p.a?(D*=b?1:-1,E+=D,M-=D):(T=0,E=M=(a+u)/2),(S-=2*R)>p.a?(R*=b?1:-1,C+=R,k-=R):(S=0,C=k=(a+u)/2)}var L=o*n.i(p.h)(C),U=o*n.i(p.i)(C),F=i*n.i(p.h)(M),j=i*n.i(p.i)(M);if(P>p.a){var B=o*n.i(p.h)(k),V=o*n.i(p.i)(k),W=i*n.i(p.h)(E),z=i*n.i(p.i)(E);if(f<p.b){var H=T>p.a?c(L,U,W,z,B,V,F,j):[F,j],q=L-H[0],Y=U-H[1],K=B-H[0],G=V-H[1],$=1/n.i(p.i)(n.i(p.l)((q*K+Y*G)/(n.i(p.d)(q*q+Y*Y)*n.i(p.d)(K*K+G*G)))/2),X=n.i(p.d)(H[0]*H[0]+H[1]*H[1]);O=n.i(p.j)(P,(i-X)/($-1)),I=n.i(p.j)(P,(o-X)/($+1))}}S>p.a?I>p.a?(x=s(W,z,L,U,o,I,b),w=s(B,V,F,j,o,I,b),_.moveTo(x.cx+x.x01,x.cy+x.y01),I<P?_.arc(x.cx,x.cy,I,n.i(p.m)(x.y01,x.x01),n.i(p.m)(w.y01,w.x01),!b):(_.arc(x.cx,x.cy,I,n.i(p.m)(x.y01,x.x01),n.i(p.m)(x.y11,x.x11),!b),_.arc(0,0,o,n.i(p.m)(x.cy+x.y11,x.cx+x.x11),n.i(p.m)(w.cy+w.y11,w.cx+w.x11),!b),_.arc(w.cx,w.cy,I,n.i(p.m)(w.y11,w.x11),n.i(p.m)(w.y01,w.x01),!b))):(_.moveTo(L,U),_.arc(0,0,o,C,k,!b)):_.moveTo(L,U),i>p.a&&T>p.a?O>p.a?(x=s(F,j,B,V,i,-O,b),w=s(L,U,W,z,i,-O,b),_.lineTo(x.cx+x.x01,x.cy+x.y01),O<P?_.arc(x.cx,x.cy,O,n.i(p.m)(x.y01,x.x01),n.i(p.m)(w.y01,w.x01),!b):(_.arc(x.cx,x.cy,O,n.i(p.m)(x.y01,x.x01),n.i(p.m)(x.y11,x.x11),!b),_.arc(0,0,i,n.i(p.m)(x.cy+x.y11,x.cx+x.x11),n.i(p.m)(w.cy+w.y11,w.cx+w.x11),b),_.arc(w.cx,w.cy,O,n.i(p.m)(w.y11,w.x11),n.i(p.m)(w.y01,w.x01),!b))):_.arc(0,0,i,M,E,b):_.lineTo(F,j)}else _.moveTo(0,0);if(_.closePath(),t)return _=null,t+""||null}var e=r,h=i,d=n.i(f.a)(0),v=null,g=o,m=a,y=u,_=null;return t.centroid=function(){var t=(+e.apply(this,arguments)+ +h.apply(this,arguments))/2,r=(+g.apply(this,arguments)+ +m.apply(this,arguments))/2-p.b/2;return[n.i(p.h)(r)*t,n.i(p.i)(r)*t]},t.innerRadius=function(r){return arguments.length?(e="function"==typeof r?r:n.i(f.a)(+r),t):e},t.outerRadius=function(e){return arguments.length?(h="function"==typeof e?e:n.i(f.a)(+e),t):h},t.cornerRadius=function(e){return arguments.length?(d="function"==typeof e?e:n.i(f.a)(+e),t):d},t.padRadius=function(e){return arguments.length?(v=null==e?null:"function"==typeof e?e:n.i(f.a)(+e),t):v},t.startAngle=function(e){return arguments.length?(g="function"==typeof e?e:n.i(f.a)(+e),t):g},t.endAngle=function(e){return arguments.length?(m="function"==typeof e?e:n.i(f.a)(+e),t):m},t.padAngle=function(e){return arguments.length?(y="function"==typeof e?e:n.i(f.a)(+e),t):y},t.context=function(e){return arguments.length?(_=null==e?null:e,t):_},t}},function(t,e,n){"use strict";var r=n(141),i=n(137),o=n(142);e.a=function(){var t=n.i(i.a)().curve(r.b),e=t.curve,a=t.lineX0,u=t.lineX1,c=t.lineY0,s=t.lineY1;return t.angle=t.x,delete t.x,t.startAngle=t.x0,delete t.x0,t.endAngle=t.x1,delete t.x1,t.radius=t.y,delete t.y,t.innerRadius=t.y0,delete t.y0,t.outerRadius=t.y1,delete t.y1,t.lineStartAngle=function(){return n.i(o.b)(a())},delete t.lineX0,t.lineEndAngle=function(){return n.i(o.b)(u())},delete t.lineX1,t.lineInnerRadius=function(){return n.i(o.b)(c())},delete t.lineY0,t.lineOuterRadius=function(){return n.i(o.b)(s())},delete t.lineY1,t.curve=function(t){return arguments.length?e(n.i(r.a)(t)):e()._curve},t}},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(50),o=n(47);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._y0=this._y1=this._y2=this._y3=this._y4=NaN,this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x2,this._y2),this._context.closePath();break;case 2:this._context.moveTo((this._x2+2*this._x3)/3,(this._y2+2*this._y3)/3),this._context.lineTo((this._x3+2*this._x2)/3,(this._y3+2*this._y2)/3),this._context.closePath();break;case 3:this.point(this._x2,this._y2),this.point(this._x3,this._y3),this.point(this._x4,this._y4)}},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._x2=t,this._y2=e;break;case 1:this._point=2,this._x3=t,this._y3=e;break;case 2:this._point=3,this._x4=t,this._y4=e,this._context.moveTo((this._x0+4*this._x1+t)/6,(this._y0+4*this._y1+e)/6);break;default:n.i(o.c)(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(47);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN,this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;var r=(this._x0+4*this._x1+t)/6,o=(this._y0+4*this._y1+e)/6;this._line?this._context.lineTo(r,o):this._context.moveTo(r,o);break;case 3:this._point=4;default:n.i(i.c)(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t,e){this._basis=new i.b(t),this._beta=e}var i=n(47);r.prototype={lineStart:function(){this._x=[],this._y=[],this._basis.lineStart()},lineEnd:function(){var t=this._x,e=this._y,n=t.length-1;if(n>0)for(var r,i=t[0],o=e[0],a=t[n]-i,u=e[n]-o,c=-1;++c<=n;)r=c/n,this._basis.point(this._beta*t[c]+(1-this._beta)*(i+r*a),this._beta*e[c]+(1-this._beta)*(o+r*u));this._x=this._y=null,this._basis.lineEnd()},point:function(t,e){this._x.push(+t),this._y.push(+e)}},e.a=function t(e){function n(t){return 1===e?new i.b(t):new r(t,e)}return n.beta=function(e){return t(+e)},n}(.85)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._alpha=e}var i=n(139),o=n(50),a=n(74);r.prototype={areaStart:o.a,areaEnd:o.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3),this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3),this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(t,e){if(t=+t,e=+e,this._point){var r=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(r*r+i*i,this._alpha))}switch(this._point){case 0:this._point=1,this._x3=t,this._y3=e;break;case 1:this._point=2,this._context.moveTo(this._x4=t,this._y4=e);break;case 2:this._point=3,this._x5=t,this._y5=e;break;default:n.i(a.b)(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new r(t,e):new i.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._alpha=e}var i=n(140),o=n(74);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){if(t=+t,e=+e,this._point){var r=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(r*r+i*i,this._alpha))}switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3,this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:n.i(o.b)(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new r(t,e):new i.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";function r(t){this._context=t}var i=n(50);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._point=0},lineEnd:function(){this._point&&this._context.closePath()},point:function(t,e){t=+t,e=+e,this._point?this._context.lineTo(t,e):(this._point=1,this._context.moveTo(t,e))}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t){return t<0?-1:1}function i(t,e,n){var i=t._x1-t._x0,o=e-t._x1,a=(t._y1-t._y0)/(i||o<0&&-0),u=(n-t._y1)/(o||i<0&&-0),c=(a*o+u*i)/(i+o);return(r(a)+r(u))*Math.min(Math.abs(a),Math.abs(u),.5*Math.abs(c))||0}function o(t,e){var n=t._x1-t._x0;return n?(3*(t._y1-t._y0)/n-e)/2:e}function a(t,e,n){var r=t._x0,i=t._y0,o=t._x1,a=t._y1,u=(o-r)/3;t._context.bezierCurveTo(r+u,i+u*e,o-u,a-u*n,o,a)}function u(t){this._context=t}function c(t){this._context=new s(t)}function s(t){this._context=t}function l(t){return new u(t)}function f(t){return new c(t)}e.a=l,e.b=f,u.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=this._t0=NaN,this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x1,this._y1);break;case 3:a(this,this._t0,o(this,this._t0))}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){var n=NaN;if(t=+t,e=+e,t!==this._x1||e!==this._y1){switch(this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3,a(this,o(this,n=i(this,t,e)),n);break;default:a(this,this._t0,n=i(this,t,e))}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e,this._t0=n}}},(c.prototype=Object.create(u.prototype)).point=function(t,e){u.prototype.point.call(this,e,t)},s.prototype={moveTo:function(t,e){this._context.moveTo(e,t)},closePath:function(){this._context.closePath()},lineTo:function(t,e){this._context.lineTo(e,t)},bezierCurveTo:function(t,e,n,r,i,o){this._context.bezierCurveTo(e,t,r,n,o,i)}}},function(t,e,n){"use strict";function r(t){this._context=t}function i(t){var e,n,r=t.length-1,i=new Array(r),o=new Array(r),a=new Array(r);for(i[0]=0,o[0]=2,a[0]=t[0]+2*t[1],e=1;e<r-1;++e)i[e]=1,o[e]=4,a[e]=4*t[e]+2*t[e+1];for(i[r-1]=2,o[r-1]=7,a[r-1]=8*t[r-1]+t[r],e=1;e<r;++e)n=i[e]/o[e-1],o[e]-=n,a[e]-=n*a[e-1];for(i[r-1]=a[r-1]/o[r-1],e=r-2;e>=0;--e)i[e]=(a[e]-i[e+1])/o[e];for(o[r-1]=(t[r]+i[r-1])/2,e=0;e<r-1;++e)o[e]=2*t[e+1]-i[e+1];return[i,o]}r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=[],this._y=[]},lineEnd:function(){var t=this._x,e=this._y,n=t.length;if(n)if(this._line?this._context.lineTo(t[0],e[0]):this._context.moveTo(t[0],e[0]),2===n)this._context.lineTo(t[1],e[1]);else for(var r=i(t),o=i(e),a=0,u=1;u<n;++a,++u)this._context.bezierCurveTo(r[0][a],o[0][a],r[1][a],o[1][a],t[u],e[u]);(this._line||0!==this._line&&1===n)&&this._context.closePath(),this._line=1-this._line,this._x=this._y=null},point:function(t,e){this._x.push(+t),this._y.push(+e)}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";function r(t,e){this._context=t,this._t=e}function i(t){return new r(t,0)}function o(t){return new r(t,1)}e.c=i,e.b=o,r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=this._y=NaN,this._point=0},lineEnd:function(){0<this._t&&this._t<1&&2===this._point&&this._context.lineTo(this._x,this._y),(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line>=0&&(this._t=1-this._t,this._line=1-this._line)},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;default:if(this._t<=0)this._context.lineTo(this._x,e),this._context.lineTo(t,e);else{var n=this._x*(1-this._t)+t*this._t;this._context.lineTo(n,this._y),this._context.lineTo(n,e)}}this._x=t,this._y=e}},e.a=function(t){return new r(t,.5)}},function(t,e,n){"use strict";e.a=function(t,e){return e<t?-1:e>t?1:e>=t?0:NaN}},function(t,e,n){"use strict";e.a=function(t){return t}},function(t,e,n){"use strict";function r(t){return t.source}function i(t){return t.target}function o(t){function e(){var e,r=h.a.call(arguments),i=o.apply(this,r),l=a.apply(this,r);if(s||(s=e=n.i(p.a)()),t(s,+u.apply(this,(r[0]=i,r)),+c.apply(this,r),+u.apply(this,(r[0]=l,r)),+c.apply(this,r)),e)return s=null,e+""||null}var o=r,a=i,u=v.a,c=v.b,s=null;return e.source=function(t){return arguments.length?(o=t,e):o},e.target=function(t){return arguments.length?(a=t,e):a},e.x=function(t){return arguments.length?(u="function"==typeof t?t:n.i(d.a)(+t),e):u},e.y=function(t){return arguments.length?(c="function"==typeof t?t:n.i(d.a)(+t),e):c},e.context=function(t){return arguments.length?(s=null==t?null:t,e):s},e}function a(t,e,n,r,i){t.moveTo(e,n),t.bezierCurveTo(e=(e+r)/2,n,e,i,r,i)}function u(t,e,n,r,i){t.moveTo(e,n),t.bezierCurveTo(e,n=(n+i)/2,r,n,r,i)}function c(t,e,r,i,o){var a=n.i(g.a)(e,r),u=n.i(g.a)(e,r=(r+o)/2),c=n.i(g.a)(i,r),s=n.i(g.a)(i,o);t.moveTo(a[0],a[1]),t.bezierCurveTo(u[0],u[1],c[0],c[1],s[0],s[1])}function s(){return o(a)}function l(){return o(u)}function f(){var t=o(c);return t.angle=t.x,delete t.x,t.radius=t.y,delete t.y,t}e.a=s,e.b=l,e.c=f;var p=n(32),h=n(138),d=n(17),v=n(77),g=n(143)},function(t,e,n){"use strict";e.a=function(t,e){if((u=t.length)>1)for(var n,r,i,o,a,u,c=0,s=t[e[0]].length;c<s;++c)for(o=a=0,n=0;n<u;++n)(i=(r=t[e[n]][c])[1]-r[0])>=0?(r[0]=o,r[1]=o+=i):i<0?(r[1]=a,r[0]=a+=i):r[0]=o}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((o=t.length)>0){for(var i,o,a,u=0,c=t[0].length;u<c;++u){for(a=i=0;i<o;++i)a+=t[i][u][1]||0;if(a)for(i=0;i<o;++i)t[i][u][1]/=a}n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((i=t.length)>0){for(var i,o=0,a=t[e[0]],u=a.length;o<u;++o){for(var c=0,s=0;c<i;++c)s+=t[c][o][1]||0;a[o][1]+=a[o][0]=-s/2}n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(36);e.a=function(t,e){if((a=t.length)>0&&(o=(i=t[e[0]]).length)>0){for(var i,o,a,u=0,c=1;c<o;++c){for(var s=0,l=0,f=0;s<a;++s){for(var p=t[e[s]],h=p[c][1]||0,d=p[c-1][1]||0,v=(h-d)/2,g=0;g<s;++g){var m=t[e[g]];v+=(m[c][1]||0)-(m[c-1][1]||0)}l+=h,f+=v*h}i[c-1][1]+=i[c-1][0]=u,l&&(u-=f/l)}i[c-1][1]+=i[c-1][0]=u,n.i(r.a)(t,e)}}},function(t,e,n){"use strict";var r=n(76);e.a=function(t){return n.i(r.a)(t).reverse()}},function(t,e,n){"use strict";var r=n(37),i=n(76);e.a=function(t){var e,o,a=t.length,u=t.map(i.b),c=n.i(r.a)(t).sort(function(t,e){return u[e]-u[t]}),s=0,l=0,f=[],p=[];for(e=0;e<a;++e)o=c[e],s<l?(s+=u[o],f.push(o)):(l+=u[o],p.push(o));return p.reverse().concat(f)}},function(t,e,n){"use strict";var r=n(37);e.a=function(t){return n.i(r.a)(t).reverse()}},function(t,e,n){"use strict";var r=n(17),i=n(301),o=n(302),a=n(35);e.a=function(){function t(t){var n,r,i,o,p,h=t.length,d=0,v=new Array(h),g=new Array(h),m=+s.apply(this,arguments),y=Math.min(a.c,Math.max(-a.c,l.apply(this,arguments)-m)),_=Math.min(Math.abs(y)/h,f.apply(this,arguments)),b=_*(y<0?-1:1);for(n=0;n<h;++n)(p=g[v[n]=n]=+e(t[n],n,t))>0&&(d+=p);for(null!=u?v.sort(function(t,e){return u(g[t],g[e])}):null!=c&&v.sort(function(e,n){return c(t[e],t[n])}),n=0,i=d?(y-h*b)/d:0;n<h;++n,m=o)r=v[n],p=g[r],o=m+(p>0?p*i:0)+b,g[r]={data:t[r],index:n,value:p,startAngle:m,endAngle:o,padAngle:_};return g}var e=o.a,u=i.a,c=null,s=n.i(r.a)(0),l=n.i(r.a)(a.c),f=n.i(r.a)(0);return t.value=function(i){return arguments.length?(e="function"==typeof i?i:n.i(r.a)(+i),t):e},t.sortValues=function(e){return arguments.length?(u=e,c=null,t):u},t.sort=function(e){return arguments.length?(c=e,u=null,t):c},t.startAngle=function(e){return arguments.length?(s="function"==typeof e?e:n.i(r.a)(+e),t):s},t.endAngle=function(e){return arguments.length?(l="function"==typeof e?e:n.i(r.a)(+e),t):l},t.padAngle=function(e){return arguments.length?(f="function"==typeof e?e:n.i(r.a)(+e),t):f},t}},function(t,e,n){"use strict";function r(t,e){return t[e]}var i=n(138),o=n(17),a=n(36),u=n(37);e.a=function(){function t(t){var n,r,i=e.apply(this,arguments),o=t.length,a=i.length,u=new Array(a);for(n=0;n<a;++n){for(var f,p=i[n],h=u[n]=new Array(o),d=0;d<o;++d)h[d]=f=[0,+l(t[d],p,d,t)],f.data=t[d];h.key=p}for(n=0,r=c(u);n<a;++n)u[r[n]].index=n;return s(u,r),u}var e=n.i(o.a)([]),c=u.a,s=a.a,l=r;return t.keys=function(r){return arguments.length?(e="function"==typeof r?r:n.i(o.a)(i.a.call(r)),t):e},t.value=function(e){return arguments.length?(l="function"==typeof e?e:n.i(o.a)(+e),t):l},t.order=function(e){return arguments.length?(c=null==e?u.a:"function"==typeof e?e:n.i(o.a)(i.a.call(e)),t):c},t.offset=function(e){return arguments.length?(s=null==e?a.a:e,t):s},t}},function(t,e,n){"use strict";n.d(e,"b",function(){return p});var r=n(32),i=n(144),o=n(145),a=n(146),u=n(148),c=n(147),s=n(149),l=n(150),f=n(17),p=[i.a,o.a,a.a,c.a,u.a,s.a,l.a];e.a=function(){function t(){var t;if(a||(a=t=n.i(r.a)()),e.apply(this,arguments).draw(a,+o.apply(this,arguments)),t)return a=null,t+""||null}var e=n.i(f.a)(i.a),o=n.i(f.a)(64),a=null;return t.type=function(r){return arguments.length?(e="function"==typeof r?r:n.i(f.a)(r),t):e},t.size=function(e){return arguments.length?(o="function"==typeof e?e:n.i(f.a)(+e),t):o},t.context=function(e){return arguments.length?(a=null==e?null:e,t):a},t}},function(t,e,n){"use strict";function r(t){var e=new Date(t);return isNaN(e)?null:e}var i=n(151),o=n(79),a=+new Date("2000-01-01T00:00:00.000Z")?r:n.i(o.e)(i.b);e.a=a},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setHours(0,0,0,0)},function(t,e){t.setDate(t.getDate()+e)},function(t,e){return(e-t-(e.getTimezoneOffset()-t.getTimezoneOffset())*i.d)/i.b},function(t){return t.getDate()-1});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){var e=t.getTimezoneOffset()*i.d%i.c;e<0&&(e+=i.c),t.setTime(Math.floor((+t-e)/i.c)*i.c+e)},function(t,e){t.setTime(+t+e*i.c)},function(t,e){return(e-t)/i.c},function(t){return t.getHours()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(){},function(t,e){t.setTime(+t+e)},function(t,e){return e-t});i.every=function(t){return t=Math.floor(t),isFinite(t)&&t>0?t>1?n.i(r.a)(function(e){e.setTime(Math.floor(e/t)*t)},function(e,n){e.setTime(+e+n*t)},function(e,n){return(n-e)/t}):i:null},e.a=i;i.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setTime(Math.floor(t/i.d)*i.d)},function(t,e){t.setTime(+t+e*i.d)},function(t,e){return(e-t)/i.d},function(t){return t.getMinutes()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setDate(1),t.setHours(0,0,0,0)},function(t,e){t.setMonth(t.getMonth()+e)},function(t,e){return e.getMonth()-t.getMonth()+12*(e.getFullYear()-t.getFullYear())},function(t){return t.getMonth()});e.a=i;i.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setTime(Math.floor(t/i.e)*i.e)},function(t,e){t.setTime(+t+e*i.e)},function(t,e){return(e-t)/i.e},function(t){return t.getUTCSeconds()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCDate(t.getUTCDate()+e)},function(t,e){return(e-t)/i.b},function(t){return t.getUTCDate()-1});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCMinutes(0,0,0)},function(t,e){t.setTime(+t+e*i.c)},function(t,e){return(e-t)/i.c},function(t){return t.getUTCHours()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n(13),o=n.i(r.a)(function(t){t.setUTCSeconds(0,0)},function(t,e){t.setTime(+t+e*i.d)},function(t,e){return(e-t)/i.d},function(t){return t.getUTCMinutes()});e.a=o;o.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setUTCDate(1),t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCMonth(t.getUTCMonth()+e)},function(t,e){return e.getUTCMonth()-t.getUTCMonth()+12*(e.getUTCFullYear()-t.getUTCFullYear())},function(t){return t.getUTCMonth()});e.a=i;i.range},function(t,e,n){"use strict";function r(t){return n.i(i.a)(function(e){e.setUTCDate(e.getUTCDate()-(e.getUTCDay()+7-t)%7),e.setUTCHours(0,0,0,0)},function(t,e){t.setUTCDate(t.getUTCDate()+7*e)},function(t,e){return(e-t)/o.a})}n.d(e,"a",function(){return a}),n.d(e,"b",function(){return u}),n.d(e,"c",function(){return l});var i=n(5),o=n(13),a=r(0),u=r(1),c=r(2),s=r(3),l=r(4),f=r(5),p=r(6);a.range,u.range,c.range,s.range,l.range,f.range,p.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setUTCMonth(0,1),t.setUTCHours(0,0,0,0)},function(t,e){t.setUTCFullYear(t.getUTCFullYear()+e)},function(t,e){return e.getUTCFullYear()-t.getUTCFullYear()},function(t){return t.getUTCFullYear()});i.every=function(t){return isFinite(t=Math.floor(t))&&t>0?n.i(r.a)(function(e){e.setUTCFullYear(Math.floor(e.getUTCFullYear()/t)*t),e.setUTCMonth(0,1),e.setUTCHours(0,0,0,0)},function(e,n){e.setUTCFullYear(e.getUTCFullYear()+n*t)}):null},e.a=i;i.range},function(t,e,n){"use strict";function r(t){return n.i(i.a)(function(e){e.setDate(e.getDate()-(e.getDay()+7-t)%7),e.setHours(0,0,0,0)},function(t,e){t.setDate(t.getDate()+7*e)},function(t,e){return(e-t-(e.getTimezoneOffset()-t.getTimezoneOffset())*o.d)/o.a})}n.d(e,"a",function(){return a}),n.d(e,"b",function(){return u}),n.d(e,"c",function(){return l});var i=n(5),o=n(13),a=r(0),u=r(1),c=r(2),s=r(3),l=r(4),f=r(5),p=r(6);a.range,u.range,c.range,s.range,l.range,f.range,p.range},function(t,e,n){"use strict";var r=n(5),i=n.i(r.a)(function(t){t.setMonth(0,1),t.setHours(0,0,0,0)},function(t,e){t.setFullYear(t.getFullYear()+e)},function(t,e){return e.getFullYear()-t.getFullYear()},function(t){return t.getFullYear()});i.every=function(t){return isFinite(t=Math.floor(t))&&t>0?n.i(r.a)(function(e){e.setFullYear(Math.floor(e.getFullYear()/t)*t),e.setMonth(0,1),e.setHours(0,0,0,0)},function(e,n){e.setFullYear(e.getFullYear()+n*t)}):null},e.a=i;i.range},function(t,e,n){"use strict";function r(t){return t.replace(i,function(t,e){return e.toUpperCase()})}var i=/-(.)/g;t.exports=r},function(t,e,n){"use strict";function r(t){return i(t.replace(o,"ms-"))}var i=n(329),o=/^-ms-/;t.exports=r},function(t,e,n){"use strict";function r(t,e){return!(!t||!e)&&(t===e||!i(t)&&(i(e)?r(t,e.parentNode):"contains"in t?t.contains(e):!!t.compareDocumentPosition&&!!(16&t.compareDocumentPosition(e))))}var i=n(339);t.exports=r},function(t,e,n){"use strict";function r(t){var e=t.length;if((Array.isArray(t)||"object"!=typeof t&&"function"!=typeof t)&&a(!1),"number"!=typeof e&&a(!1),0===e||e-1 in t||a(!1),"function"==typeof t.callee&&a(!1),t.hasOwnProperty)try{return Array.prototype.slice.call(t)}catch(t){}for(var n=Array(e),r=0;r<e;r++)n[r]=t[r];return n}function i(t){return!!t&&("object"==typeof t||"function"==typeof t)&&"length"in t&&!("setInterval"in t)&&"number"!=typeof t.nodeType&&(Array.isArray(t)||"callee"in t||"item"in t)}function o(t){return i(t)?Array.isArray(t)?t.slice():r(t):[t]}var a=n(0);t.exports=o},function(t,e,n){"use strict";function r(t){var e=t.match(l);return e&&e[1].toLowerCase()}function i(t,e){var n=s;s||c(!1);var i=r(t),o=i&&u(i);if(o){n.innerHTML=o[1]+t+o[2];for(var l=o[0];l--;)n=n.lastChild}else n.innerHTML=t;var f=n.getElementsByTagName("script");f.length&&(e||c(!1),a(f).forEach(e));for(var p=Array.from(n.childNodes);n.lastChild;)n.removeChild(n.lastChild);return p}var o=n(6),a=n(332),u=n(334),c=n(0),s=o.canUseDOM?document.createElement("div"):null,l=/^\s*<(\w+)/;t.exports=i},function(t,e,n){"use strict";function r(t){return a||o(!1),p.hasOwnProperty(t)||(t="*"),u.hasOwnProperty(t)||(a.innerHTML="*"===t?"<link />":"<"+t+"></"+t+">",u[t]=!a.firstChild),u[t]?p[t]:null}var i=n(6),o=n(0),a=i.canUseDOM?document.createElement("div"):null,u={},c=[1,'<select multiple="true">',"</select>"],s=[1,"<table>","</table>"],l=[3,"<table><tbody><tr>","</tr></tbody></table>"],f=[1,'<svg xmlns="http://www.w3.org/2000/svg">',"</svg>"],p={"*":[1,"?<div>","</div>"],area:[1,"<map>","</map>"],col:[2,"<table><tbody></tbody><colgroup>","</colgroup></table>"],legend:[1,"<fieldset>","</fieldset>"],param:[1,"<object>","</object>"],tr:[2,"<table><tbody>","</tbody></table>"],optgroup:c,option:c,caption:s,colgroup:s,tbody:s,tfoot:s,thead:s,td:l,th:l};["circle","clipPath","defs","ellipse","g","image","line","linearGradient","mask","path","pattern","polygon","polyline","radialGradient","rect","stop","text","tspan"].forEach(function(t){p[t]=f,u[t]=!0}),t.exports=r},function(t,e,n){"use strict";function r(t){return t.Window&&t instanceof t.Window?{x:t.pageXOffset||t.document.documentElement.scrollLeft,y:t.pageYOffset||t.document.documentElement.scrollTop}:{x:t.scrollLeft,y:t.scrollTop}}t.exports=r},function(t,e,n){"use strict";function r(t){return t.replace(i,"-$1").toLowerCase()}var i=/([A-Z])/g;t.exports=r},function(t,e,n){"use strict";function r(t){return i(t).replace(o,"-ms-")}var i=n(336),o=/^ms-/;t.exports=r},function(t,e,n){"use strict";function r(t){var e=t?t.ownerDocument||t:document,n=e.defaultView||window;return!(!t||!("function"==typeof n.Node?t instanceof n.Node:"object"==typeof t&&"number"==typeof t.nodeType&&"string"==typeof t.nodeName))}t.exports=r},function(t,e,n){"use strict";function r(t){return i(t)&&3==t.nodeType}var i=n(338);t.exports=r},function(t,e,n){"use strict";var r=function(t){var e;for(e in t)if(t.hasOwnProperty(e))return e;return null};t.exports=r},function(t,e,n){"use strict";function r(t){var e={};return function(n){return e.hasOwnProperty(n)||(e[n]=t.call(this,n)),e[n]}}t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r,i){}t.exports=r},function(t,e,n){"use strict";function r(){return null}var i=n(3),o=n(344),a=n(342),u=function(){};t.exports=function(t,e){function n(t){var e=t&&(E&&t[E]||t[M]);if("function"==typeof e)return e}function c(t,e){return t===e?0!==t||1/t==1/e:t!==t&&e!==e}function s(t){this.message=t,this.stack=""}function l(t){function n(n,r,i,a,u,c,l){if(a=a||T,c=c||i,l!==o){if(e){var f=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use `PropTypes.checkPropTypes()` to call them. Read more at http://fb.me/use-check-prop-types");throw f.name="Invariant Violation",f}}return null==r[i]?n?new s(null===r[i]?"The "+u+" `"+c+"` is marked as required in `"+a+"`, but its value is `null`.":"The "+u+" `"+c+"` is marked as required in `"+a+"`, but its value is `undefined`."):null:t(r,i,a,u,c)}var r=n.bind(null,!1);return r.isRequired=n.bind(null,!0),r}function f(t){function e(e,n,r,i,o,a){var u=e[n];if(x(u)!==t)return new s("Invalid "+i+" `"+o+"` of type `"+w(u)+"` supplied to `"+r+"`, expected `"+t+"`.");return null}return l(e)}function p(t){function e(e,n,r,i,a){if("function"!=typeof t)return new s("Property `"+a+"` of component `"+r+"` has invalid PropType notation inside arrayOf.");var u=e[n];if(!Array.isArray(u)){return new s("Invalid "+i+" `"+a+"` of type `"+x(u)+"` supplied to `"+r+"`, expected an array.")}for(var c=0;c<u.length;c++){var l=t(u,c,r,i,a+"["+c+"]",o);if(l instanceof Error)return l}return null}return l(e)}function h(t){function e(e,n,r,i,o){if(!(e[n]instanceof t)){var a=t.name||T;return new s("Invalid "+i+" `"+o+"` of type `"+k(e[n])+"` supplied to `"+r+"`, expected instance of `"+a+"`.")}return null}return l(e)}function d(t){function e(e,n,r,i,o){for(var a=e[n],u=0;u<t.length;u++)if(c(a,t[u]))return null;return new s("Invalid "+i+" `"+o+"` of value `"+a+"` supplied to `"+r+"`, expected one of "+JSON.stringify(t)+".")}return Array.isArray(t)?l(e):r}function v(t){function e(e,n,r,i,a){if("function"!=typeof t)return new s("Property `"+a+"` of component `"+r+"` has invalid PropType notation inside objectOf.");var u=e[n],c=x(u);if("object"!==c)return new s("Invalid "+i+" `"+a+"` of type `"+c+"` supplied to `"+r+"`, expected an object.");for(var l in u)if(u.hasOwnProperty(l)){var f=t(u,l,r,i,a+"."+l,o);if(f instanceof Error)return f}return null}return l(e)}function g(t){function e(e,n,r,i,a){for(var u=0;u<t.length;u++){if(null==(0,t[u])(e,n,r,i,a,o))return null}return new s("Invalid "+i+" `"+a+"` supplied to `"+r+"`.")}if(!Array.isArray(t))return r;for(var n=0;n<t.length;n++){var i=t[n];if("function"!=typeof i)return u("Invalid argument supplied to oneOfType. Expected an array of check functions, but received "+C(i)+" at index "+n+"."),r}return l(e)}function m(t){function e(e,n,r,i,a){var u=e[n],c=x(u);if("object"!==c)return new s("Invalid "+i+" `"+a+"` of type `"+c+"` supplied to `"+r+"`, expected `object`.");for(var l in t){var f=t[l];if(f){var p=f(u,l,r,i,a+"."+l,o);if(p)return p}}return null}return l(e)}function y(t){function e(e,n,r,a,u){var c=e[n],l=x(c);if("object"!==l)return new s("Invalid "+a+" `"+u+"` of type `"+l+"` supplied to `"+r+"`, expected `object`.");var f=i({},e[n],t);for(var p in f){var h=t[p];if(!h)return new s("Invalid "+a+" `"+u+"` key `"+p+"` supplied to `"+r+"`.\nBad object: "+JSON.stringify(e[n],null,"  ")+"\nValid keys: "+JSON.stringify(Object.keys(t),null,"  "));var d=h(c,p,r,a,u+"."+p,o);if(d)return d}return null}return l(e)}function _(e){switch(typeof e){case"number":case"string":case"undefined":return!0;case"boolean":return!e;case"object":if(Array.isArray(e))return e.every(_);if(null===e||t(e))return!0;var r=n(e);if(!r)return!1;var i,o=r.call(e);if(r!==e.entries){for(;!(i=o.next()).done;)if(!_(i.value))return!1}else for(;!(i=o.next()).done;){var a=i.value;if(a&&!_(a[1]))return!1}return!0;default:return!1}}function b(t,e){return"symbol"===t||("Symbol"===e["@@toStringTag"]||"function"==typeof Symbol&&e instanceof Symbol)}function x(t){var e=typeof t;return Array.isArray(t)?"array":t instanceof RegExp?"object":b(e,t)?"symbol":e}function w(t){if(void 0===t||null===t)return""+t;var e=x(t);if("object"===e){if(t instanceof Date)return"date";if(t instanceof RegExp)return"regexp"}return e}function C(t){var e=w(t);switch(e){case"array":case"object":return"an "+e;case"boolean":case"date":case"regexp":return"a "+e;default:return e}}function k(t){return t.constructor&&t.constructor.name?t.constructor.name:T}var E="function"==typeof Symbol&&Symbol.iterator,M="@@iterator",T="<<anonymous>>",S={array:f("array"),bool:f("boolean"),func:f("function"),number:f("number"),object:f("object"),string:f("string"),symbol:f("symbol"),any:function(){return l(r)}(),arrayOf:p,element:function(){function e(e,n,r,i,o){var a=e[n];if(!t(a)){return new s("Invalid "+i+" `"+o+"` of type `"+x(a)+"` supplied to `"+r+"`, expected a single ReactElement.")}return null}return l(e)}(),instanceOf:h,node:function(){function t(t,e,n,r,i){return _(t[e])?null:new s("Invalid "+r+" `"+i+"` supplied to `"+n+"`, expected a ReactNode.")}return l(t)}(),objectOf:v,oneOf:d,oneOfType:g,shape:m,exact:y};return s.prototype=Error.prototype,S.checkPropTypes=a,S.PropTypes=S,S}},function(t,e,n){"use strict";t.exports="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED"},function(t,e,n){"use strict";var r={Properties:{"aria-current":0,"aria-details":0,"aria-disabled":0,"aria-hidden":0,"aria-invalid":0,"aria-keyshortcuts":0,"aria-label":0,"aria-roledescription":0,"aria-autocomplete":0,"aria-checked":0,"aria-expanded":0,"aria-haspopup":0,"aria-level":0,"aria-modal":0,"aria-multiline":0,"aria-multiselectable":0,"aria-orientation":0,"aria-placeholder":0,"aria-pressed":0,"aria-readonly":0,"aria-required":0,"aria-selected":0,"aria-sort":0,"aria-valuemax":0,"aria-valuemin":0,"aria-valuenow":0,"aria-valuetext":0,"aria-atomic":0,"aria-busy":0,"aria-live":0,"aria-relevant":0,"aria-dropeffect":0,"aria-grabbed":0,"aria-activedescendant":0,"aria-colcount":0,"aria-colindex":0,"aria-colspan":0,"aria-controls":0,"aria-describedby":0,"aria-errormessage":0,"aria-flowto":0,"aria-labelledby":0,"aria-owns":0,"aria-posinset":0,"aria-rowcount":0,"aria-rowindex":0,"aria-rowspan":0,"aria-setsize":0},DOMAttributeNames:{},DOMPropertyNames:{}};t.exports=r},function(t,e,n){"use strict";var r=n(4),i=n(154),o={focusDOMComponent:function(){i(r.getNodeFromInstance(this))}};t.exports=o},function(t,e,n){"use strict";function r(t){return(t.ctrlKey||t.altKey||t.metaKey)&&!(t.ctrlKey&&t.altKey)}function i(t){switch(t){case"topCompositionStart":return E.compositionStart;case"topCompositionEnd":return E.compositionEnd;case"topCompositionUpdate":return E.compositionUpdate}}function o(t,e){return"topKeyDown"===t&&e.keyCode===y}function a(t,e){switch(t){case"topKeyUp":return-1!==m.indexOf(e.keyCode);case"topKeyDown":return e.keyCode!==y;case"topKeyPress":case"topMouseDown":case"topBlur":return!0;default:return!1}}function u(t){var e=t.detail;return"object"==typeof e&&"data"in e?e.data:null}function c(t,e,n,r){var c,s;if(_?c=i(t):T?a(t,n)&&(c=E.compositionEnd):o(t,n)&&(c=E.compositionStart),!c)return null;w&&(T||c!==E.compositionStart?c===E.compositionEnd&&T&&(s=T.getData()):T=d.getPooled(r));var l=v.getPooled(c,e,n,r);if(s)l.data=s;else{var f=u(n);null!==f&&(l.data=f)}return p.accumulateTwoPhaseDispatches(l),l}function s(t,e){switch(t){case"topCompositionEnd":return u(e);case"topKeyPress":return e.which!==C?null:(M=!0,k);case"topTextInput":var n=e.data;return n===k&&M?null:n;default:return null}}function l(t,e){if(T){if("topCompositionEnd"===t||!_&&a(t,e)){var n=T.getData();return d.release(T),T=null,n}return null}switch(t){case"topPaste":return null;case"topKeyPress":return e.which&&!r(e)?String.fromCharCode(e.which):null;case"topCompositionEnd":return w?null:e.data;default:return null}}function f(t,e,n,r){var i;if(!(i=x?s(t,n):l(t,n)))return null;var o=g.getPooled(E.beforeInput,e,n,r);return o.data=i,p.accumulateTwoPhaseDispatches(o),o}var p=n(23),h=n(6),d=n(354),v=n(391),g=n(394),m=[9,13,27,32],y=229,_=h.canUseDOM&&"CompositionEvent"in window,b=null;h.canUseDOM&&"documentMode"in document&&(b=document.documentMode);var x=h.canUseDOM&&"TextEvent"in window&&!b&&!function(){var t=window.opera;return"object"==typeof t&&"function"==typeof t.version&&parseInt(t.version(),10)<=12}(),w=h.canUseDOM&&(!_||b&&b>8&&b<=11),C=32,k=String.fromCharCode(C),E={beforeInput:{phasedRegistrationNames:{bubbled:"onBeforeInput",captured:"onBeforeInputCapture"},dependencies:["topCompositionEnd","topKeyPress","topTextInput","topPaste"]},compositionEnd:{phasedRegistrationNames:{bubbled:"onCompositionEnd",captured:"onCompositionEndCapture"},dependencies:["topBlur","topCompositionEnd","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]},compositionStart:{phasedRegistrationNames:{bubbled:"onCompositionStart",captured:"onCompositionStartCapture"},dependencies:["topBlur","topCompositionStart","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]},compositionUpdate:{phasedRegistrationNames:{bubbled:"onCompositionUpdate",captured:"onCompositionUpdateCapture"},dependencies:["topBlur","topCompositionUpdate","topKeyDown","topKeyPress","topKeyUp","topMouseDown"]}},M=!1,T=null,S={eventTypes:E,extractEvents:function(t,e,n,r){return[c(t,e,n,r),f(t,e,n,r)]}};t.exports=S},function(t,e,n){"use strict";var r=n(158),i=n(6),o=(n(9),n(330),n(400)),a=n(337),u=n(341),c=(n(2),u(function(t){return a(t)})),s=!1,l="cssFloat";if(i.canUseDOM){var f=document.createElement("div").style;try{f.font=""}catch(t){s=!0}void 0===document.documentElement.style.cssFloat&&(l="styleFloat")}var p={createMarkupForStyles:function(t,e){var n="";for(var r in t)if(t.hasOwnProperty(r)){var i=0===r.indexOf("--"),a=t[r];null!=a&&(n+=c(r)+":",n+=o(r,a,e,i)+";")}return n||null},setValueForStyles:function(t,e,n){var i=t.style;for(var a in e)if(e.hasOwnProperty(a)){var u=0===a.indexOf("--"),c=o(a,e[a],n,u);if("float"!==a&&"cssFloat"!==a||(a=l),u)i.setProperty(a,c);else if(c)i[a]=c;else{var f=s&&r.shorthandPropertyExpansions[a];if(f)for(var p in f)i[p]="";else i[a]=""}}}};t.exports=p},function(t,e,n){"use strict";function r(t,e,n){var r=M.getPooled(P.change,t,e,n);return r.type="change",w.accumulateTwoPhaseDispatches(r),r}function i(t){var e=t.nodeName&&t.nodeName.toLowerCase();return"select"===e||"input"===e&&"file"===t.type}function o(t){var e=r(I,t,S(t));E.batchedUpdates(a,e)}function a(t){x.enqueueEvents(t),x.processEventQueue(!1)}function u(t,e){O=t,I=e,O.attachEvent("onchange",o)}function c(){O&&(O.detachEvent("onchange",o),O=null,I=null)}function s(t,e){var n=T.updateValueIfChanged(t),r=!0===e.simulated&&L._allowSimulatedPassThrough;if(n||r)return t}function l(t,e){if("topChange"===t)return e}function f(t,e,n){"topFocus"===t?(c(),u(e,n)):"topBlur"===t&&c()}function p(t,e){O=t,I=e,O.attachEvent("onpropertychange",d)}function h(){O&&(O.detachEvent("onpropertychange",d),O=null,I=null)}function d(t){"value"===t.propertyName&&s(I,t)&&o(t)}function v(t,e,n){"topFocus"===t?(h(),p(e,n)):"topBlur"===t&&h()}function g(t,e,n){if("topSelectionChange"===t||"topKeyUp"===t||"topKeyDown"===t)return s(I,n)}function m(t){var e=t.nodeName;return e&&"input"===e.toLowerCase()&&("checkbox"===t.type||"radio"===t.type)}function y(t,e,n){if("topClick"===t)return s(e,n)}function _(t,e,n){if("topInput"===t||"topChange"===t)return s(e,n)}function b(t,e){if(null!=t){var n=t._wrapperState||e._wrapperState;if(n&&n.controlled&&"number"===e.type){var r=""+e.value;e.getAttribute("value")!==r&&e.setAttribute("value",r)}}}var x=n(22),w=n(23),C=n(6),k=n(4),E=n(12),M=n(14),T=n(173),S=n(94),N=n(95),A=n(175),P={change:{phasedRegistrationNames:{bubbled:"onChange",captured:"onChangeCapture"},dependencies:["topBlur","topChange","topClick","topFocus","topInput","topKeyDown","topKeyUp","topSelectionChange"]}},O=null,I=null,D=!1;C.canUseDOM&&(D=N("change")&&(!document.documentMode||document.documentMode>8));var R=!1;C.canUseDOM&&(R=N("input")&&(!document.documentMode||document.documentMode>9));var L={eventTypes:P,_allowSimulatedPassThrough:!0,_isInputEventSupported:R,extractEvents:function(t,e,n,o){var a,u,c=e?k.getNodeFromInstance(e):window;if(i(c)?D?a=l:u=f:A(c)?R?a=_:(a=g,u=v):m(c)&&(a=y),a){var s=a(t,e,n);if(s){return r(s,n,o)}}u&&u(t,c,e),"topBlur"===t&&b(e,c)}};t.exports=L},function(t,e,n){"use strict";var r=n(1),i=n(20),o=n(6),a=n(333),u=n(11),c=(n(0),{dangerouslyReplaceNodeWithMarkup:function(t,e){if(o.canUseDOM||r("56"),e||r("57"),"HTML"===t.nodeName&&r("58"),"string"==typeof e){var n=a(e,u)[0];t.parentNode.replaceChild(n,t)}else i.replaceChildWithTree(t,e)}});t.exports=c},function(t,e,n){"use strict";var r=["ResponderEventPlugin","SimpleEventPlugin","TapEventPlugin","EnterLeaveEventPlugin","ChangeEventPlugin","SelectEventPlugin","BeforeInputEventPlugin"];t.exports=r},function(t,e,n){"use strict";var r=n(23),i=n(4),o=n(54),a={mouseEnter:{registrationName:"onMouseEnter",dependencies:["topMouseOut","topMouseOver"]},mouseLeave:{registrationName:"onMouseLeave",dependencies:["topMouseOut","topMouseOver"]}},u={eventTypes:a,extractEvents:function(t,e,n,u){if("topMouseOver"===t&&(n.relatedTarget||n.fromElement))return null;if("topMouseOut"!==t&&"topMouseOver"!==t)return null;var c;if(u.window===u)c=u;else{var s=u.ownerDocument;c=s?s.defaultView||s.parentWindow:window}var l,f;if("topMouseOut"===t){l=e;var p=n.relatedTarget||n.toElement;f=p?i.getClosestInstanceFromNode(p):null}else l=null,f=e;if(l===f)return null;var h=null==l?c:i.getNodeFromInstance(l),d=null==f?c:i.getNodeFromInstance(f),v=o.getPooled(a.mouseLeave,l,n,u);v.type="mouseleave",v.target=h,v.relatedTarget=d;var g=o.getPooled(a.mouseEnter,f,n,u);return g.type="mouseenter",g.target=d,g.relatedTarget=h,r.accumulateEnterLeaveDispatches(v,g,l,f),[v,g]}};t.exports=u},function(t,e,n){"use strict";var r={topAbort:null,topAnimationEnd:null,topAnimationIteration:null,topAnimationStart:null,topBlur:null,topCanPlay:null,topCanPlayThrough:null,topChange:null,topClick:null,topCompositionEnd:null,topCompositionStart:null,topCompositionUpdate:null,topContextMenu:null,topCopy:null,topCut:null,topDoubleClick:null,topDrag:null,topDragEnd:null,topDragEnter:null,topDragExit:null,topDragLeave:null,topDragOver:null,topDragStart:null,topDrop:null,topDurationChange:null,topEmptied:null,topEncrypted:null,topEnded:null,topError:null,topFocus:null,topInput:null,topInvalid:null,topKeyDown:null,topKeyPress:null,topKeyUp:null,topLoad:null,topLoadedData:null,topLoadedMetadata:null,topLoadStart:null,topMouseDown:null,topMouseMove:null,topMouseOut:null,topMouseOver:null,topMouseUp:null,topPaste:null,topPause:null,topPlay:null,topPlaying:null,topProgress:null,topRateChange:null,topReset:null,topScroll:null,topSeeked:null,topSeeking:null,topSelectionChange:null,topStalled:null,topSubmit:null,topSuspend:null,topTextInput:null,topTimeUpdate:null,topTouchCancel:null,topTouchEnd:null,topTouchMove:null,topTouchStart:null,topTransitionEnd:null,topVolumeChange:null,topWaiting:null,topWheel:null},i={topLevelTypes:r};t.exports=i},function(t,e,n){"use strict";function r(t){this._root=t,this._startText=this.getText(),this._fallbackText=null}var i=n(3),o=n(18),a=n(172);i(r.prototype,{destructor:function(){this._root=null,this._startText=null,this._fallbackText=null},getText:function(){return"value"in this._root?this._root.value:this._root[a()]},getData:function(){if(this._fallbackText)return this._fallbackText;var t,e,n=this._startText,r=n.length,i=this.getText(),o=i.length;for(t=0;t<r&&n[t]===i[t];t++);var a=r-t;for(e=1;e<=a&&n[r-e]===i[o-e];e++);var u=e>1?1-e:void 0;return this._fallbackText=i.slice(t,u),this._fallbackText}}),o.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";var r=n(21),i=r.injection.MUST_USE_PROPERTY,o=r.injection.HAS_BOOLEAN_VALUE,a=r.injection.HAS_NUMERIC_VALUE,u=r.injection.HAS_POSITIVE_NUMERIC_VALUE,c=r.injection.HAS_OVERLOADED_BOOLEAN_VALUE,s={isCustomAttribute:RegExp.prototype.test.bind(new RegExp("^(data|aria)-["+r.ATTRIBUTE_NAME_CHAR+"]*$")),Properties:{accept:0,acceptCharset:0,accessKey:0,action:0,allowFullScreen:o,allowTransparency:0,alt:0,as:0,async:o,autoComplete:0,autoPlay:o,capture:o,cellPadding:0,cellSpacing:0,charSet:0,challenge:0,checked:i|o,cite:0,classID:0,className:0,cols:u,colSpan:0,content:0,contentEditable:0,contextMenu:0,controls:o,controlsList:0,coords:0,crossOrigin:0,data:0,dateTime:0,default:o,defer:o,dir:0,disabled:o,download:c,draggable:0,encType:0,form:0,formAction:0,formEncType:0,formMethod:0,formNoValidate:o,formTarget:0,frameBorder:0,headers:0,height:0,hidden:o,high:0,href:0,hrefLang:0,htmlFor:0,httpEquiv:0,icon:0,id:0,inputMode:0,integrity:0,is:0,keyParams:0,keyType:0,kind:0,label:0,lang:0,list:0,loop:o,low:0,manifest:0,marginHeight:0,marginWidth:0,max:0,maxLength:0,media:0,mediaGroup:0,method:0,min:0,minLength:0,multiple:i|o,muted:i|o,name:0,nonce:0,noValidate:o,open:o,optimum:0,pattern:0,placeholder:0,playsInline:o,poster:0,preload:0,profile:0,radioGroup:0,readOnly:o,referrerPolicy:0,rel:0,required:o,reversed:o,role:0,rows:u,rowSpan:a,sandbox:0,scope:0,scoped:o,scrolling:0,seamless:o,selected:i|o,shape:0,size:u,sizes:0,span:u,spellCheck:0,src:0,srcDoc:0,srcLang:0,srcSet:0,start:a,step:0,style:0,summary:0,tabIndex:0,target:0,title:0,type:0,useMap:0,value:0,width:0,wmode:0,wrap:0,about:0,datatype:0,inlist:0,prefix:0,property:0,resource:0,typeof:0,vocab:0,autoCapitalize:0,autoCorrect:0,autoSave:0,color:0,itemProp:0,itemScope:o,itemType:0,itemID:0,itemRef:0,results:0,security:0,unselectable:0},DOMAttributeNames:{acceptCharset:"accept-charset",className:"class",htmlFor:"for",httpEquiv:"http-equiv"},DOMPropertyNames:{},DOMMutationMethods:{value:function(t,e){if(null==e)return t.removeAttribute("value");"number"!==t.type||!1===t.hasAttribute("value")?t.setAttribute("value",""+e):t.validity&&!t.validity.badInput&&t.ownerDocument.activeElement!==t&&t.setAttribute("value",""+e)}}};t.exports=s},function(t,e,n){"use strict";(function(e){function r(t,e,n,r){var i=void 0===t[n];null!=e&&i&&(t[n]=o(e,!0))}var i=n(24),o=n(174),a=(n(85),n(96)),u=n(177);n(2);void 0!==e&&e.env;var c={instantiateChildren:function(t,e,n,i){if(null==t)return null;var o={};return u(t,r,o),o},updateChildren:function(t,e,n,r,u,c,s,l,f){if(e||t){var p,h;for(p in e)if(e.hasOwnProperty(p)){h=t&&t[p];var d=h&&h._currentElement,v=e[p];if(null!=h&&a(d,v))i.receiveComponent(h,v,u,l),e[p]=h;else{h&&(r[p]=i.getHostNode(h),i.unmountComponent(h,!1));var g=o(v,!0);e[p]=g;var m=i.mountComponent(g,u,c,s,l,f);n.push(m)}}for(p in t)!t.hasOwnProperty(p)||e&&e.hasOwnProperty(p)||(h=t[p],r[p]=i.getHostNode(h),i.unmountComponent(h,!1))}},unmountChildren:function(t,e){for(var n in t)if(t.hasOwnProperty(n)){var r=t[n];i.unmountComponent(r,e)}}};t.exports=c}).call(e,n(156))},function(t,e,n){"use strict";var r=n(82),i=n(364),o={processChildrenUpdates:i.dangerouslyProcessChildrenUpdates,replaceNodeWithMarkup:r.dangerouslyReplaceNodeWithMarkup};t.exports=o},function(t,e,n){"use strict";function r(t){}function i(t){return!(!t.prototype||!t.prototype.isReactComponent)}function o(t){return!(!t.prototype||!t.prototype.isPureReactComponent)}var a=n(1),u=n(3),c=n(26),s=n(87),l=n(15),f=n(88),p=n(39),h=(n(9),n(168)),d=n(24),v=n(51),g=(n(0),n(81)),m=n(96),y=(n(2),{ImpureClass:0,PureClass:1,StatelessFunctional:2});r.prototype.render=function(){var t=p.get(this)._currentElement.type,e=t(this.props,this.context,this.updater);return e};var _=1,b={construct:function(t){this._currentElement=t,this._rootNodeID=0,this._compositeType=null,this._instance=null,this._hostParent=null,this._hostContainerInfo=null,this._updateBatchNumber=null,this._pendingElement=null,this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1,this._renderedNodeType=null,this._renderedComponent=null,this._context=null,this._mountOrder=0,this._topLevelWrapper=null,this._pendingCallbacks=null,this._calledComponentWillUnmount=!1},mountComponent:function(t,e,n,u){this._context=u,this._mountOrder=_++,this._hostParent=e,this._hostContainerInfo=n;var s,l=this._currentElement.props,f=this._processContext(u),h=this._currentElement.type,d=t.getUpdateQueue(),g=i(h),m=this._constructComponent(g,l,f,d);g||null!=m&&null!=m.render?o(h)?this._compositeType=y.PureClass:this._compositeType=y.ImpureClass:(s=m,null===m||!1===m||c.isValidElement(m)||a("105",h.displayName||h.name||"Component"),m=new r(h),this._compositeType=y.StatelessFunctional);m.props=l,m.context=f,m.refs=v,m.updater=d,this._instance=m,p.set(m,this);var b=m.state;void 0===b&&(m.state=b=null),("object"!=typeof b||Array.isArray(b))&&a("106",this.getName()||"ReactCompositeComponent"),this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1;var x;return x=m.unstable_handleError?this.performInitialMountWithErrorHandling(s,e,n,t,u):this.performInitialMount(s,e,n,t,u),m.componentDidMount&&t.getReactMountReady().enqueue(m.componentDidMount,m),x},_constructComponent:function(t,e,n,r){return this._constructComponentWithoutOwner(t,e,n,r)},_constructComponentWithoutOwner:function(t,e,n,r){var i=this._currentElement.type;return t?new i(e,n,r):i(e,n,r)},performInitialMountWithErrorHandling:function(t,e,n,r,i){var o,a=r.checkpoint();try{o=this.performInitialMount(t,e,n,r,i)}catch(u){r.rollback(a),this._instance.unstable_handleError(u),this._pendingStateQueue&&(this._instance.state=this._processPendingState(this._instance.props,this._instance.context)),a=r.checkpoint(),this._renderedComponent.unmountComponent(!0),r.rollback(a),o=this.performInitialMount(t,e,n,r,i)}return o},performInitialMount:function(t,e,n,r,i){var o=this._instance,a=0;o.componentWillMount&&(o.componentWillMount(),this._pendingStateQueue&&(o.state=this._processPendingState(o.props,o.context))),void 0===t&&(t=this._renderValidatedComponent());var u=h.getType(t);this._renderedNodeType=u;var c=this._instantiateReactComponent(t,u!==h.EMPTY);this._renderedComponent=c;var s=d.mountComponent(c,r,e,n,this._processChildContext(i),a);return s},getHostNode:function(){return d.getHostNode(this._renderedComponent)},unmountComponent:function(t){if(this._renderedComponent){var e=this._instance;if(e.componentWillUnmount&&!e._calledComponentWillUnmount)if(e._calledComponentWillUnmount=!0,t){var n=this.getName()+".componentWillUnmount()";f.invokeGuardedCallback(n,e.componentWillUnmount.bind(e))}else e.componentWillUnmount();this._renderedComponent&&(d.unmountComponent(this._renderedComponent,t),this._renderedNodeType=null,this._renderedComponent=null,this._instance=null),this._pendingStateQueue=null,this._pendingReplaceState=!1,this._pendingForceUpdate=!1,this._pendingCallbacks=null,this._pendingElement=null,this._context=null,this._rootNodeID=0,this._topLevelWrapper=null,p.remove(e)}},_maskContext:function(t){var e=this._currentElement.type,n=e.contextTypes;if(!n)return v;var r={};for(var i in n)r[i]=t[i];return r},_processContext:function(t){var e=this._maskContext(t);return e},_processChildContext:function(t){var e,n=this._currentElement.type,r=this._instance;if(r.getChildContext&&(e=r.getChildContext()),e){"object"!=typeof n.childContextTypes&&a("107",this.getName()||"ReactCompositeComponent");for(var i in e)i in n.childContextTypes||a("108",this.getName()||"ReactCompositeComponent",i);return u({},t,e)}return t},_checkContextTypes:function(t,e,n){},receiveComponent:function(t,e,n){var r=this._currentElement,i=this._context;this._pendingElement=null,this.updateComponent(e,r,t,i,n)},performUpdateIfNecessary:function(t){null!=this._pendingElement?d.receiveComponent(this,this._pendingElement,t,this._context):null!==this._pendingStateQueue||this._pendingForceUpdate?this.updateComponent(t,this._currentElement,this._currentElement,this._context,this._context):this._updateBatchNumber=null},updateComponent:function(t,e,n,r,i){var o=this._instance;null==o&&a("136",this.getName()||"ReactCompositeComponent");var u,c=!1;this._context===i?u=o.context:(u=this._processContext(i),c=!0);var s=e.props,l=n.props;e!==n&&(c=!0),c&&o.componentWillReceiveProps&&o.componentWillReceiveProps(l,u);var f=this._processPendingState(l,u),p=!0;this._pendingForceUpdate||(o.shouldComponentUpdate?p=o.shouldComponentUpdate(l,f,u):this._compositeType===y.PureClass&&(p=!g(s,l)||!g(o.state,f))),this._updateBatchNumber=null,p?(this._pendingForceUpdate=!1,this._performComponentUpdate(n,l,f,u,t,i)):(this._currentElement=n,this._context=i,o.props=l,o.state=f,o.context=u)},_processPendingState:function(t,e){var n=this._instance,r=this._pendingStateQueue,i=this._pendingReplaceState;if(this._pendingReplaceState=!1,this._pendingStateQueue=null,!r)return n.state;if(i&&1===r.length)return r[0];for(var o=u({},i?r[0]:n.state),a=i?1:0;a<r.length;a++){var c=r[a];u(o,"function"==typeof c?c.call(n,o,t,e):c)}return o},_performComponentUpdate:function(t,e,n,r,i,o){var a,u,c,s=this._instance,l=Boolean(s.componentDidUpdate);l&&(a=s.props,u=s.state,c=s.context),s.componentWillUpdate&&s.componentWillUpdate(e,n,r),this._currentElement=t,this._context=o,s.props=e,s.state=n,s.context=r,this._updateRenderedComponent(i,o),l&&i.getReactMountReady().enqueue(s.componentDidUpdate.bind(s,a,u,c),s)},_updateRenderedComponent:function(t,e){var n=this._renderedComponent,r=n._currentElement,i=this._renderValidatedComponent(),o=0;if(m(r,i))d.receiveComponent(n,i,t,this._processChildContext(e));else{var a=d.getHostNode(n);d.unmountComponent(n,!1);var u=h.getType(i);this._renderedNodeType=u;var c=this._instantiateReactComponent(i,u!==h.EMPTY);this._renderedComponent=c;var s=d.mountComponent(c,t,this._hostParent,this._hostContainerInfo,this._processChildContext(e),o);this._replaceNodeWithMarkup(a,s,n)}},_replaceNodeWithMarkup:function(t,e,n){s.replaceNodeWithMarkup(t,e,n)},_renderValidatedComponentWithoutOwnerOrContext:function(){var t=this._instance;return t.render()},_renderValidatedComponent:function(){var t;if(this._compositeType!==y.StatelessFunctional){l.current=this;try{t=this._renderValidatedComponentWithoutOwnerOrContext()}finally{l.current=null}}else t=this._renderValidatedComponentWithoutOwnerOrContext();return null===t||!1===t||c.isValidElement(t)||a("109",this.getName()||"ReactCompositeComponent"),t},attachRef:function(t,e){var n=this.getPublicInstance();null==n&&a("110");var r=e.getPublicInstance();(n.refs===v?n.refs={}:n.refs)[t]=r},detachRef:function(t){delete this.getPublicInstance().refs[t]},getName:function(){var t=this._currentElement.type,e=this._instance&&this._instance.constructor;return t.displayName||e&&e.displayName||t.name||e&&e.name||null},getPublicInstance:function(){var t=this._instance;return this._compositeType===y.StatelessFunctional?null:t},_instantiateReactComponent:null};t.exports=b},function(t,e,n){"use strict";var r=n(4),i=n(372),o=n(167),a=n(24),u=n(12),c=n(385),s=n(401),l=n(171),f=n(408);n(2);i.inject();var p={findDOMNode:s,render:o.render,unmountComponentAtNode:o.unmountComponentAtNode,version:c,unstable_batchedUpdates:u.batchedUpdates,unstable_renderSubtreeIntoContainer:f};"undefined"!=typeof __REACT_DEVTOOLS_GLOBAL_HOOK__&&"function"==typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.inject&&__REACT_DEVTOOLS_GLOBAL_HOOK__.inject({ComponentTree:{getClosestInstanceFromNode:r.getClosestInstanceFromNode,getNodeFromInstance:function(t){return t._renderedComponent&&(t=l(t)),t?r.getNodeFromInstance(t):null}},Mount:o,Reconciler:a});t.exports=p},function(t,e,n){"use strict";function r(t){if(t){var e=t._currentElement._owner||null;if(e){var n=e.getName();if(n)return" This DOM node was rendered by `"+n+"`."}}return""}function i(t,e){e&&($[t._tag]&&(null!=e.children||null!=e.dangerouslySetInnerHTML)&&g("137",t._tag,t._currentElement._owner?" Check the render method of "+t._currentElement._owner.getName()+".":""),null!=e.dangerouslySetInnerHTML&&(null!=e.children&&g("60"),"object"==typeof e.dangerouslySetInnerHTML&&z in e.dangerouslySetInnerHTML||g("61")),null!=e.style&&"object"!=typeof e.style&&g("62",r(t)))}function o(t,e,n,r){if(!(r instanceof D)){var i=t._hostContainerInfo,o=i._node&&i._node.nodeType===q,u=o?i._node:i._ownerDocument;B(e,u),r.getReactMountReady().enqueue(a,{inst:t,registrationName:e,listener:n})}}function a(){var t=this;k.putListener(t.inst,t.registrationName,t.listener)}function u(){var t=this;N.postMountWrapper(t)}function c(){var t=this;O.postMountWrapper(t)}function s(){var t=this;A.postMountWrapper(t)}function l(){L.track(this)}function f(){var t=this;t._rootNodeID||g("63");var e=j(t);switch(e||g("64"),t._tag){case"iframe":case"object":t._wrapperState.listeners=[M.trapBubbledEvent("topLoad","load",e)];break;case"video":case"audio":t._wrapperState.listeners=[];for(var n in Y)Y.hasOwnProperty(n)&&t._wrapperState.listeners.push(M.trapBubbledEvent(n,Y[n],e));break;case"source":t._wrapperState.listeners=[M.trapBubbledEvent("topError","error",e)];break;case"img":t._wrapperState.listeners=[M.trapBubbledEvent("topError","error",e),M.trapBubbledEvent("topLoad","load",e)];break;case"form":t._wrapperState.listeners=[M.trapBubbledEvent("topReset","reset",e),M.trapBubbledEvent("topSubmit","submit",e)];break;case"input":case"select":case"textarea":t._wrapperState.listeners=[M.trapBubbledEvent("topInvalid","invalid",e)]}}function p(){P.postUpdateWrapper(this)}function h(t){Z.call(Q,t)||(X.test(t)||g("65",t),Q[t]=!0)}function d(t,e){return t.indexOf("-")>=0||null!=e.is}function v(t){var e=t.type;h(e),this._currentElement=t,this._tag=e.toLowerCase(),this._namespaceURI=null,this._renderedChildren=null,this._previousStyle=null,this._previousStyleCopy=null,this._hostNode=null,this._hostParent=null,this._rootNodeID=0,this._domID=0,this._hostContainerInfo=null,this._wrapperState=null,this._topLevelWrapper=null,this._flags=0}var g=n(1),m=n(3),y=n(346),_=n(348),b=n(20),x=n(83),w=n(21),C=n(160),k=n(22),E=n(84),M=n(53),T=n(161),S=n(4),N=n(365),A=n(366),P=n(162),O=n(369),I=(n(9),n(378)),D=n(383),R=(n(11),n(56)),L=(n(0),n(95),n(81),n(173)),U=(n(97),n(2),T),F=k.deleteListener,j=S.getNodeFromInstance,B=M.listenTo,V=E.registrationNameModules,W={string:!0,number:!0},z="__html",H={children:null,dangerouslySetInnerHTML:null,suppressContentEditableWarning:null},q=11,Y={topAbort:"abort",topCanPlay:"canplay",topCanPlayThrough:"canplaythrough",topDurationChange:"durationchange",topEmptied:"emptied",topEncrypted:"encrypted",topEnded:"ended",topError:"error",topLoadedData:"loadeddata",topLoadedMetadata:"loadedmetadata",topLoadStart:"loadstart",topPause:"pause",topPlay:"play",topPlaying:"playing",topProgress:"progress",topRateChange:"ratechange",topSeeked:"seeked",topSeeking:"seeking",topStalled:"stalled",topSuspend:"suspend",topTimeUpdate:"timeupdate",topVolumeChange:"volumechange",topWaiting:"waiting"},K={area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0},G={listing:!0,pre:!0,textarea:!0},$=m({menuitem:!0},K),X=/^[a-zA-Z][a-zA-Z:_\.\-\d]*$/,Q={},Z={}.hasOwnProperty,J=1;v.displayName="ReactDOMComponent",v.Mixin={mountComponent:function(t,e,n,r){this._rootNodeID=J++,this._domID=n._idCounter++,this._hostParent=e,this._hostContainerInfo=n;var o=this._currentElement.props;switch(this._tag){case"audio":case"form":case"iframe":case"img":case"link":case"object":case"source":case"video":this._wrapperState={listeners:null},t.getReactMountReady().enqueue(f,this);break;case"input":N.mountWrapper(this,o,e),o=N.getHostProps(this,o),t.getReactMountReady().enqueue(l,this),t.getReactMountReady().enqueue(f,this);break;case"option":A.mountWrapper(this,o,e),o=A.getHostProps(this,o);break;case"select":P.mountWrapper(this,o,e),o=P.getHostProps(this,o),t.getReactMountReady().enqueue(f,this);break;case"textarea":O.mountWrapper(this,o,e),o=O.getHostProps(this,o),t.getReactMountReady().enqueue(l,this),t.getReactMountReady().enqueue(f,this)}i(this,o);var a,p;null!=e?(a=e._namespaceURI,p=e._tag):n._tag&&(a=n._namespaceURI,p=n._tag),(null==a||a===x.svg&&"foreignobject"===p)&&(a=x.html),a===x.html&&("svg"===this._tag?a=x.svg:"math"===this._tag&&(a=x.mathml)),this._namespaceURI=a;var h;if(t.useCreateElement){var d,v=n._ownerDocument;if(a===x.html)if("script"===this._tag){var g=v.createElement("div"),m=this._currentElement.type;g.innerHTML="<"+m+"></"+m+">",d=g.removeChild(g.firstChild)}else d=o.is?v.createElement(this._currentElement.type,o.is):v.createElement(this._currentElement.type);else d=v.createElementNS(a,this._currentElement.type);S.precacheNode(this,d),this._flags|=U.hasCachedChildNodes,this._hostParent||C.setAttributeForRoot(d),this._updateDOMProperties(null,o,t);var _=b(d);this._createInitialChildren(t,o,r,_),h=_}else{var w=this._createOpenTagMarkupAndPutListeners(t,o),k=this._createContentMarkup(t,o,r);h=!k&&K[this._tag]?w+"/>":w+">"+k+"</"+this._currentElement.type+">"}switch(this._tag){case"input":t.getReactMountReady().enqueue(u,this),o.autoFocus&&t.getReactMountReady().enqueue(y.focusDOMComponent,this);break;case"textarea":t.getReactMountReady().enqueue(c,this),o.autoFocus&&t.getReactMountReady().enqueue(y.focusDOMComponent,this);break;case"select":case"button":o.autoFocus&&t.getReactMountReady().enqueue(y.focusDOMComponent,this);break;case"option":t.getReactMountReady().enqueue(s,this)}return h},_createOpenTagMarkupAndPutListeners:function(t,e){var n="<"+this._currentElement.type;for(var r in e)if(e.hasOwnProperty(r)){var i=e[r];if(null!=i)if(V.hasOwnProperty(r))i&&o(this,r,i,t);else{"style"===r&&(i&&(i=this._previousStyleCopy=m({},e.style)),i=_.createMarkupForStyles(i,this));var a=null;null!=this._tag&&d(this._tag,e)?H.hasOwnProperty(r)||(a=C.createMarkupForCustomAttribute(r,i)):a=C.createMarkupForProperty(r,i),a&&(n+=" "+a)}}return t.renderToStaticMarkup?n:(this._hostParent||(n+=" "+C.createMarkupForRoot()),n+=" "+C.createMarkupForID(this._domID))},_createContentMarkup:function(t,e,n){var r="",i=e.dangerouslySetInnerHTML;if(null!=i)null!=i.__html&&(r=i.__html);else{var o=W[typeof e.children]?e.children:null,a=null!=o?null:e.children;if(null!=o)r=R(o);else if(null!=a){var u=this.mountChildren(a,t,n);r=u.join("")}}return G[this._tag]&&"\n"===r.charAt(0)?"\n"+r:r},_createInitialChildren:function(t,e,n,r){var i=e.dangerouslySetInnerHTML;if(null!=i)null!=i.__html&&b.queueHTML(r,i.__html);else{var o=W[typeof e.children]?e.children:null,a=null!=o?null:e.children;if(null!=o)""!==o&&b.queueText(r,o);else if(null!=a)for(var u=this.mountChildren(a,t,n),c=0;c<u.length;c++)b.queueChild(r,u[c])}},receiveComponent:function(t,e,n){var r=this._currentElement;this._currentElement=t,this.updateComponent(e,r,t,n)},updateComponent:function(t,e,n,r){var o=e.props,a=this._currentElement.props;switch(this._tag){case"input":o=N.getHostProps(this,o),a=N.getHostProps(this,a);break;case"option":o=A.getHostProps(this,o),a=A.getHostProps(this,a);break;case"select":o=P.getHostProps(this,o),a=P.getHostProps(this,a);break;case"textarea":o=O.getHostProps(this,o),a=O.getHostProps(this,a)}switch(i(this,a),this._updateDOMProperties(o,a,t),this._updateDOMChildren(o,a,t,r),this._tag){case"input":N.updateWrapper(this),L.updateValueIfChanged(this);break;case"textarea":O.updateWrapper(this);break;case"select":t.getReactMountReady().enqueue(p,this)}},_updateDOMProperties:function(t,e,n){var r,i,a;for(r in t)if(!e.hasOwnProperty(r)&&t.hasOwnProperty(r)&&null!=t[r])if("style"===r){var u=this._previousStyleCopy;for(i in u)u.hasOwnProperty(i)&&(a=a||{},a[i]="");this._previousStyleCopy=null}else V.hasOwnProperty(r)?t[r]&&F(this,r):d(this._tag,t)?H.hasOwnProperty(r)||C.deleteValueForAttribute(j(this),r):(w.properties[r]||w.isCustomAttribute(r))&&C.deleteValueForProperty(j(this),r);for(r in e){var c=e[r],s="style"===r?this._previousStyleCopy:null!=t?t[r]:void 0;if(e.hasOwnProperty(r)&&c!==s&&(null!=c||null!=s))if("style"===r)if(c?c=this._previousStyleCopy=m({},c):this._previousStyleCopy=null,s){for(i in s)!s.hasOwnProperty(i)||c&&c.hasOwnProperty(i)||(a=a||{},a[i]="");for(i in c)c.hasOwnProperty(i)&&s[i]!==c[i]&&(a=a||{},a[i]=c[i])}else a=c;else if(V.hasOwnProperty(r))c?o(this,r,c,n):s&&F(this,r);else if(d(this._tag,e))H.hasOwnProperty(r)||C.setValueForAttribute(j(this),r,c);else if(w.properties[r]||w.isCustomAttribute(r)){var l=j(this);null!=c?C.setValueForProperty(l,r,c):C.deleteValueForProperty(l,r)}}a&&_.setValueForStyles(j(this),a,this)},_updateDOMChildren:function(t,e,n,r){var i=W[typeof t.children]?t.children:null,o=W[typeof e.children]?e.children:null,a=t.dangerouslySetInnerHTML&&t.dangerouslySetInnerHTML.__html,u=e.dangerouslySetInnerHTML&&e.dangerouslySetInnerHTML.__html,c=null!=i?null:t.children,s=null!=o?null:e.children,l=null!=i||null!=a,f=null!=o||null!=u;null!=c&&null==s?this.updateChildren(null,n,r):l&&!f&&this.updateTextContent(""),null!=o?i!==o&&this.updateTextContent(""+o):null!=u?a!==u&&this.updateMarkup(""+u):null!=s&&this.updateChildren(s,n,r)},getHostNode:function(){return j(this)},unmountComponent:function(t){switch(this._tag){case"audio":case"form":case"iframe":case"img":case"link":case"object":case"source":case"video":var e=this._wrapperState.listeners;if(e)for(var n=0;n<e.length;n++)e[n].remove();break;case"input":case"textarea":L.stopTracking(this);break;case"html":case"head":case"body":g("66",this._tag)}this.unmountChildren(t),S.uncacheNode(this),k.deleteAllListeners(this),this._rootNodeID=0,this._domID=0,this._wrapperState=null},getPublicInstance:function(){return j(this)}},m(v.prototype,v.Mixin,I.Mixin),t.exports=v},function(t,e,n){"use strict";function r(t,e){var n={_topLevelWrapper:t,_idCounter:1,_ownerDocument:e?e.nodeType===i?e:e.ownerDocument:null,_node:e,_tag:e?e.nodeName.toLowerCase():null,_namespaceURI:e?e.namespaceURI:null};return n}var i=(n(97),9);t.exports=r},function(t,e,n){"use strict";var r=n(3),i=n(20),o=n(4),a=function(t){this._currentElement=null,this._hostNode=null,this._hostParent=null,this._hostContainerInfo=null,this._domID=0};r(a.prototype,{mountComponent:function(t,e,n,r){var a=n._idCounter++;this._domID=a,this._hostParent=e,this._hostContainerInfo=n;var u=" react-empty: "+this._domID+" ";if(t.useCreateElement){var c=n._ownerDocument,s=c.createComment(u);return o.precacheNode(this,s),i(s)}return t.renderToStaticMarkup?"":"\x3c!--"+u+"--\x3e"},receiveComponent:function(){},getHostNode:function(){return o.getNodeFromInstance(this)},unmountComponent:function(){o.uncacheNode(this)}}),t.exports=a},function(t,e,n){"use strict";var r={useCreateElement:!0,useFiber:!1};t.exports=r},function(t,e,n){"use strict";var r=n(82),i=n(4),o={dangerouslyProcessChildrenUpdates:function(t,e){var n=i.getNodeFromInstance(t);r.processUpdates(n,e)}};t.exports=o},function(t,e,n){"use strict";function r(){this._rootNodeID&&p.updateWrapper(this)}function i(t){return"checkbox"===t.type||"radio"===t.type?null!=t.checked:null!=t.value}function o(t){var e=this._currentElement.props,n=s.executeOnChange(e,t);f.asap(r,this);var i=e.name;if("radio"===e.type&&null!=i){for(var o=l.getNodeFromInstance(this),u=o;u.parentNode;)u=u.parentNode;for(var c=u.querySelectorAll("input[name="+JSON.stringify(""+i)+'][type="radio"]'),p=0;p<c.length;p++){var h=c[p];if(h!==o&&h.form===o.form){var d=l.getInstanceFromNode(h);d||a("90"),f.asap(r,d)}}}return n}var a=n(1),u=n(3),c=n(160),s=n(86),l=n(4),f=n(12),p=(n(0),n(2),{getHostProps:function(t,e){var n=s.getValue(e),r=s.getChecked(e);return u({type:void 0,step:void 0,min:void 0,max:void 0},e,{defaultChecked:void 0,defaultValue:void 0,value:null!=n?n:t._wrapperState.initialValue,checked:null!=r?r:t._wrapperState.initialChecked,onChange:t._wrapperState.onChange})},mountWrapper:function(t,e){var n=e.defaultValue;t._wrapperState={initialChecked:null!=e.checked?e.checked:e.defaultChecked,initialValue:null!=e.value?e.value:n,listeners:null,onChange:o.bind(t),controlled:i(e)}},updateWrapper:function(t){var e=t._currentElement.props,n=e.checked;null!=n&&c.setValueForProperty(l.getNodeFromInstance(t),"checked",n||!1);var r=l.getNodeFromInstance(t),i=s.getValue(e);if(null!=i)if(0===i&&""===r.value)r.value="0";else if("number"===e.type){var o=parseFloat(r.value,10)||0;(i!=o||i==o&&r.value!=i)&&(r.value=""+i)}else r.value!==""+i&&(r.value=""+i);else null==e.value&&null!=e.defaultValue&&r.defaultValue!==""+e.defaultValue&&(r.defaultValue=""+e.defaultValue),null==e.checked&&null!=e.defaultChecked&&(r.defaultChecked=!!e.defaultChecked)},postMountWrapper:function(t){var e=t._currentElement.props,n=l.getNodeFromInstance(t);switch(e.type){case"submit":case"reset":break;case"color":case"date":case"datetime":case"datetime-local":case"month":case"time":case"week":n.value="",n.value=n.defaultValue;break;default:n.value=n.value}var r=n.name;""!==r&&(n.name=""),n.defaultChecked=!n.defaultChecked,n.defaultChecked=!n.defaultChecked,""!==r&&(n.name=r)}});t.exports=p},function(t,e,n){"use strict";function r(t){var e="";return o.Children.forEach(t,function(t){null!=t&&("string"==typeof t||"number"==typeof t?e+=t:c||(c=!0))}),e}var i=n(3),o=n(26),a=n(4),u=n(162),c=(n(2),!1),s={mountWrapper:function(t,e,n){var i=null;if(null!=n){var o=n;"optgroup"===o._tag&&(o=o._hostParent),null!=o&&"select"===o._tag&&(i=u.getSelectValueContext(o))}var a=null;if(null!=i){var c;if(c=null!=e.value?e.value+"":r(e.children),a=!1,Array.isArray(i)){for(var s=0;s<i.length;s++)if(""+i[s]===c){a=!0;break}}else a=""+i===c}t._wrapperState={selected:a}},postMountWrapper:function(t){var e=t._currentElement.props;if(null!=e.value){a.getNodeFromInstance(t).setAttribute("value",e.value)}},getHostProps:function(t,e){var n=i({selected:void 0,children:void 0},e);null!=t._wrapperState.selected&&(n.selected=t._wrapperState.selected);var o=r(e.children);return o&&(n.children=o),n}};t.exports=s},function(t,e,n){"use strict";function r(t,e,n,r){return t===n&&e===r}function i(t){var e=document.selection,n=e.createRange(),r=n.text.length,i=n.duplicate();i.moveToElementText(t),i.setEndPoint("EndToStart",n);var o=i.text.length;return{start:o,end:o+r}}function o(t){var e=window.getSelection&&window.getSelection();if(!e||0===e.rangeCount)return null;var n=e.anchorNode,i=e.anchorOffset,o=e.focusNode,a=e.focusOffset,u=e.getRangeAt(0);try{u.startContainer.nodeType,u.endContainer.nodeType}catch(t){return null}var c=r(e.anchorNode,e.anchorOffset,e.focusNode,e.focusOffset),s=c?0:u.toString().length,l=u.cloneRange();l.selectNodeContents(t),l.setEnd(u.startContainer,u.startOffset);var f=r(l.startContainer,l.startOffset,l.endContainer,l.endOffset),p=f?0:l.toString().length,h=p+s,d=document.createRange();d.setStart(n,i),d.setEnd(o,a);var v=d.collapsed;return{start:v?h:p,end:v?p:h}}function a(t,e){var n,r,i=document.selection.createRange().duplicate();void 0===e.end?(n=e.start,r=n):e.start>e.end?(n=e.end,r=e.start):(n=e.start,r=e.end),i.moveToElementText(t),i.moveStart("character",n),i.setEndPoint("EndToStart",i),i.moveEnd("character",r-n),i.select()}function u(t,e){if(window.getSelection){var n=window.getSelection(),r=t[l()].length,i=Math.min(e.start,r),o=void 0===e.end?i:Math.min(e.end,r);if(!n.extend&&i>o){var a=o;o=i,i=a}var u=s(t,i),c=s(t,o);if(u&&c){var f=document.createRange();f.setStart(u.node,u.offset),n.removeAllRanges(),i>o?(n.addRange(f),n.extend(c.node,c.offset)):(f.setEnd(c.node,c.offset),n.addRange(f))}}}var c=n(6),s=n(405),l=n(172),f=c.canUseDOM&&"selection"in document&&!("getSelection"in window),p={getOffsets:f?i:o,setOffsets:f?a:u};t.exports=p},function(t,e,n){"use strict";var r=n(1),i=n(3),o=n(82),a=n(20),u=n(4),c=n(56),s=(n(0),n(97),function(t){this._currentElement=t,this._stringText=""+t,this._hostNode=null,this._hostParent=null,this._domID=0,this._mountIndex=0,this._closingComment=null,this._commentNodes=null});i(s.prototype,{mountComponent:function(t,e,n,r){var i=n._idCounter++,o=" react-text: "+i+" ";if(this._domID=i,this._hostParent=e,t.useCreateElement){var s=n._ownerDocument,l=s.createComment(o),f=s.createComment(" /react-text "),p=a(s.createDocumentFragment());return a.queueChild(p,a(l)),this._stringText&&a.queueChild(p,a(s.createTextNode(this._stringText))),a.queueChild(p,a(f)),u.precacheNode(this,l),this._closingComment=f,p}var h=c(this._stringText);return t.renderToStaticMarkup?h:"\x3c!--"+o+"--\x3e"+h+"\x3c!-- /react-text --\x3e"},receiveComponent:function(t,e){if(t!==this._currentElement){this._currentElement=t;var n=""+t;if(n!==this._stringText){this._stringText=n;var r=this.getHostNode();o.replaceDelimitedText(r[0],r[1],n)}}},getHostNode:function(){var t=this._commentNodes;if(t)return t;if(!this._closingComment)for(var e=u.getNodeFromInstance(this),n=e.nextSibling;;){if(null==n&&r("67",this._domID),8===n.nodeType&&" /react-text "===n.nodeValue){this._closingComment=n;break}n=n.nextSibling}return t=[this._hostNode,this._closingComment],this._commentNodes=t,t},unmountComponent:function(){this._closingComment=null,this._commentNodes=null,u.uncacheNode(this)}}),t.exports=s},function(t,e,n){"use strict";function r(){this._rootNodeID&&l.updateWrapper(this)}function i(t){var e=this._currentElement.props,n=u.executeOnChange(e,t);return s.asap(r,this),n}var o=n(1),a=n(3),u=n(86),c=n(4),s=n(12),l=(n(0),n(2),{getHostProps:function(t,e){return null!=e.dangerouslySetInnerHTML&&o("91"),a({},e,{value:void 0,defaultValue:void 0,children:""+t._wrapperState.initialValue,onChange:t._wrapperState.onChange})},mountWrapper:function(t,e){var n=u.getValue(e),r=n;if(null==n){var a=e.defaultValue,c=e.children;null!=c&&(null!=a&&o("92"),Array.isArray(c)&&(c.length<=1||o("93"),c=c[0]),a=""+c),null==a&&(a=""),r=a}t._wrapperState={initialValue:""+r,listeners:null,onChange:i.bind(t)}},updateWrapper:function(t){var e=t._currentElement.props,n=c.getNodeFromInstance(t),r=u.getValue(e);if(null!=r){var i=""+r;i!==n.value&&(n.value=i),null==e.defaultValue&&(n.defaultValue=i)}null!=e.defaultValue&&(n.defaultValue=e.defaultValue)},postMountWrapper:function(t){var e=c.getNodeFromInstance(t),n=e.textContent;n===t._wrapperState.initialValue&&(e.value=n)}});t.exports=l},function(t,e,n){"use strict";function r(t,e){"_hostNode"in t||c("33"),"_hostNode"in e||c("33");for(var n=0,r=t;r;r=r._hostParent)n++;for(var i=0,o=e;o;o=o._hostParent)i++;for(;n-i>0;)t=t._hostParent,n--;for(;i-n>0;)e=e._hostParent,i--;for(var a=n;a--;){if(t===e)return t;t=t._hostParent,e=e._hostParent}return null}function i(t,e){"_hostNode"in t||c("35"),"_hostNode"in e||c("35");for(;e;){if(e===t)return!0;e=e._hostParent}return!1}function o(t){return"_hostNode"in t||c("36"),t._hostParent}function a(t,e,n){for(var r=[];t;)r.push(t),t=t._hostParent;var i;for(i=r.length;i-- >0;)e(r[i],"captured",n);for(i=0;i<r.length;i++)e(r[i],"bubbled",n)}function u(t,e,n,i,o){for(var a=t&&e?r(t,e):null,u=[];t&&t!==a;)u.push(t),t=t._hostParent;for(var c=[];e&&e!==a;)c.push(e),e=e._hostParent;var s;for(s=0;s<u.length;s++)n(u[s],"bubbled",i);for(s=c.length;s-- >0;)n(c[s],"captured",o)}var c=n(1);n(0);t.exports={isAncestor:i,getLowestCommonAncestor:r,getParentInstance:o,traverseTwoPhase:a,traverseEnterLeave:u}},function(t,e,n){"use strict";function r(){this.reinitializeTransaction()}var i=n(3),o=n(12),a=n(55),u=n(11),c={initialize:u,close:function(){p.isBatchingUpdates=!1}},s={initialize:u,close:o.flushBatchedUpdates.bind(o)},l=[s,c];i(r.prototype,a,{getTransactionWrappers:function(){return l}});var f=new r,p={isBatchingUpdates:!1,batchedUpdates:function(t,e,n,r,i,o){var a=p.isBatchingUpdates;return p.isBatchingUpdates=!0,a?t(e,n,r,i,o):f.perform(t,null,e,n,r,i,o)}};t.exports=p},function(t,e,n){"use strict";function r(){C||(C=!0,y.EventEmitter.injectReactEventListener(m),y.EventPluginHub.injectEventPluginOrder(u),y.EventPluginUtils.injectComponentTree(p),y.EventPluginUtils.injectTreeTraversal(d),y.EventPluginHub.injectEventPluginsByName({SimpleEventPlugin:w,EnterLeaveEventPlugin:c,ChangeEventPlugin:a,SelectEventPlugin:x,BeforeInputEventPlugin:o}),y.HostComponent.injectGenericComponentClass(f),y.HostComponent.injectTextComponentClass(v),y.DOMProperty.injectDOMPropertyConfig(i),y.DOMProperty.injectDOMPropertyConfig(s),y.DOMProperty.injectDOMPropertyConfig(b),y.EmptyComponent.injectEmptyComponentFactory(function(t){return new h(t)}),y.Updates.injectReconcileTransaction(_),y.Updates.injectBatchingStrategy(g),y.Component.injectEnvironment(l))}var i=n(345),o=n(347),a=n(349),u=n(351),c=n(352),s=n(355),l=n(357),f=n(360),p=n(4),h=n(362),d=n(370),v=n(368),g=n(371),m=n(375),y=n(376),_=n(381),b=n(386),x=n(387),w=n(388),C=!1;t.exports={inject:r}},function(t,e,n){"use strict";var r="function"==typeof Symbol&&Symbol.for&&Symbol.for("react.element")||60103;t.exports=r},function(t,e,n){"use strict";function r(t){i.enqueueEvents(t),i.processEventQueue(!1)}var i=n(22),o={handleTopLevel:function(t,e,n,o){r(i.extractEvents(t,e,n,o))}};t.exports=o},function(t,e,n){"use strict";function r(t){for(;t._hostParent;)t=t._hostParent;var e=f.getNodeFromInstance(t),n=e.parentNode;return f.getClosestInstanceFromNode(n)}function i(t,e){this.topLevelType=t,this.nativeEvent=e,this.ancestors=[]}function o(t){var e=h(t.nativeEvent),n=f.getClosestInstanceFromNode(e),i=n;do{t.ancestors.push(i),i=i&&r(i)}while(i);for(var o=0;o<t.ancestors.length;o++)n=t.ancestors[o],v._handleTopLevel(t.topLevelType,n,t.nativeEvent,h(t.nativeEvent))}function a(t){t(d(window))}var u=n(3),c=n(153),s=n(6),l=n(18),f=n(4),p=n(12),h=n(94),d=n(335);u(i.prototype,{destructor:function(){this.topLevelType=null,this.nativeEvent=null,this.ancestors.length=0}}),l.addPoolingTo(i,l.twoArgumentPooler);var v={_enabled:!0,_handleTopLevel:null,WINDOW_HANDLE:s.canUseDOM?window:null,setHandleTopLevel:function(t){v._handleTopLevel=t},setEnabled:function(t){v._enabled=!!t},isEnabled:function(){return v._enabled},trapBubbledEvent:function(t,e,n){return n?c.listen(n,e,v.dispatchEvent.bind(null,t)):null},trapCapturedEvent:function(t,e,n){return n?c.capture(n,e,v.dispatchEvent.bind(null,t)):null},monitorScrollValue:function(t){var e=a.bind(null,t);c.listen(window,"scroll",e)},dispatchEvent:function(t,e){if(v._enabled){var n=i.getPooled(t,e);try{p.batchedUpdates(o,n)}finally{i.release(n)}}}};t.exports=v},function(t,e,n){"use strict";var r=n(21),i=n(22),o=n(52),a=n(87),u=n(163),c=n(53),s=n(165),l=n(12),f={Component:a.injection,DOMProperty:r.injection,EmptyComponent:u.injection,EventPluginHub:i.injection,EventPluginUtils:o.injection,EventEmitter:c.injection,HostComponent:s.injection,Updates:l.injection};t.exports=f},function(t,e,n){"use strict";var r=n(399),i=/\/?>/,o=/^<\!\-\-/,a={CHECKSUM_ATTR_NAME:"data-react-checksum",addChecksumToMarkup:function(t){var e=r(t);return o.test(t)?t:t.replace(i," "+a.CHECKSUM_ATTR_NAME+'="'+e+'"$&')},canReuseMarkup:function(t,e){var n=e.getAttribute(a.CHECKSUM_ATTR_NAME);return n=n&&parseInt(n,10),r(t)===n}};t.exports=a},function(t,e,n){"use strict";function r(t,e,n){return{type:"INSERT_MARKUP",content:t,fromIndex:null,fromNode:null,toIndex:n,afterNode:e}}function i(t,e,n){return{type:"MOVE_EXISTING",content:null,fromIndex:t._mountIndex,fromNode:p.getHostNode(t),toIndex:n,afterNode:e}}function o(t,e){return{type:"REMOVE_NODE",content:null,fromIndex:t._mountIndex,fromNode:e,toIndex:null,afterNode:null}}function a(t){return{type:"SET_MARKUP",content:t,fromIndex:null,fromNode:null,toIndex:null,afterNode:null}}function u(t){return{type:"TEXT_CONTENT",content:t,fromIndex:null,fromNode:null,toIndex:null,afterNode:null}}function c(t,e){return e&&(t=t||[],t.push(e)),t}function s(t,e){f.processChildrenUpdates(t,e)}var l=n(1),f=n(87),p=(n(39),n(9),n(15),n(24)),h=n(356),d=(n(11),n(402)),v=(n(0),{Mixin:{_reconcilerInstantiateChildren:function(t,e,n){return h.instantiateChildren(t,e,n)},_reconcilerUpdateChildren:function(t,e,n,r,i,o){var a,u=0;return a=d(e,u),h.updateChildren(t,a,n,r,i,this,this._hostContainerInfo,o,u),a},mountChildren:function(t,e,n){var r=this._reconcilerInstantiateChildren(t,e,n);this._renderedChildren=r;var i=[],o=0;for(var a in r)if(r.hasOwnProperty(a)){var u=r[a],c=0,s=p.mountComponent(u,e,this,this._hostContainerInfo,n,c);u._mountIndex=o++,i.push(s)}return i},updateTextContent:function(t){var e=this._renderedChildren;h.unmountChildren(e,!1);for(var n in e)e.hasOwnProperty(n)&&l("118");s(this,[u(t)])},updateMarkup:function(t){var e=this._renderedChildren;h.unmountChildren(e,!1);for(var n in e)e.hasOwnProperty(n)&&l("118");s(this,[a(t)])},updateChildren:function(t,e,n){this._updateChildren(t,e,n)},_updateChildren:function(t,e,n){var r=this._renderedChildren,i={},o=[],a=this._reconcilerUpdateChildren(r,t,o,i,e,n);if(a||r){var u,l=null,f=0,h=0,d=0,v=null;for(u in a)if(a.hasOwnProperty(u)){var g=r&&r[u],m=a[u];g===m?(l=c(l,this.moveChild(g,v,f,h)),h=Math.max(g._mountIndex,h),g._mountIndex=f):(g&&(h=Math.max(g._mountIndex,h)),l=c(l,this._mountChildAtIndex(m,o[d],v,f,e,n)),d++),f++,v=p.getHostNode(m)}for(u in i)i.hasOwnProperty(u)&&(l=c(l,this._unmountChild(r[u],i[u])));l&&s(this,l),this._renderedChildren=a}},unmountChildren:function(t){var e=this._renderedChildren;h.unmountChildren(e,t),this._renderedChildren=null},moveChild:function(t,e,n,r){if(t._mountIndex<r)return i(t,e,n)},createChild:function(t,e,n){return r(n,e,t._mountIndex)},removeChild:function(t,e){return o(t,e)},_mountChildAtIndex:function(t,e,n,r,i,o){return t._mountIndex=r,this.createChild(t,n,e)},_unmountChild:function(t,e){var n=this.removeChild(t,e);return t._mountIndex=null,n}}});t.exports=v},function(t,e,n){"use strict";function r(t){return!(!t||"function"!=typeof t.attachRef||"function"!=typeof t.detachRef)}var i=n(1),o=(n(0),{addComponentAsRefTo:function(t,e,n){r(n)||i("119"),n.attachRef(e,t)},removeComponentAsRefFrom:function(t,e,n){r(n)||i("120");var o=n.getPublicInstance();o&&o.refs[e]===t.getPublicInstance()&&n.detachRef(e)}});t.exports=o},function(t,e,n){"use strict";t.exports="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED"},function(t,e,n){"use strict";function r(t){this.reinitializeTransaction(),this.renderToStaticMarkup=!1,this.reactMountReady=o.getPooled(null),this.useCreateElement=t}var i=n(3),o=n(159),a=n(18),u=n(53),c=n(166),s=(n(9),n(55)),l=n(89),f={initialize:c.getSelectionInformation,close:c.restoreSelection},p={initialize:function(){var t=u.isEnabled();return u.setEnabled(!1),t},close:function(t){u.setEnabled(t)}},h={initialize:function(){this.reactMountReady.reset()},close:function(){this.reactMountReady.notifyAll()}},d=[f,p,h],v={getTransactionWrappers:function(){return d},getReactMountReady:function(){return this.reactMountReady},getUpdateQueue:function(){return l},checkpoint:function(){return this.reactMountReady.checkpoint()},rollback:function(t){this.reactMountReady.rollback(t)},destructor:function(){o.release(this.reactMountReady),this.reactMountReady=null}};i(r.prototype,s,v),a.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";function r(t,e,n){"function"==typeof t?t(e.getPublicInstance()):o.addComponentAsRefTo(e,t,n)}function i(t,e,n){"function"==typeof t?t(null):o.removeComponentAsRefFrom(e,t,n)}var o=n(379),a={};a.attachRefs=function(t,e){if(null!==e&&"object"==typeof e){var n=e.ref;null!=n&&r(n,t,e._owner)}},a.shouldUpdateRefs=function(t,e){var n=null,r=null;null!==t&&"object"==typeof t&&(n=t.ref,r=t._owner);var i=null,o=null;return null!==e&&"object"==typeof e&&(i=e.ref,o=e._owner),n!==i||"string"==typeof i&&o!==r},a.detachRefs=function(t,e){if(null!==e&&"object"==typeof e){var n=e.ref;null!=n&&i(n,t,e._owner)}},t.exports=a},function(t,e,n){"use strict";function r(t){this.reinitializeTransaction(),this.renderToStaticMarkup=t,this.useCreateElement=!1,this.updateQueue=new u(this)}var i=n(3),o=n(18),a=n(55),u=(n(9),n(384)),c=[],s={enqueue:function(){}},l={getTransactionWrappers:function(){return c},getReactMountReady:function(){return s},getUpdateQueue:function(){return this.updateQueue},destructor:function(){},checkpoint:function(){},rollback:function(){}};i(r.prototype,a,l),o.addPoolingTo(r),t.exports=r},function(t,e,n){"use strict";function r(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}var i=n(89),o=(n(2),function(){function t(e){r(this,t),this.transaction=e}return t.prototype.isMounted=function(t){return!1},t.prototype.enqueueCallback=function(t,e,n){this.transaction.isInTransaction()&&i.enqueueCallback(t,e,n)},t.prototype.enqueueForceUpdate=function(t){this.transaction.isInTransaction()&&i.enqueueForceUpdate(t)},t.prototype.enqueueReplaceState=function(t,e){this.transaction.isInTransaction()&&i.enqueueReplaceState(t,e)},t.prototype.enqueueSetState=function(t,e){this.transaction.isInTransaction()&&i.enqueueSetState(t,e)},t}());t.exports=o},function(t,e,n){"use strict";t.exports="15.6.2"},function(t,e,n){"use strict";var r={xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace"},i={accentHeight:"accent-height",accumulate:0,additive:0,alignmentBaseline:"alignment-baseline",allowReorder:"allowReorder",alphabetic:0,amplitude:0,arabicForm:"arabic-form",ascent:0,attributeName:"attributeName",attributeType:"attributeType",autoReverse:"autoReverse",azimuth:0,baseFrequency:"baseFrequency",baseProfile:"baseProfile",baselineShift:"baseline-shift",bbox:0,begin:0,bias:0,by:0,calcMode:"calcMode",capHeight:"cap-height",clip:0,clipPath:"clip-path",clipRule:"clip-rule",clipPathUnits:"clipPathUnits",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",contentScriptType:"contentScriptType",contentStyleType:"contentStyleType",cursor:0,cx:0,cy:0,d:0,decelerate:0,descent:0,diffuseConstant:"diffuseConstant",direction:0,display:0,divisor:0,dominantBaseline:"dominant-baseline",dur:0,dx:0,dy:0,edgeMode:"edgeMode",elevation:0,enableBackground:"enable-background",end:0,exponent:0,externalResourcesRequired:"externalResourcesRequired",fill:0,fillOpacity:"fill-opacity",fillRule:"fill-rule",filter:0,filterRes:"filterRes",filterUnits:"filterUnits",floodColor:"flood-color",floodOpacity:"flood-opacity",focusable:0,fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",format:0,from:0,fx:0,fy:0,g1:0,g2:0,glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",glyphRef:"glyphRef",gradientTransform:"gradientTransform",gradientUnits:"gradientUnits",hanging:0,horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",ideographic:0,imageRendering:"image-rendering",in:0,in2:0,intercept:0,k:0,k1:0,k2:0,k3:0,k4:0,kernelMatrix:"kernelMatrix",kernelUnitLength:"kernelUnitLength",kerning:0,keyPoints:"keyPoints",keySplines:"keySplines",keyTimes:"keyTimes",lengthAdjust:"lengthAdjust",letterSpacing:"letter-spacing",lightingColor:"lighting-color",limitingConeAngle:"limitingConeAngle",local:0,markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",markerHeight:"markerHeight",markerUnits:"markerUnits",markerWidth:"markerWidth",mask:0,maskContentUnits:"maskContentUnits",maskUnits:"maskUnits",mathematical:0,mode:0,numOctaves:"numOctaves",offset:0,opacity:0,operator:0,order:0,orient:0,orientation:0,origin:0,overflow:0,overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pathLength:"pathLength",patternContentUnits:"patternContentUnits",patternTransform:"patternTransform",patternUnits:"patternUnits",pointerEvents:"pointer-events",points:0,pointsAtX:"pointsAtX",pointsAtY:"pointsAtY",pointsAtZ:"pointsAtZ",preserveAlpha:"preserveAlpha",preserveAspectRatio:"preserveAspectRatio",primitiveUnits:"primitiveUnits",r:0,radius:0,refX:"refX",refY:"refY",renderingIntent:"rendering-intent",repeatCount:"repeatCount",repeatDur:"repeatDur",requiredExtensions:"requiredExtensions",requiredFeatures:"requiredFeatures",restart:0,result:0,rotate:0,rx:0,ry:0,scale:0,seed:0,shapeRendering:"shape-rendering",slope:0,spacing:0,specularConstant:"specularConstant",specularExponent:"specularExponent",speed:0,spreadMethod:"spreadMethod",startOffset:"startOffset",stdDeviation:"stdDeviation",stemh:0,stemv:0,stitchTiles:"stitchTiles",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",string:0,stroke:0,strokeDasharray:"stroke-dasharray",strokeDashoffset:"stroke-dashoffset",strokeLinecap:"stroke-linecap",strokeLinejoin:"stroke-linejoin",strokeMiterlimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",surfaceScale:"surfaceScale",systemLanguage:"systemLanguage",tableValues:"tableValues",targetX:"targetX",targetY:"targetY",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",textLength:"textLength",to:0,transform:0,u1:0,u2:0,underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicode:0,unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",values:0,vectorEffect:"vector-effect",version:0,vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",viewBox:"viewBox",viewTarget:"viewTarget",visibility:0,widths:0,wordSpacing:"word-spacing",writingMode:"writing-mode",x:0,xHeight:"x-height",x1:0,x2:0,xChannelSelector:"xChannelSelector",xlinkActuate:"xlink:actuate",xlinkArcrole:"xlink:arcrole",xlinkHref:"xlink:href",xlinkRole:"xlink:role",xlinkShow:"xlink:show",xlinkTitle:"xlink:title",xlinkType:"xlink:type",xmlBase:"xml:base",xmlns:0,xmlnsXlink:"xmlns:xlink",xmlLang:"xml:lang",xmlSpace:"xml:space",y:0,y1:0,y2:0,yChannelSelector:"yChannelSelector",z:0,zoomAndPan:"zoomAndPan"},o={Properties:{},DOMAttributeNamespaces:{xlinkActuate:r.xlink,xlinkArcrole:r.xlink,xlinkHref:r.xlink,xlinkRole:r.xlink,xlinkShow:r.xlink,xlinkTitle:r.xlink,xlinkType:r.xlink,xmlBase:r.xml,xmlLang:r.xml,xmlSpace:r.xml},DOMAttributeNames:{}};Object.keys(i).forEach(function(t){o.Properties[t]=0,i[t]&&(o.DOMAttributeNames[t]=i[t])}),t.exports=o},function(t,e,n){"use strict";function r(t){if("selectionStart"in t&&c.hasSelectionCapabilities(t))return{start:t.selectionStart,end:t.selectionEnd};if(window.getSelection){var e=window.getSelection();return{anchorNode:e.anchorNode,anchorOffset:e.anchorOffset,focusNode:e.focusNode,focusOffset:e.focusOffset}}if(document.selection){var n=document.selection.createRange();return{parentElement:n.parentElement(),text:n.text,top:n.boundingTop,left:n.boundingLeft}}}function i(t,e){if(y||null==v||v!==l())return null;var n=r(v);if(!m||!p(m,n)){m=n;var i=s.getPooled(d.select,g,t,e);return i.type="select",i.target=v,o.accumulateTwoPhaseDispatches(i),i}return null}var o=n(23),a=n(6),u=n(4),c=n(166),s=n(14),l=n(155),f=n(175),p=n(81),h=a.canUseDOM&&"documentMode"in document&&document.documentMode<=11,d={select:{phasedRegistrationNames:{bubbled:"onSelect",captured:"onSelectCapture"},dependencies:["topBlur","topContextMenu","topFocus","topKeyDown","topKeyUp","topMouseDown","topMouseUp","topSelectionChange"]}},v=null,g=null,m=null,y=!1,_=!1,b={eventTypes:d,extractEvents:function(t,e,n,r){if(!_)return null;var o=e?u.getNodeFromInstance(e):window;switch(t){case"topFocus":(f(o)||"true"===o.contentEditable)&&(v=o,g=e,m=null);break;case"topBlur":v=null,g=null,m=null;break;case"topMouseDown":y=!0;break;case"topContextMenu":case"topMouseUp":return y=!1,i(n,r);case"topSelectionChange":if(h)break;case"topKeyDown":case"topKeyUp":return i(n,r)}return null},didPutListener:function(t,e,n){"onSelect"===e&&(_=!0)}};t.exports=b},function(t,e,n){"use strict";function r(t){return"."+t._rootNodeID}function i(t){return"button"===t||"input"===t||"select"===t||"textarea"===t}var o=n(1),a=n(153),u=n(23),c=n(4),s=n(389),l=n(390),f=n(14),p=n(393),h=n(395),d=n(54),v=n(392),g=n(396),m=n(397),y=n(25),_=n(398),b=n(11),x=n(92),w=(n(0),{}),C={};["abort","animationEnd","animationIteration","animationStart","blur","canPlay","canPlayThrough","click","contextMenu","copy","cut","doubleClick","drag","dragEnd","dragEnter","dragExit","dragLeave","dragOver","dragStart","drop","durationChange","emptied","encrypted","ended","error","focus","input","invalid","keyDown","keyPress","keyUp","load","loadedData","loadedMetadata","loadStart","mouseDown","mouseMove","mouseOut","mouseOver","mouseUp","paste","pause","play","playing","progress","rateChange","reset","scroll","seeked","seeking","stalled","submit","suspend","timeUpdate","touchCancel","touchEnd","touchMove","touchStart","transitionEnd","volumeChange","waiting","wheel"].forEach(function(t){var e=t[0].toUpperCase()+t.slice(1),n="on"+e,r="top"+e,i={phasedRegistrationNames:{bubbled:n,captured:n+"Capture"},dependencies:[r]};w[t]=i,C[r]=i});var k={},E={eventTypes:w,extractEvents:function(t,e,n,r){var i=C[t];if(!i)return null;var a;switch(t){case"topAbort":case"topCanPlay":case"topCanPlayThrough":case"topDurationChange":case"topEmptied":case"topEncrypted":case"topEnded":case"topError":case"topInput":case"topInvalid":case"topLoad":case"topLoadedData":case"topLoadedMetadata":case"topLoadStart":case"topPause":case"topPlay":case"topPlaying":case"topProgress":case"topRateChange":case"topReset":case"topSeeked":case"topSeeking":case"topStalled":case"topSubmit":case"topSuspend":case"topTimeUpdate":case"topVolumeChange":case"topWaiting":a=f;break;case"topKeyPress":if(0===x(n))return null;case"topKeyDown":case"topKeyUp":a=h;break;case"topBlur":case"topFocus":a=p;break;case"topClick":if(2===n.button)return null;case"topDoubleClick":case"topMouseDown":case"topMouseMove":case"topMouseUp":case"topMouseOut":case"topMouseOver":case"topContextMenu":a=d;break;case"topDrag":case"topDragEnd":case"topDragEnter":case"topDragExit":case"topDragLeave":case"topDragOver":case"topDragStart":case"topDrop":a=v;break;case"topTouchCancel":case"topTouchEnd":case"topTouchMove":case"topTouchStart":a=g;break;case"topAnimationEnd":case"topAnimationIteration":case"topAnimationStart":a=s;break;case"topTransitionEnd":a=m;break;case"topScroll":a=y;break;case"topWheel":a=_;break;case"topCopy":case"topCut":case"topPaste":a=l}a||o("86",t);var c=a.getPooled(i,e,n,r);return u.accumulateTwoPhaseDispatches(c),c},didPutListener:function(t,e,n){if("onClick"===e&&!i(t._tag)){var o=r(t),u=c.getNodeFromInstance(t);k[o]||(k[o]=a.listen(u,"click",b))}},willDeleteListener:function(t,e){if("onClick"===e&&!i(t._tag)){var n=r(t);k[n].remove(),delete k[n]}}};t.exports=E},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={animationName:null,elapsedTime:null,pseudoElement:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={data:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(54),o={dataTransfer:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o={relatedTarget:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={data:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(92),a=n(403),u=n(93),c={key:a,location:null,ctrlKey:null,shiftKey:null,altKey:null,metaKey:null,repeat:null,locale:null,getModifierState:u,charCode:function(t){return"keypress"===t.type?o(t):0},keyCode:function(t){return"keydown"===t.type||"keyup"===t.type?t.keyCode:0},which:function(t){return"keypress"===t.type?o(t):"keydown"===t.type||"keyup"===t.type?t.keyCode:0}};i.augmentClass(r,c),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(93),a={touches:null,targetTouches:null,changedTouches:null,altKey:null,metaKey:null,ctrlKey:null,shiftKey:null,getModifierState:o};i.augmentClass(r,a),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o={propertyName:null,elapsedTime:null,pseudoElement:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(54),o={deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:null,deltaMode:null};i.augmentClass(r,o),t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=1,n=0,r=0,o=t.length,a=-4&o;r<a;){for(var u=Math.min(r+4096,a);r<u;r+=4)n+=(e+=t.charCodeAt(r))+(e+=t.charCodeAt(r+1))+(e+=t.charCodeAt(r+2))+(e+=t.charCodeAt(r+3));e%=i,n%=i}for(;r<o;r++)n+=e+=t.charCodeAt(r);return e%=i,n%=i,e|n<<16}var i=65521;t.exports=r},function(t,e,n){"use strict";function r(t,e,n,r){if(null==e||"boolean"==typeof e||""===e)return"";var i=isNaN(e);if(r||i||0===e||o.hasOwnProperty(t)&&o[t])return""+e;if("string"==typeof e){e=e.trim()}return e+"px"}var i=n(158),o=(n(2),i.isUnitlessNumber);t.exports=r},function(t,e,n){"use strict";function r(t){if(null==t)return null;if(1===t.nodeType)return t;var e=a.get(t);if(e)return e=u(e),e?o.getNodeFromInstance(e):null;"function"==typeof t.render?i("44"):i("45",Object.keys(t))}var i=n(1),o=(n(15),n(4)),a=n(39),u=n(171);n(0),n(2);t.exports=r},function(t,e,n){"use strict";(function(e){function r(t,e,n,r){if(t&&"object"==typeof t){var i=t,o=void 0===i[n];o&&null!=e&&(i[n]=e)}}function i(t,e){if(null==t)return t;var n={};return o(t,r,n),n}var o=(n(85),n(177));n(2);void 0!==e&&e.env,t.exports=i}).call(e,n(156))},function(t,e,n){"use strict";function r(t){if(t.key){var e=o[t.key]||t.key;if("Unidentified"!==e)return e}if("keypress"===t.type){var n=i(t);return 13===n?"Enter":String.fromCharCode(n)}return"keydown"===t.type||"keyup"===t.type?a[t.keyCode]||"Unidentified":""}var i=n(92),o={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},a={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"};t.exports=r},function(t,e,n){"use strict";function r(t){var e=t&&(i&&t[i]||t[o]);if("function"==typeof e)return e}var i="function"==typeof Symbol&&Symbol.iterator,o="@@iterator";t.exports=r},function(t,e,n){"use strict";function r(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function i(t){for(;t;){if(t.nextSibling)return t.nextSibling;t=t.parentNode}}function o(t,e){for(var n=r(t),o=0,a=0;n;){if(3===n.nodeType){if(a=o+n.textContent.length,o<=e&&a>=e)return{node:n,offset:e-o};o=a}n=r(i(n))}}t.exports=o},function(t,e,n){"use strict";function r(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n["ms"+t]="MS"+e,n["O"+t]="o"+e.toLowerCase(),n}function i(t){if(u[t])return u[t];if(!a[t])return t;var e=a[t];for(var n in e)if(e.hasOwnProperty(n)&&n in c)return u[t]=e[n];return""}var o=n(6),a={animationend:r("Animation","AnimationEnd"),animationiteration:r("Animation","AnimationIteration"),animationstart:r("Animation","AnimationStart"),transitionend:r("Transition","TransitionEnd")},u={},c={};o.canUseDOM&&(c=document.createElement("div").style,"AnimationEvent"in window||(delete a.animationend.animation,delete a.animationiteration.animation,delete a.animationstart.animation),"TransitionEvent"in window||delete a.transitionend.transition),t.exports=i},function(t,e,n){"use strict";function r(t){return'"'+i(t)+'"'}var i=n(56);t.exports=r},function(t,e,n){"use strict";var r=n(167);t.exports=r.renderSubtreeIntoContainer},function(t,e,n){"use strict";function r(t,e){var n=l.extractSingleTouch(e);return n?n[t.page]:t.page in e?e[t.page]:e[t.client]+f[t.envScroll]}function i(t,e){var n=r(b.x,e),i=r(b.y,e);return Math.pow(Math.pow(n-t.x,2)+Math.pow(i-t.y,2),.5)}function o(t){return{tapMoveThreshold:g,ignoreMouseThreshold:m,eventTypes:C,extractEvents:function(e,n,o,a){if(!h(e)&&!d(e))return null;if(v(e))_=k();else if(t(_,k()))return null;var u=null,l=i(y,o);return d(e)&&l<g&&(u=s.getPooled(C.touchTap,n,o,a)),h(e)?(y.x=r(b.x,o),y.y=r(b.y,o)):d(e)&&(y.x=0,y.y=0),c.accumulateTwoPhaseDispatches(u),u}}}var a=n(353),u=n(52),c=n(23),s=n(25),l=n(410),f=n(90),p=n(340),h=(a.topLevelTypes,u.isStartish),d=u.isEndish,v=function(t){return["topTouchCancel","topTouchEnd","topTouchStart","topTouchMove"].indexOf(t)>=0},g=10,m=750,y={x:null,y:null},_=null,b={x:{page:"pageX",client:"clientX",envScroll:"currentPageScrollLeft"},y:{page:"pageY",client:"clientY",envScroll:"currentPageScrollTop"}},x=["topTouchStart","topTouchCancel","topTouchEnd","topTouchMove"],w=["topMouseDown","topMouseMove","topMouseUp"].concat(x),C={touchTap:{phasedRegistrationNames:{bubbled:p({onTouchTap:null}),captured:p({onTouchTapCapture:null})},dependencies:w}},k=function(){return Date.now?Date.now:function(){return+new Date}}();t.exports=o},function(t,e){var n={extractSingleTouch:function(t){var e=t.touches,n=t.changedTouches,r=e&&e.length>0,i=n&&n.length>0;return!r&&i?n[0]:r?e[0]:t}};t.exports=n},function(t,e){t.exports=function(t,e){if(t&&e-t<750)return!0}},function(t,e,n){"use strict";function r(t){var e={"=":"=0",":":"=2"};return"$"+(""+t).replace(/[=:]/g,function(t){return e[t]})}function i(t){var e=/(=0|=2)/g,n={"=0":"=","=2":":"};return(""+("."===t[0]&&"$"===t[1]?t.substring(2):t.substring(1))).replace(e,function(t){return n[t]})}var o={escape:r,unescape:i};t.exports=o},function(t,e,n){"use strict";var r=n(40),i=(n(0),function(t){var e=this;if(e.instancePool.length){var n=e.instancePool.pop();return e.call(n,t),n}return new e(t)}),o=function(t,e){var n=this;if(n.instancePool.length){var r=n.instancePool.pop();return n.call(r,t,e),r}return new n(t,e)},a=function(t,e,n){var r=this;if(r.instancePool.length){var i=r.instancePool.pop();return r.call(i,t,e,n),i}return new r(t,e,n)},u=function(t,e,n,r){var i=this;if(i.instancePool.length){var o=i.instancePool.pop();return i.call(o,t,e,n,r),o}return new i(t,e,n,r)},c=function(t){var e=this;t instanceof e||r("25"),t.destructor(),e.instancePool.length<e.poolSize&&e.instancePool.push(t)},s=i,l=function(t,e){var n=t;return n.instancePool=[],n.getPooled=e||s,n.poolSize||(n.poolSize=10),n.release=c,n},f={addPoolingTo:l,oneArgumentPooler:i,twoArgumentPooler:o,threeArgumentPooler:a,fourArgumentPooler:u};t.exports=f},function(t,e,n){"use strict";function r(t){return(""+t).replace(b,"$&/")}function i(t,e){this.func=t,this.context=e,this.count=0}function o(t,e,n){var r=t.func,i=t.context;r.call(i,e,t.count++)}function a(t,e,n){if(null==t)return t;var r=i.getPooled(e,n);m(t,o,r),i.release(r)}function u(t,e,n,r){this.result=t,this.keyPrefix=e,this.func=n,this.context=r,this.count=0}function c(t,e,n){var i=t.result,o=t.keyPrefix,a=t.func,u=t.context,c=a.call(u,e,t.count++);Array.isArray(c)?s(c,i,n,g.thatReturnsArgument):null!=c&&(v.isValidElement(c)&&(c=v.cloneAndReplaceKey(c,o+(!c.key||e&&e.key===c.key?"":r(c.key)+"/")+n)),i.push(c))}function s(t,e,n,i,o){var a="";null!=n&&(a=r(n)+"/");var s=u.getPooled(e,a,i,o);m(t,c,s),u.release(s)}function l(t,e,n){if(null==t)return t;var r=[];return s(t,r,null,e,n),r}function f(t,e,n){return null}function p(t,e){return m(t,f,null)}function h(t){var e=[];return s(t,e,null,g.thatReturnsArgument),e}var d=n(413),v=n(27),g=n(11),m=n(423),y=d.twoArgumentPooler,_=d.fourArgumentPooler,b=/\/+/g;i.prototype.destructor=function(){this.func=null,this.context=null,this.count=0},d.addPoolingTo(i,y),u.prototype.destructor=function(){this.result=null,this.keyPrefix=null,this.func=null,this.context=null,this.count=0},d.addPoolingTo(u,_);var x={forEach:a,map:l,mapIntoWithKeyPrefixInternal:s,count:p,toArray:h};t.exports=x},function(t,e,n){"use strict";var r=n(27),i=r.createFactory,o={a:i("a"),abbr:i("abbr"),address:i("address"),area:i("area"),article:i("article"),aside:i("aside"),audio:i("audio"),b:i("b"),base:i("base"),bdi:i("bdi"),bdo:i("bdo"),big:i("big"),blockquote:i("blockquote"),body:i("body"),br:i("br"),button:i("button"),canvas:i("canvas"),caption:i("caption"),cite:i("cite"),code:i("code"),col:i("col"),colgroup:i("colgroup"),data:i("data"),datalist:i("datalist"),dd:i("dd"),del:i("del"),details:i("details"),dfn:i("dfn"),dialog:i("dialog"),div:i("div"),dl:i("dl"),dt:i("dt"),em:i("em"),embed:i("embed"),fieldset:i("fieldset"),figcaption:i("figcaption"),figure:i("figure"),footer:i("footer"),form:i("form"),h1:i("h1"),h2:i("h2"),h3:i("h3"),h4:i("h4"),h5:i("h5"),h6:i("h6"),head:i("head"),header:i("header"),hgroup:i("hgroup"),hr:i("hr"),html:i("html"),i:i("i"),iframe:i("iframe"),img:i("img"),input:i("input"),ins:i("ins"),kbd:i("kbd"),keygen:i("keygen"),label:i("label"),legend:i("legend"),li:i("li"),link:i("link"),main:i("main"),map:i("map"),mark:i("mark"),menu:i("menu"),menuitem:i("menuitem"),meta:i("meta"),meter:i("meter"),nav:i("nav"),noscript:i("noscript"),object:i("object"),ol:i("ol"),optgroup:i("optgroup"),option:i("option"),output:i("output"),p:i("p"),param:i("param"),picture:i("picture"),pre:i("pre"),progress:i("progress"),q:i("q"),rp:i("rp"),rt:i("rt"),ruby:i("ruby"),s:i("s"),samp:i("samp"),script:i("script"),section:i("section"),select:i("select"),small:i("small"),source:i("source"),span:i("span"),strong:i("strong"),style:i("style"),sub:i("sub"),summary:i("summary"),sup:i("sup"),table:i("table"),tbody:i("tbody"),td:i("td"),textarea:i("textarea"),tfoot:i("tfoot"),th:i("th"),thead:i("thead"),time:i("time"),title:i("title"),tr:i("tr"),track:i("track"),u:i("u"),ul:i("ul"),var:i("var"),video:i("video"),wbr:i("wbr"),circle:i("circle"),clipPath:i("clipPath"),defs:i("defs"),ellipse:i("ellipse"),g:i("g"),image:i("image"),line:i("line"),linearGradient:i("linearGradient"),mask:i("mask"),path:i("path"),pattern:i("pattern"),polygon:i("polygon"),polyline:i("polyline"),radialGradient:i("radialGradient"),rect:i("rect"),stop:i("stop"),svg:i("svg"),text:i("text"),tspan:i("tspan")};t.exports=o},function(t,e,n){"use strict";var r=n(27),i=r.isValidElement,o=n(157);t.exports=o(i)},function(t,e,n){"use strict";t.exports="15.6.2"},function(t,e,n){"use strict";var r=n(178),i=r.Component,o=n(27),a=o.isValidElement,u=n(181),c=n(191);t.exports=c(i,a,u)},function(t,e,n){"use strict";function r(t){var e=t&&(i&&t[i]||t[o]);if("function"==typeof e)return e}var i="function"==typeof Symbol&&Symbol.iterator,o="@@iterator";t.exports=r},function(t,e,n){"use strict";function r(){return i++}var i=1;t.exports=r},function(t,e,n){"use strict";var r=function(){};t.exports=r},function(t,e,n){"use strict";function r(t){return o.isValidElement(t)||i("143"),t}var i=n(40),o=n(27);n(0);t.exports=r},function(t,e,n){"use strict";function r(t,e){return t&&"object"==typeof t&&null!=t.key?s.escape(t.key):e.toString(36)}function i(t,e,n,o){var p=typeof t;if("undefined"!==p&&"boolean"!==p||(t=null),null===t||"string"===p||"number"===p||"object"===p&&t.$$typeof===u)return n(o,t,""===e?l+r(t,0):e),1;var h,d,v=0,g=""===e?l:e+f;if(Array.isArray(t))for(var m=0;m<t.length;m++)h=t[m],d=g+r(h,m),v+=i(h,d,n,o);else{var y=c(t);if(y){var _,b=y.call(t);if(y!==t.entries)for(var x=0;!(_=b.next()).done;)h=_.value,d=g+r(h,x++),v+=i(h,d,n,o);else for(;!(_=b.next()).done;){var w=_.value;w&&(h=w[1],d=g+s.escape(w[0])+f+r(h,0),v+=i(h,d,n,o))}}else if("object"===p){var C="",k=String(t);a("31","[object Object]"===k?"object with keys {"+Object.keys(t).join(", ")+"}":k,C)}}return v}function o(t,e,n){return null==t?0:i(t,"",e,n)}var a=n(40),u=(n(15),n(180)),c=n(419),s=(n(0),n(412)),l=(n(2),"."),f=":";t.exports=o}]);</script>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>Using 712 background data samples could cause slower run times. Consider using shap.kmeans(data, K) to summarize the background as K weighted samples.
100%|██████████| 179/179 [00:43&lt;00:00,  4.40it/s]
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[52]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">

<div id='iEZ21XTKJB28Y4UIKTO0F'>
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div></div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceArrayVisualizer, {"featureNames": ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], "ordering_keys": null, "outNames": ["output value"], "explanations": [{"outValue": 0.8800785117291052, "features": {"0": {"effect": 0.0839644246727187, "value": 3.0}, "1": {"effect": 0.0195903793037578, "value": 35.0}, "2": {"effect": -0.02570180427170872, "value": 0.0}, "3": {"effect": -0.006833611432304557, "value": 0.0}, "4": {"effect": 0.019579521173067917, "value": 7.125}, "5": {"effect": 0.1778695307945483, "value": 1.0}}, "simIndex": 18.0}, {"outValue": 0.8395940576033127, "features": {"0": {"effect": 0.08901620029528126, "value": 3.0}, "1": {"effect": -0.02933956810917618, "value": 20.0}, "2": {"effect": -0.027473490030915974, "value": 0.0}, "3": {"effect": -0.007234587955735814, "value": 0.0}, "4": {"effect": 0.020575953324410815, "value": 7.05}, "5": {"effect": 0.18243947859042284, "value": 1.0}}, "simIndex": 66.0}, {"outValue": 0.8564832166267902, "features": {"0": {"effect": 0.08712894018509143, "value": 3.0}, "1": {"effect": -0.009045130113886898, "value": 26.0}, "2": {"effect": -0.026770513478858737, "value": 0.0}, "3": {"effect": -0.007077654271610279, "value": 0.0}, "4": {"effect": 0.019609417781727426, "value": 7.8958}, "5": {"effect": 0.18102808503530143, "value": 1.0}}, "simIndex": 38.0}, {"outValue": 0.1054854906869589, "features": {"0": {"effect": -0.16111399866991832, "value": 1.0}, "1": {"effect": 0.09198443153130183, "value": 58.0}, "2": {"effect": -0.033279999348341555, "value": 0.0}, "3": {"effect": -0.007264693186840199, "value": 0.0}, "4": {"effect": -0.08786748984546783, "value": 146.5208}, "5": {"effect": -0.30858283128280084, "value": 0.0}}, "simIndex": 159.0}, {"outValue": 0.12564987623972224, "features": {"0": {"effect": -0.16678659556190123, "value": 1.0}, "1": {"effect": 0.0200191150910392, "value": 35.0}, "2": {"effect": 0.027043463308896773, "value": 1.0}, "3": {"effect": -0.007601326300899866, "value": 0.0}, "4": {"effect": -0.041302456784187015, "value": 83.475}, "5": {"effect": -0.3173323950022514, "value": 0.0}}, "simIndex": 163.0}, {"outValue": 0.835963288922992, "features": {"0": {"effect": 0.08939161938848883, "value": 3.0}, "1": {"effect": -0.032848071810948606, "value": 19.0}, "2": {"effect": -0.027619401028633153, "value": 0.0}, "3": {"effect": -0.007267023121024552, "value": 0.0}, "4": {"effect": 0.020014343729025018, "value": 7.8958}, "5": {"effect": 0.18268175027705866, "value": 1.0}}, "simIndex": 70.0}, {"outValue": 0.8538393199885123, "features": {"0": {"effect": 0.08744324986490444, "value": 3.0}, "1": {"effect": -0.012360791070557226, "value": 25.0}, "2": {"effect": -0.02688356920050555, "value": 0.0}, "3": {"effect": -0.007103061986108862, "value": 0.0}, "4": {"effect": 0.01984394770104112, "value": 7.65}, "5": {"effect": 0.18128947319071254, "value": 1.0}}, "simIndex": 44.0}, {"outValue": 0.815167008598589, "features": {"0": {"effect": -0.025779038351512417, "value": 2.0}, "1": {"effect": 0.06518606780360613, "value": 47.0}, "2": {"effect": -0.028612648831841534, "value": 0.0}, "3": {"effect": -0.007479580812749739, "value": 0.0}, "4": {"effect": 0.015331415649481435, "value": 15.0}, "5": {"effect": 0.18491072165257938, "value": 1.0}}, "simIndex": 30.0}, {"outValue": 0.8723524082879734, "features": {"0": {"effect": 0.08507386822230589, "value": 3.0}, "1": {"effect": 0.010298820730325668, "value": 32.0}, "2": {"effect": -0.02606440386154321, "value": 0.0}, "3": {"effect": -0.00691715549851072, "value": 0.0}, "4": {"effect": 0.019273324703732074, "value": 7.8542}, "5": {"effect": 0.17907788250263795, "value": 1.0}}, "simIndex": 11.0}, {"outValue": 0.8450326554502261, "features": {"0": {"effect": 0.0884430196030141, "value": 3.0}, "1": {"effect": -0.022483821673378243, "value": 22.0}, "2": {"effect": -0.02725229906522579, "value": 0.0}, "3": {"effect": -0.007185607799393014, "value": 0.0}, "4": {"effect": 0.019843952269428124, "value": 7.8958}, "5": {"effect": 0.18205734062675513, "value": 1.0}}, "simIndex": 62.0}, {"outValue": 0.730578350591021, "features": {"0": {"effect": -0.03068958325528679, "value": 2.0}, "1": {"effect": -0.014850690495682141, "value": 25.0}, "2": {"effect": -0.031507457631376086, "value": 0.0}, "3": {"effect": -0.008106602439630115, "value": 0.0}, "4": {"effect": 0.017733529498796433, "value": 13.0}, "5": {"effect": 0.18638908342517396, "value": 1.0}}, "simIndex": 84.0}, {"outValue": 0.18681792019328836, "features": {"0": {"effect": -0.046382821849038997, "value": 2.0}, "1": {"effect": -0.09984805300181976, "value": 3.0}, "2": {"effect": 0.02925349938997361, "value": 1.0}, "3": {"effect": 0.031220520488753845, "value": 2.0}, "4": {"effect": -0.009738712515271392, "value": 41.5792}, "5": {"effect": -0.32929658380833476, "value": 0.0}}, "simIndex": 149.0}, {"outValue": 0.34503889384788256, "features": {"0": {"effect": -0.04765275146106447, "value": 2.0}, "1": {"effect": -0.020306612072240937, "value": 24.0}, "2": {"effect": 0.10545359651093024, "value": 2.0}, "3": {"effect": 0.012790259101279768, "value": 1.0}, "4": {"effect": 0.003303105325106137, "value": 27.0}, "5": {"effect": -0.320158775045154, "value": 0.0}}, "simIndex": 124.0}, {"outValue": 0.5479639714732647, "features": {"0": {"effect": -0.18053339413459582, "value": 1.0}, "1": {"effect": -0.017532237729399047, "value": 25.0}, "2": {"effect": -0.03596718477032043, "value": 0.0}, "3": {"effect": -0.008859114033030968, "value": 0.0}, "4": {"effect": 0.0060535989793606415, "value": 26.0}, "5": {"effect": 0.17319223167222453, "value": 1.0}}, "simIndex": 111.0}, {"outValue": 0.750274963975193, "features": {"0": {"effect": -0.029535860143429064, "value": 2.0}, "1": {"effect": 0.0007076239275860922, "value": 29.0}, "2": {"effect": -0.030888241513404947, "value": 0.0}, "3": {"effect": -0.007977813831424561, "value": 0.0}, "4": {"effect": 0.019693593002735765, "value": 10.5}, "5": {"effect": 0.1866655910441039, "value": 1.0}}, "simIndex": 79.0}, {"outValue": 0.7429792675079574, "features": {"0": {"effect": -0.03000973058923976, "value": 2.0}, "1": {"effect": -0.014637095984425585, "value": 25.0}, "2": {"effect": -0.031114929628936923, "value": 0.0}, "3": {"effect": -0.008023373518472668, "value": 0.0}, "4": {"effect": 0.028657242056952595, "value": 0.0}, "5": {"effect": 0.18649708368305395, "value": 1.0}}, "simIndex": 82.0}, {"outValue": 0.9533933344749991, "features": {"0": {"effect": 0.06871555768616225, "value": 3.0}, "1": {"effect": -0.009636270317555318, "value": 25.0}, "2": {"effect": 0.11651322999686728, "value": 3.0}, "3": {"effect": 0.007871511573635206, "value": 1.0}, "4": {"effect": 0.006366706442608061, "value": 25.4667}, "5": {"effect": 0.15195252760425582, "value": 1.0}}, "simIndex": 90.0}, {"outValue": 0.9494526519730098, "features": {"0": {"effect": 0.06973653988585637, "value": 3.0}, "1": {"effect": -0.06360558770842115, "value": 4.0}, "2": {"effect": 0.16855828452796967, "value": 4.0}, "3": {"effect": 0.008174458057135375, "value": 1.0}, "4": {"effect": 0.00410283743633727, "value": 29.125}, "5": {"effect": 0.15087604828510642, "value": 1.0}}, "simIndex": 88.0}, {"outValue": 0.42273488986729124, "features": {"0": {"effect": 0.09069247343937106, "value": 3.0}, "1": {"effect": -0.016097435669747966, "value": 25.0}, "2": {"effect": 0.03639110238448166, "value": 1.0}, "3": {"effect": -0.00969662625949938, "value": 0.0}, "4": {"effect": 0.02125443933585555, "value": 7.925}, "5": {"effect": -0.31141913485219547, "value": 0.0}}, "simIndex": 131.0}, {"outValue": 0.8541000485908506, "features": {"0": {"effect": 0.08741055306312039, "value": 3.0}, "1": {"effect": -0.012355154544311905, "value": 25.0}, "2": {"effect": -0.02687227165857214, "value": 0.0}, "3": {"effect": -0.007100458621038301, "value": 0.0}, "4": {"effect": 0.020146645386774742, "value": 7.225}, "5": {"effect": 0.18126066347585207, "value": 1.0}}, "simIndex": 40.0}, {"outValue": 0.8937679298601189, "features": {"0": {"effect": 0.08185480770361979, "value": 3.0}, "1": {"effect": -0.011460916152993474, "value": 25.0}, "2": {"effect": 0.029671788108965216, "value": 1.0}, "3": {"effect": -0.006665831670869829, "value": 0.0}, "4": {"effect": 0.013152799518914605, "value": 16.1}, "5": {"effect": 0.1756052108634568, "value": 1.0}}, "simIndex": 2.0}, {"outValue": 0.7766559330825517, "features": {"0": {"effect": -0.027985755374162306, "value": 2.0}, "1": {"effect": 0.02683993977125845, "value": 36.0}, "2": {"effect": -0.030010451690709483, "value": 0.0}, "3": {"effect": -0.007790414500249388, "value": 0.0}, "4": {"effect": 0.01745709840567431, "value": 12.875}, "5": {"effect": 0.18653544498171426, "value": 1.0}}, "simIndex": 77.0}, {"outValue": 0.2903832359959515, "features": {"0": {"effect": -0.047942021194014875, "value": 2.0}, "1": {"effect": 0.027983518184168832, "value": 36.0}, "2": {"effect": 0.03302644688588208, "value": 1.0}, "3": {"effect": -0.009175323505501223, "value": 0.0}, "4": {"effect": 0.0037476836830878726, "value": 26.0}, "5": {"effect": -0.328867139546697, "value": 0.0}}, "simIndex": 150.0}, {"outValue": 0.8975941638930389, "features": {"0": {"effect": 0.08119840973317015, "value": 3.0}, "1": {"effect": -0.011359522471303632, "value": 25.0}, "2": {"effect": 0.029371209179888738, "value": 1.0}, "3": {"effect": -0.006618251814044407, "value": 0.0}, "4": {"effect": 0.018609413067939756, "value": 7.775}, "5": {"effect": 0.17478283470836253, "value": 1.0}}, "simIndex": 4.0}, {"outValue": 0.8013048604610913, "features": {"0": {"effect": 0.09245429993458153, "value": 3.0}, "1": {"effect": -0.06237982197508147, "value": 11.0}, "2": {"effect": -0.028928079833817066, "value": 0.0}, "3": {"effect": -0.007553960770469665, "value": 0.0}, "4": {"effect": 0.012036928201886676, "value": 18.7875}, "5": {"effect": 0.18406542341496548, "value": 1.0}}, "simIndex": 35.0}, {"outValue": 0.8377330994649721, "features": {"0": {"effect": 0.08922452635003943, "value": 3.0}, "1": {"effect": -0.029427587044461925, "value": 20.0}, "2": {"effect": -0.027549652155753397, "value": 0.0}, "3": {"effect": -0.007251985462510123, "value": 0.0}, "4": {"effect": 0.018530003085298747, "value": 9.8458}, "5": {"effect": 0.18259772320333356, "value": 1.0}}, "simIndex": 65.0}, {"outValue": 0.8390134979287406, "features": {"0": {"effect": 0.08908151449095529, "value": 3.0}, "1": {"effect": -0.029367096661847926, "value": 20.0}, "2": {"effect": -0.02749730263185154, "value": 0.0}, "3": {"effect": -0.007240031183947598, "value": 0.0}, "4": {"effect": 0.01993677007023695, "value": 7.925}, "5": {"effect": 0.1824895723561696, "value": 1.0}}, "simIndex": 67.0}, {"outValue": 0.4225451818212014, "features": {"0": {"effect": -0.19131063917582022, "value": 1.0}, "1": {"effect": -0.11122653092079907, "value": 4.0}, "2": {"effect": -0.037592907444982726, "value": 0.0}, "3": {"effect": 0.03797051248336919, "value": 2.0}, "4": {"effect": -0.04842055436463927, "value": 81.8583}, "5": {"effect": 0.16151522975504767, "value": 1.0}}, "simIndex": 121.0}, {"outValue": 0.8325901949188911, "features": {"0": {"effect": -0.02479883663980458, "value": 2.0}, "1": {"effect": 0.08128940468620897, "value": 52.0}, "2": {"effect": -0.02792838131816889, "value": 0.0}, "3": {"effect": -0.007322761615660513, "value": 0.0}, "4": {"effect": 0.01624093127682721, "value": 13.5}, "5": {"effect": 0.18349976704046306, "value": 1.0}}, "simIndex": 31.0}, {"outValue": 0.14035566801101262, "features": {"0": {"effect": -0.16960759139551607, "value": 1.0}, "1": {"effect": 0.030677413595421144, "value": 38.0}, "2": {"effect": 0.02768174433200936, "value": 1.0}, "3": {"effect": -0.007772895969861526, "value": 0.0}, "4": {"effect": -0.03242039074291975, "value": 71.2833}, "5": {"effect": -0.31981268329714635, "value": 0.0}}, "simIndex": 165.0}, {"outValue": 0.14366927506361737, "features": {"0": {"effect": -0.17031486053192235, "value": 1.0}, "1": {"effect": 0.020512907797814416, "value": 35.0}, "2": {"effect": 0.027821831380962087, "value": 1.0}, "3": {"effect": -0.007813762620229231, "value": 0.0}, "4": {"effect": -0.017486218548440874, "value": 52.0}, "5": {"effect": -0.3206606939035925, "value": 0.0}}, "simIndex": 164.0}, {"outValue": 0.7429792675079574, "features": {"0": {"effect": -0.03000973058923976, "value": 2.0}, "1": {"effect": -0.014637095984425585, "value": 25.0}, "2": {"effect": -0.031114929628936923, "value": 0.0}, "3": {"effect": -0.008023373518472668, "value": 0.0}, "4": {"effect": 0.028657242056952595, "value": 0.0}, "5": {"effect": 0.18649708368305395, "value": 1.0}}, "simIndex": 83.0}, {"outValue": 0.2906179390342335, "features": {"0": {"effect": 0.08291597378980961, "value": 3.0}, "1": {"effect": -0.07762487802901909, "value": 10.0}, "2": {"effect": -0.039111489347426025, "value": 0.0}, "3": {"effect": 0.03390730445288536, "value": 2.0}, "4": {"effect": 0.0052690994220526925, "value": 24.15}, "5": {"effect": -0.32634814274309487, "value": 0.0}}, "simIndex": 145.0}, {"outValue": 0.9451662442494341, "features": {"0": {"effect": 0.07098084863180307, "value": 3.0}, "1": {"effect": 0.02138820008166374, "value": 37.0}, "2": {"effect": 0.07247735787030771, "value": 2.0}, "3": {"effect": -0.005884967712071032, "value": 0.0}, "4": {"effect": 0.016619718444689444, "value": 7.925}, "5": {"effect": 0.15797501544401532, "value": 1.0}}, "simIndex": 91.0}, {"outValue": 0.28009781325545036, "features": {"0": {"effect": 0.0823115659389071, "value": 3.0}, "1": {"effect": -0.05663974722659223, "value": 15.0}, "2": {"effect": -0.03901774413158442, "value": 0.0}, "3": {"effect": -0.009090165943301209, "value": 0.0}, "4": {"effect": 0.019199832077888487, "value": 8.0292}, "5": {"effect": -0.32827599894889314, "value": 0.0}}, "simIndex": 147.0}, {"outValue": 0.8537625616476027, "features": {"0": {"effect": 0.08745286312993071, "value": 3.0}, "1": {"effect": -0.012362449271134456, "value": 25.0}, "2": {"effect": -0.02688689304850888, "value": 0.0}, "3": {"effect": -0.007103827777745146, "value": 0.0}, "4": {"effect": 0.019754872853028176, "value": 7.775}, "5": {"effect": 0.18129792427300648, "value": 1.0}}, "simIndex": 45.0}, {"outValue": 0.3219084516167995, "features": {"0": {"effect": 0.0851218711452856, "value": 3.0}, "1": {"effect": -0.020228156558564525, "value": 24.0}, "2": {"effect": -0.03937336946065881, "value": 0.0}, "3": {"effect": -0.00932007633075644, "value": 0.0}, "4": {"effect": 0.019145122766551996, "value": 8.85}, "5": {"effect": -0.32504701143408415, "value": 0.0}}, "simIndex": 137.0}, {"outValue": 0.5875735523704717, "features": {"0": {"effect": -0.16929598489896638, "value": 1.0}, "1": {"effect": -0.040532855657088696, "value": 19.0}, "2": {"effect": 0.1848495294417693, "value": 3.0}, "3": {"effect": 0.03545487531700098, "value": 2.0}, "4": {"effect": -0.20280635287995363, "value": 263.0}, "5": {"effect": 0.16829426955868435, "value": 1.0}}, "simIndex": 120.0}, {"outValue": 0.8537779159877965, "features": {"0": {"effect": 0.08745094060247136, "value": 3.0}, "1": {"effect": -0.012362117616543093, "value": 25.0}, "2": {"effect": -0.026886228241055826, "value": 0.0}, "3": {"effect": -0.007103674615936001, "value": 0.0}, "4": {"effect": 0.019772689499711355, "value": 7.75}, "5": {"effect": 0.18129623487012292, "value": 1.0}}, "simIndex": 49.0}, {"outValue": 0.8723136192920607, "features": {"0": {"effect": 0.08507944903337367, "value": 3.0}, "1": {"effect": 0.010299647841057052, "value": 32.0}, "2": {"effect": -0.026066220611340597, "value": 0.0}, "3": {"effect": -0.006917578183552826, "value": 0.0}, "4": {"effect": 0.019224622347365616, "value": 7.925}, "5": {"effect": 0.17908362737613204, "value": 1.0}}, "simIndex": 12.0}, {"outValue": 0.9108958211766538, "features": {"0": {"effect": 0.07879360148789617, "value": 3.0}, "1": {"effect": 0.034360485837528376, "value": 40.5}, "2": {"effect": -0.0240853205916619, "value": 0.0}, "3": {"effect": 0.024972616101114245, "value": 2.0}, "4": {"effect": 0.01388562242608811, "value": 14.5}, "5": {"effect": 0.171358744426663, "value": 1.0}}, "simIndex": 20.0}, {"outValue": 0.8314444539410389, "features": {"0": {"effect": -0.024735063391632943, "value": 2.0}, "1": {"effect": 0.007650412707735338, "value": 31.0}, "2": {"effect": 0.03403511450813545, "value": 1.0}, "3": {"effect": 0.011220863714495537, "value": 1.0}, "4": {"effect": 0.006635907003258942, "value": 26.25}, "5": {"effect": 0.18502714791002084, "value": 1.0}}, "simIndex": 95.0}, {"outValue": 0.8536883506486206, "features": {"0": {"effect": 0.08746215189680547, "value": 3.0}, "1": {"effect": -0.012364051928016654, "value": 25.0}, "2": {"effect": -0.026890105664753212, "value": 0.0}, "3": {"effect": -0.007104567880135401, "value": 0.0}, "4": {"effect": 0.019668771000840433, "value": 7.8958}, "5": {"effect": 0.18130608173485416, "value": 1.0}}, "simIndex": 54.0}, {"outValue": 0.7252654505227139, "features": {"0": {"effect": 0.09346602093119499, "value": 3.0}, "1": {"effect": -0.044055836826103506, "value": 17.0}, "2": {"effect": 0.25444054168655633, "value": 4.0}, "3": {"effect": 0.033043296849889076, "value": 2.0}, "4": {"effect": 0.021356703817060035, "value": 7.925}, "5": {"effect": -0.24459534742490885, "value": 0.0}}, "simIndex": 126.0}, {"outValue": 0.7290229723679198, "features": {"0": {"effect": -0.1547052127022691, "value": 1.0}, "1": {"effect": 0.07523036491071411, "value": 48.0}, "2": {"effect": 0.0383302595242672, "value": 1.0}, "3": {"effect": -0.008077953339136551, "value": 0.0}, "4": {"effect": -0.015293127564568063, "value": 52.0}, "5": {"effect": 0.1819285700498864, "value": 1.0}}, "simIndex": 118.0}, {"outValue": 0.5653716074661777, "features": {"0": {"effect": -0.17844415057890403, "value": 1.0}, "1": {"effect": -0.00019326631209277967, "value": 29.0}, "2": {"effect": -0.03564025393333786, "value": 0.0}, "3": {"effect": -0.008814232438075859, "value": 0.0}, "4": {"effect": 0.0023940393712684055, "value": 30.0}, "5": {"effect": 0.174459399868294, "value": 1.0}}, "simIndex": 107.0}, {"outValue": 0.8421349412449591, "features": {"0": {"effect": 0.08875399500960432, "value": 3.0}, "1": {"effect": -0.025908061614724573, "value": 21.0}, "2": {"effect": -0.027370862231268604, "value": 0.0}, "3": {"effect": -0.007211960211386849, "value": 0.0}, "4": {"effect": 0.01998917086722237, "value": 7.775}, "5": {"effect": 0.18227258793648668, "value": 1.0}}, "simIndex": 60.0}, {"outValue": 0.8420368154996714, "features": {"0": {"effect": 0.08876526290951364, "value": 3.0}, "1": {"effect": -0.025912233119364345, "value": 21.0}, "2": {"effect": -0.027374927581832992, "value": 0.0}, "3": {"effect": -0.007212890823900239, "value": 0.0}, "4": {"effect": 0.019880071952048627, "value": 7.925}, "5": {"effect": 0.18228146067418086, "value": 1.0}}, "simIndex": 61.0}, {"outValue": 0.0739440045089681, "features": {"0": {"effect": -0.15307445630618954, "value": 1.0}, "1": {"effect": -0.04170521153511318, "value": 16.0}, "2": {"effect": -0.031905328291484514, "value": 0.0}, "3": {"effect": 0.010016675997371965, "value": 1.0}, "4": {"effect": -0.020298026964861368, "value": 57.9792}, "5": {"effect": -0.30069971987978106, "value": 0.0}}, "simIndex": 178.0}, {"outValue": 0.710831228795821, "features": {"0": {"effect": -0.15765887884548951, "value": 1.0}, "1": {"effect": 0.0681882022715678, "value": 46.0}, "2": {"effect": 0.03886263478033586, "value": 1.0}, "3": {"effect": -0.008189723409960978, "value": 0.0}, "4": {"effect": -0.023551072873590397, "value": 61.175}, "5": {"effect": 0.18156999538393237, "value": 1.0}}, "simIndex": 117.0}, {"outValue": 0.7028009816895983, "features": {"0": {"effect": -0.15909863582477948, "value": 1.0}, "1": {"effect": 0.03237585369023323, "value": 37.0}, "2": {"effect": 0.0391541962429946, "value": 1.0}, "3": {"effect": 0.013239680503380119, "value": 1.0}, "4": {"effect": -0.016347585492930295, "value": 52.5542}, "5": {"effect": 0.18186740108167426, "value": 1.0}}, "simIndex": 116.0}, {"outValue": 0.8479332934561021, "features": {"0": {"effect": -0.02376171354708569, "value": 2.0}, "1": {"effect": 0.04478829888144195, "value": 42.0}, "2": {"effect": 0.0330539011490123, "value": 1.0}, "3": {"effect": -0.007181174402855171, "value": 0.0}, "4": {"effect": 0.00607845594636388, "value": 27.0}, "5": {"effect": 0.18334545394019908, "value": 1.0}}, "simIndex": 32.0}, {"outValue": 0.8796777369867343, "features": {"0": {"effect": 0.08402551195069763, "value": 3.0}, "1": {"effect": 0.019607429160576416, "value": 35.0}, "2": {"effect": -0.02572125648644867, "value": 0.0}, "3": {"effect": -0.006838156174392193, "value": 0.0}, "4": {"effect": 0.019057732193083973, "value": 7.8958}, "5": {"effect": 0.17793640485419138, "value": 1.0}}, "simIndex": 19.0}, {"outValue": 0.7286883217117388, "features": {"0": {"effect": -0.1547329198158682, "value": 1.0}, "1": {"effect": 0.07920604665981593, "value": 49.0}, "2": {"effect": 0.03832379909271284, "value": 1.0}, "3": {"effect": -0.008077944139576819, "value": 0.0}, "4": {"effect": -0.019461036882904875, "value": 56.9292}, "5": {"effect": 0.18182030530853413, "value": 1.0}}, "simIndex": 119.0}, {"outValue": 0.731302638817644, "features": {"0": {"effect": -0.030979583683954252, "value": 2.0}, "1": {"effect": -0.09829562147376879, "value": 3.0}, "2": {"effect": 0.03827147810239982, "value": 1.0}, "3": {"effect": 0.01284250093263073, "value": 1.0}, "4": {"effect": 0.012688502861915212, "value": 18.75}, "5": {"effect": 0.1851652905893955, "value": 1.0}}, "simIndex": 73.0}, {"outValue": 0.07791230323077725, "features": {"0": {"effect": -0.1544431452367952, "value": 1.0}, "1": {"effect": -0.03874816430254786, "value": 17.0}, "2": {"effect": 0.02463334442073109, "value": 1.0}, "3": {"effect": -0.006901579077760012, "value": 0.0}, "4": {"effect": -0.05663110607355373, "value": 108.9}, "5": {"effect": -0.30160711798832285, "value": 0.0}}, "simIndex": 172.0}, {"outValue": 0.9472432521910663, "features": {"0": {"effect": 0.07030103083496027, "value": 3.0}, "1": {"effect": -0.06928003551651378, "value": 2.0}, "2": {"effect": 0.17044010475178092, "value": 4.0}, "3": {"effect": 0.008262618514368006, "value": 1.0}, "4": {"effect": 0.0040931604420330475, "value": 29.125}, "5": {"effect": 0.15181630167541205, "value": 1.0}}, "simIndex": 89.0}, {"outValue": 0.3286083043946708, "features": {"0": {"effect": 0.08554155522419378, "value": 3.0}, "1": {"effect": -0.01612409492338135, "value": 25.0}, "2": {"effect": -0.03940541656831726, "value": 0.0}, "3": {"effect": -0.00934985318841973, "value": 0.0}, "4": {"effect": 0.02068379978528276, "value": 7.2292}, "5": {"effect": -0.3243477574237132, "value": 0.0}}, "simIndex": 139.0}, {"outValue": 0.06337720628905652, "features": {"0": {"effect": -0.14782018953281384, "value": 1.0}, "1": {"effect": 0.061621236042087316, "value": 50.0}, "2": {"effect": -0.03076391968329295, "value": 0.0}, "3": {"effect": 0.00970341351306614, "value": 1.0}, "4": {"effect": -0.15397962265173754, "value": 247.5208}, "5": {"effect": -0.2869937828872784, "value": 0.0}}, "simIndex": 160.0}, {"outValue": 0.9223000553521778, "features": {"0": {"effect": 0.0764995236795667, "value": 3.0}, "1": {"effect": -0.013496759343252791, "value": 24.0}, "2": {"effect": 0.07920158635807584, "value": 2.0}, "3": {"effect": -0.006266690653428719, "value": 0.0}, "4": {"effect": 0.007536418609370273, "value": 24.15}, "5": {"effect": 0.16721590521282076, "value": 1.0}}, "simIndex": 92.0}, {"outValue": 0.8537830133333182, "features": {"0": {"effect": 0.08745030230946657, "value": 3.0}, "1": {"effect": -0.012362007508818607, "value": 25.0}, "2": {"effect": -0.026886007529165472, "value": 0.0}, "3": {"effect": -0.00710362376659987, "value": 0.0}, "4": {"effect": 0.019778604440996727, "value": 7.7417}, "5": {"effect": 0.18129567389841306, "value": 1.0}}, "simIndex": 47.0}, {"outValue": 0.16924049744580405, "features": {"0": {"effect": -0.17380196724503696, "value": 1.0}, "1": {"effect": 0.06977951265896315, "value": 49.0}, "2": {"effect": 0.028832600719812314, "value": 1.0}, "3": {"effect": -0.008053355265509432, "value": 0.0}, "4": {"effect": -0.0375978398652608, "value": 76.7292}, "5": {"effect": -0.32152852504619, "value": 0.0}}, "simIndex": 167.0}, {"outValue": 0.32804330374971824, "features": {"0": {"effect": 0.08550761388456408, "value": 3.0}, "1": {"effect": -0.016122639415488478, "value": 25.0}, "2": {"effect": -0.03940314867986555, "value": 0.0}, "3": {"effect": -0.009347588064389494, "value": 0.0}, "4": {"effect": 0.020212149962918667, "value": 7.75}, "5": {"effect": -0.3244131554270468, "value": 0.0}}, "simIndex": 143.0}, {"outValue": 0.41843109992139116, "features": {"0": {"effect": 0.09050990382255017, "value": 3.0}, "1": {"effect": -0.011854366448847423, "value": 26.0}, "2": {"effect": 0.0363076385995042, "value": 1.0}, "3": {"effect": -0.009692128058938376, "value": 0.0}, "4": {"effect": 0.01366531606397562, "value": 16.1}, "5": {"effect": -0.31211533554587884, "value": 0.0}}, "simIndex": 132.0}, {"outValue": 0.9463250422515002, "features": {"0": {"effect": 0.07050694778847602, "value": 3.0}, "1": {"effect": 0.11958258550834172, "value": 74.0}, "2": {"effect": -0.021775640083114445, "value": 0.0}, "3": {"effect": -0.005869182863491917, "value": 0.0}, "4": {"effect": 0.016620561592122697, "value": 7.775}, "5": {"effect": 0.1556496988201403, "value": 1.0}}, "simIndex": 26.0}, {"outValue": 0.8325698045916746, "features": {"0": {"effect": 0.08972938948753685, "value": 3.0}, "1": {"effect": -0.036372764390602136, "value": 18.0}, "2": {"effect": -0.027753906237223343, "value": 0.0}, "3": {"effect": -0.0072967539290324235, "value": 0.0}, "4": {"effect": 0.01977140111119464, "value": 8.3}, "5": {"effect": 0.18288236706077524, "value": 1.0}}, "simIndex": 68.0}, {"outValue": 0.8056072365301984, "features": {"0": {"effect": -0.026288416811382543, "value": 2.0}, "1": {"effect": 0.0007459158409039322, "value": 29.0}, "2": {"effect": 0.03539018567907774, "value": 1.0}, "3": {"effect": -0.007561496943690393, "value": 0.0}, "4": {"effect": 0.0054531225115996, "value": 27.7208}, "5": {"effect": 0.18625785476466425, "value": 1.0}}, "simIndex": 96.0}, {"outValue": 0.8684272372606643, "features": {"0": {"effect": 0.08560951892645398, "value": 3.0}, "1": {"effect": 0.005555441947668785, "value": 30.5}, "2": {"effect": -0.02624366972908529, "value": 0.0}, "3": {"effect": -0.00695818143633764, "value": 0.0}, "4": {"effect": 0.0192302370386783, "value": 8.05}, "5": {"effect": 0.17962381902426036, "value": 1.0}}, "simIndex": 9.0}, {"outValue": 0.8771881414772206, "features": {"0": {"effect": 0.08438970868688028, "value": 3.0}, "1": {"effect": 0.01653443586995744, "value": 34.0}, "2": {"effect": -0.02583928204128852, "value": 0.0}, "3": {"effect": -0.00686541633409464, "value": 0.0}, "4": {"effect": 0.019015337182846467, "value": 8.05}, "5": {"effect": 0.17834328662389376, "value": 1.0}}, "simIndex": 13.0}, {"outValue": 0.4522574938518872, "features": {"0": {"effect": -0.045019361205739405, "value": 2.0}, "1": {"effect": 0.005014033036266602, "value": 30.0}, "2": {"effect": 0.18168142311606175, "value": 3.0}, "3": {"effect": -0.009593123230028373, "value": 0.0}, "4": {"effect": 0.009616110223522142, "value": 21.0}, "5": {"effect": -0.30105165957722135, "value": 0.0}}, "simIndex": 125.0}, {"outValue": 0.6749384830577183, "features": {"0": {"effect": -0.03390572077410672, "value": 2.0}, "1": {"effect": -0.05201465909960913, "value": 16.0}, "2": {"effect": -0.033103329187424646, "value": 0.0}, "3": {"effect": -0.008422991899742563, "value": 0.0}, "4": {"effect": 0.006375761088668809, "value": 26.0}, "5": {"effect": 0.18439935144090677, "value": 1.0}}, "simIndex": 75.0}, {"outValue": 0.8955726179863565, "features": {"0": {"effect": 0.08150803862774951, "value": 3.0}, "1": {"effect": 0.04037757952829797, "value": 42.0}, "2": {"effect": -0.024931942052455226, "value": 0.0}, "3": {"effect": -0.00665376230602481, "value": 0.0}, "4": {"effect": 0.018838890513405933, "value": 7.55}, "5": {"effect": 0.17482374218635727, "value": 1.0}}, "simIndex": 22.0}, {"outValue": 0.858473649632139, "features": {"0": {"effect": 0.08685207211042877, "value": 3.0}, "1": {"effect": -0.012259662043634212, "value": 25.0}, "2": {"effect": -0.02668105719783475, "value": 0.0}, "3": {"effect": -0.007056281211718529, "value": 0.0}, "4": {"effect": 0.025255465057026277, "value": 0.0}, "5": {"effect": 0.18075304142884566, "value": 1.0}}, "simIndex": 56.0}, {"outValue": 0.9298235504929961, "features": {"0": {"effect": 0.07470329779967605, "value": 3.0}, "1": {"effect": 0.09026381226639102, "value": 61.0}, "2": {"effect": -0.02294533015591929, "value": 0.0}, "3": {"effect": -0.006170520165514545, "value": 0.0}, "4": {"effect": 0.01836692788033044, "value": 6.2375}, "5": {"effect": 0.16399529137900665, "value": 1.0}}, "simIndex": 28.0}, {"outValue": 0.2097976351300087, "features": {"0": {"effect": -0.04689946540106697, "value": 2.0}, "1": {"effect": 0.011105555158852427, "value": 32.0}, "2": {"effect": -0.03765475780339586, "value": 0.0}, "3": {"effect": -0.008565601921970187, "value": 0.0}, "4": {"effect": 0.013800427018408709, "value": 13.0}, "5": {"effect": -0.3335985934098452, "value": 0.0}}, "simIndex": 155.0}, {"outValue": 0.8255382345317239, "features": {"0": {"effect": 0.09039771159743987, "value": 3.0}, "1": {"effect": -0.04351726519360391, "value": 16.0}, "2": {"effect": -0.028027675909036244, "value": 0.0}, "3": {"effect": -0.007356998794800736, "value": 0.0}, "4": {"effect": 0.019193070675415584, "value": 9.2167}, "5": {"effect": 0.18323932066728357, "value": 1.0}}, "simIndex": 36.0}, {"outValue": 0.8479897989648677, "features": {"0": {"effect": 0.08811622115429121, "value": 3.0}, "1": {"effect": -0.019082538230285547, "value": 23.0}, "2": {"effect": -0.02712989088612937, "value": 0.0}, "3": {"effect": -0.007158291306018076, "value": 0.0}, "4": {"effect": 0.01981601514843943, "value": 7.8542}, "5": {"effect": 0.18181821159554426, "value": 1.0}}, "simIndex": 57.0}, {"outValue": 0.9535194807745839, "features": {"0": {"effect": 0.06860537285705817, "value": 3.0}, "1": {"effect": -0.06505891414985691, "value": 3.0}, "2": {"effect": 0.16510294576718854, "value": 4.0}, "3": {"effect": 0.021714240162019866, "value": 2.0}, "4": {"effect": 0.0028364000972267017, "value": 31.3875}, "5": {"effect": 0.14870936455192174, "value": 1.0}}, "simIndex": 86.0}, {"outValue": 0.5458662704067244, "features": {"0": {"effect": -0.1807732545447352, "value": 1.0}, "1": {"effect": -0.017554940313828413, "value": 25.0}, "2": {"effect": -0.03600619654942036, "value": 0.0}, "3": {"effect": -0.008864699251472957, "value": 0.0}, "4": {"effect": 0.004414746856956295, "value": 27.7208}, "5": {"effect": 0.17304054272019925, "value": 1.0}}, "simIndex": 112.0}, {"outValue": 0.8792427004813073, "features": {"0": {"effect": 0.08408615597765379, "value": 3.0}, "1": {"effect": 0.018043116129622658, "value": 34.5}, "2": {"effect": -0.025741346263317533, "value": 0.0}, "3": {"effect": -0.00684270649908892, "value": 0.0}, "4": {"effect": 0.02007812892070312, "value": 6.4375}, "5": {"effect": 0.17800928072670835, "value": 1.0}}, "simIndex": 17.0}, {"outValue": 0.9001341820320313, "features": {"0": {"effect": 0.0807835495653301, "value": 3.0}, "1": {"effect": -0.0023461807388777053, "value": 28.0}, "2": {"effect": 0.029173149693691003, "value": 1.0}, "3": {"effect": -0.00658889667008869, "value": 0.0}, "4": {"effect": 0.013212690237025593, "value": 15.85}, "5": {"effect": 0.17428979845592524, "value": 1.0}}, "simIndex": 5.0}, {"outValue": 0.8537779159877965, "features": {"0": {"effect": 0.08745094060247136, "value": 3.0}, "1": {"effect": -0.012362117616543093, "value": 25.0}, "2": {"effect": -0.026886228241055826, "value": 0.0}, "3": {"effect": -0.007103674615936001, "value": 0.0}, "4": {"effect": 0.019772689499711355, "value": 7.75}, "5": {"effect": 0.18129623487012292, "value": 1.0}}, "simIndex": 50.0}, {"outValue": 0.8057170996417236, "features": {"0": {"effect": -0.026566875732098785, "value": 2.0}, "1": {"effect": -0.027648385322189082, "value": 21.0}, "2": {"effect": 0.10142508266891295, "value": 2.0}, "3": {"effect": -0.00753796124583613, "value": 0.0}, "4": {"effect": -0.03015881344851886, "value": 73.5}, "5": {"effect": 0.18459398123242765, "value": 1.0}}, "simIndex": 98.0}, {"outValue": 0.42109132945589967, "features": {"0": {"effect": -0.18949655011845015, "value": 1.0}, "1": {"effect": -0.12376104841975166, "value": 0.92}, "2": {"effect": 0.03982560521902839, "value": 1.0}, "3": {"effect": 0.03754324463130451, "value": 2.0}, "4": {"effect": -0.11514018629773298, "value": 151.55}, "5": {"effect": 0.1605101929524757, "value": 1.0}}, "simIndex": 122.0}, {"outValue": 0.18144479333580465, "features": {"0": {"effect": -0.046208837768269534, "value": 2.0}, "1": {"effect": -0.01883588653560707, "value": 24.0}, "2": {"effect": -0.037012889273357844, "value": 0.0}, "3": {"effect": -0.008308802549678009, "value": 0.0}, "4": {"effect": 0.013272095551542545, "value": 13.0}, "5": {"effect": -0.3330709575778512, "value": 0.0}}, "simIndex": 154.0}, {"outValue": 0.7686240491486778, "features": {"0": {"effect": -0.028452835832282655, "value": 2.0}, "1": {"effect": 0.019517536927116375, "value": 34.0}, "2": {"effect": -0.03028443641623778, "value": 0.0}, "3": {"effect": -0.007849662088418068, "value": 0.0}, "4": {"effect": 0.017434517662437286, "value": 13.0}, "5": {"effect": 0.18664885740703682, "value": 1.0}}, "simIndex": 78.0}, {"outValue": 0.13438216893706523, "features": {"0": {"effect": -0.16824742839400372, "value": 1.0}, "1": {"effect": 0.03377562855425441, "value": 39.0}, "2": {"effect": 0.027399270856270447, "value": 1.0}, "3": {"effect": 0.010911962156966562, "value": 1.0}, "4": {"effect": -0.0632368791549496, "value": 110.8833}, "5": {"effect": -0.31783045657049863, "value": 0.0}}, "simIndex": 168.0}, {"outValue": 0.14949055053026938, "features": {"0": {"effect": -0.17071667972787552, "value": 1.0}, "1": {"effect": 0.07156429448632345, "value": 50.0}, "2": {"effect": -0.035128752190110496, "value": 0.0}, "3": {"effect": -0.007819605616911182, "value": 0.0}, "4": {"effect": 0.0005374536719563172, "value": 28.7125}, "5": {"effect": -0.320556231582139, "value": 0.0}}, "simIndex": 161.0}, {"outValue": 0.7521917856611051, "features": {"0": {"effect": -0.029414872624270134, "value": 2.0}, "1": {"effect": 0.004523035914163063, "value": 30.0}, "2": {"effect": -0.030827445365287115, "value": 0.0}, "3": {"effect": -0.007965321962086644, "value": 0.0}, "4": {"effect": 0.01757886097834501, "value": 13.0}, "5": {"effect": 0.18668745723121513, "value": 1.0}}, "simIndex": 80.0}, {"outValue": 0.19809425840941108, "features": {"0": {"effect": -0.04664108450537078, "value": 2.0}, "1": {"effect": -0.042068550305083725, "value": 18.0}, "2": {"effect": -0.03744511901521605, "value": 0.0}, "3": {"effect": 0.031622146709110455, "value": 2.0}, "4": {"effect": 0.013585155854819654, "value": 13.0}, "5": {"effect": -0.3325683618178743, "value": 0.0}}, "simIndex": 153.0}, {"outValue": 0.872245111331566, "features": {"0": {"effect": 0.08508930146962516, "value": 3.0}, "1": {"effect": 0.0103011082735027, "value": 32.0}, "2": {"effect": -0.026069428537827753, "value": 0.0}, "3": {"effect": -0.0069183244918799985, "value": 0.0}, "4": {"effect": 0.01913862040839882, "value": 8.05}, "5": {"effect": 0.17909376272072128, "value": 1.0}}, "simIndex": 10.0}, {"outValue": 0.853788171942705, "features": {"0": {"effect": 0.08744965631913923, "value": 3.0}, "1": {"effect": -0.012361896075309145, "value": 25.0}, "2": {"effect": -0.026885784160216804, "value": 0.0}, "3": {"effect": -0.007103572304814176, "value": 0.0}, "4": {"effect": 0.019784590552525953, "value": 7.7333}, "5": {"effect": 0.1812951061223541, "value": 1.0}}, "simIndex": 46.0}, {"outValue": 0.5369755807793959, "features": {"0": {"effect": -0.1817718271675076, "value": 1.0}, "1": {"effect": -0.01764869896450233, "value": 25.0}, "2": {"effect": -0.036167837387115015, "value": 0.0}, "3": {"effect": -0.00888720329442827, "value": 0.0}, "4": {"effect": -0.0025412135186773654, "value": 35.0}, "5": {"effect": 0.17238228962260066, "value": 1.0}}, "simIndex": 113.0}, {"outValue": 0.8535935757085678, "features": {"0": {"effect": 0.08747400677881124, "value": 3.0}, "1": {"effect": -0.012366097948707449, "value": 25.0}, "2": {"effect": -0.026894207179202205, "value": 0.0}, "3": {"effect": -0.007105512672432014, "value": 0.0}, "4": {"effect": 0.019558834393508827, "value": 8.05}, "5": {"effect": 0.18131648084756355, "value": 1.0}}, "simIndex": 52.0}, {"outValue": 0.15263183116010726, "features": {"0": {"effect": -0.17173748042684864, "value": 1.0}, "1": {"effect": 0.03459531278759578, "value": 39.0}, "2": {"effect": 0.028189512129039862, "value": 1.0}, "3": {"effect": -0.007906020205031977, "value": 0.0}, "4": {"effect": -0.02065721287841995, "value": 55.9}, "5": {"effect": -0.3214623517352536, "value": 0.0}}, "simIndex": 166.0}, {"outValue": 0.32826021909388303, "features": {"0": {"effect": 0.08552065144596913, "value": 3.0}, "1": {"effect": -0.016123200631340543, "value": 25.0}, "2": {"effect": -0.039404024677054866, "value": 0.0}, "3": {"effect": -0.009348459123132525, "value": 0.0}, "4": {"effect": 0.020393259416462185, "value": 7.55}, "5": {"effect": -0.32438807882604614, "value": 0.0}}, "simIndex": 141.0}, {"outValue": 0.3901835115702976, "features": {"0": {"effect": 0.08894901833048231, "value": 3.0}, "1": {"effect": -0.04155698317593761, "value": 19.0}, "2": {"effect": 0.035603357404686085, "value": 1.0}, "3": {"effect": -0.009618051744642468, "value": 0.0}, "4": {"effect": 0.020949599595318967, "value": 7.8542}, "5": {"effect": -0.3157535003286355, "value": 0.0}}, "simIndex": 130.0}, {"outValue": 0.8899488508192426, "features": {"0": {"effect": 0.08248864016419526, "value": 3.0}, "1": {"effect": -0.011559860364230762, "value": 25.0}, "2": {"effect": 0.02996510581392299, "value": 1.0}, "3": {"effect": -0.0067121088928912415, "value": 0.0}, "4": {"effect": 0.007792408389102501, "value": 24.15}, "5": {"effect": 0.17636459422011802, "value": 1.0}}, "simIndex": 1.0}, {"outValue": 0.09834606161761572, "features": {"0": {"effect": -0.16055557720568187, "value": 1.0}, "1": {"effect": -0.040350687141463365, "value": 17.0}, "2": {"effect": 0.025735757956837454, "value": 1.0}, "3": {"effect": -0.007236645725544333, "value": 0.0}, "4": {"effect": -0.02036071597091582, "value": 57.0}, "5": {"effect": -0.31049614178464213, "value": 0.0}}, "simIndex": 171.0}, {"outValue": 0.7893594908874998, "features": {"0": {"effect": -0.027324786903484714, "value": 2.0}, "1": {"effect": -0.02827832614603926, "value": 21.0}, "2": {"effect": 0.03614219539988592, "value": 1.0}, "3": {"effect": -0.007686665020296106, "value": 0.0}, "4": {"effect": 0.018390194406940653, "value": 11.5}, "5": {"effect": 0.18650680766146752, "value": 1.0}}, "simIndex": 94.0}, {"outValue": 0.8593798248627225, "features": {"0": {"effect": 0.08645440176866624, "value": 3.0}, "1": {"effect": -0.08863468146355516, "value": 1.0}, "2": {"effect": 0.03207810872508656, "value": 1.0}, "3": {"effect": 0.028254460828932314, "value": 2.0}, "4": {"effect": 0.01032497527934445, "value": 20.575}, "5": {"effect": 0.17929248823522226, "value": 1.0}}, "simIndex": 34.0}, {"outValue": 0.6724589415040524, "features": {"0": {"effect": -0.16334718723169334, "value": 1.0}, "1": {"effect": 0.10262034780403714, "value": 54.0}, "2": {"effect": -0.03309027651375612, "value": 0.0}, "3": {"effect": -0.008360014156780365, "value": 0.0}, "4": {"effect": -0.016290328976823726, "value": 51.8625}, "5": {"effect": 0.179316329090043, "value": 1.0}}, "simIndex": 101.0}, {"outValue": 0.9336436652159373, "features": {"0": {"effect": 0.07394072354227255, "value": 3.0}, "1": {"effect": -0.0021807799721759308, "value": 28.0}, "2": {"effect": 0.07603878968875742, "value": 2.0}, "3": {"effect": -0.006088588665920297, "value": 0.0}, "4": {"effect": 0.017155769446775562, "value": 7.925}, "5": {"effect": 0.16316767968720225, "value": 1.0}}, "simIndex": 93.0}, {"outValue": 0.70444734513295, "features": {"0": {"effect": -0.1585202486627835, "value": 1.0}, "1": {"effect": 0.10883316102622394, "value": 56.0}, "2": {"effect": -0.032230423501507115, "value": 0.0}, "3": {"effect": -0.008188678521533148, "value": 0.0}, "4": {"effect": 0.002616503289901477, "value": 30.6958}, "5": {"effect": 0.18032696001362258, "value": 1.0}}, "simIndex": 103.0}, {"outValue": 0.22514630981588335, "features": {"0": {"effect": -0.04716108752060205, "value": 2.0}, "1": {"effect": 0.026437785488740047, "value": 36.0}, "2": {"effect": -0.037927984707313414, "value": 0.0}, "3": {"effect": -0.008686874653773453, "value": 0.0}, "4": {"effect": 0.014068661728446327, "value": 13.0}, "5": {"effect": -0.3331942620086399, "value": 0.0}}, "simIndex": 156.0}, {"outValue": 0.07818357774619833, "features": {"0": {"effect": -0.15441121633411647, "value": 1.0}, "1": {"effect": -0.035577743896180125, "value": 18.0}, "2": {"effect": -0.03216333079343359, "value": 0.0}, "3": {"effect": 0.026839861850748797, "value": 2.0}, "4": {"effect": -0.0357317018207893, "value": 79.65}, "5": {"effect": -0.3023823627490568, "value": 0.0}}, "simIndex": 179.0}, {"outValue": 0.7429792675079574, "features": {"0": {"effect": -0.03000973058923976, "value": 2.0}, "1": {"effect": -0.014637095984425585, "value": 25.0}, "2": {"effect": -0.031114929628936923, "value": 0.0}, "3": {"effect": -0.008023373518472668, "value": 0.0}, "4": {"effect": 0.028657242056952595, "value": 0.0}, "5": {"effect": 0.18649708368305395, "value": 1.0}}, "simIndex": 81.0}, {"outValue": 0.5160624755817284, "features": {"0": {"effect": -0.1825927335431739, "value": 1.0}, "1": {"effect": -0.07814784268192031, "value": 11.0}, "2": {"effect": 0.04088881547503351, "value": 1.0}, "3": {"effect": 0.03766720178860811, "value": 2.0}, "4": {"effect": -0.08313914452468771, "value": 120.0}, "5": {"effect": 0.16977610757884287, "value": 1.0}}, "simIndex": 123.0}, {"outValue": 0.2145296690796925, "features": {"0": {"effect": -0.04697574548346614, "value": 2.0}, "1": {"effect": 0.022438808413709016, "value": 35.0}, "2": {"effect": -0.03773395511096945, "value": 0.0}, "3": {"effect": -0.008604138901771988, "value": 0.0}, "4": {"effect": 0.007268974172757143, "value": 21.0}, "5": {"effect": -0.3334743454995919, "value": 0.0}}, "simIndex": 157.0}, {"outValue": 0.5813209864684532, "features": {"0": {"effect": -0.1764453298745074, "value": 1.0}, "1": {"effect": 0.01268334197284847, "value": 32.0}, "2": {"effect": -0.03531866199802944, "value": 0.0}, "3": {"effect": -0.008765639036054679, "value": 0.0}, "4": {"effect": 0.002033960403763857, "value": 30.5}, "5": {"effect": 0.17552324351140658, "value": 1.0}}, "simIndex": 108.0}, {"outValue": 0.25464935358660634, "features": {"0": {"effect": -0.04779815875726247, "value": 2.0}, "1": {"effect": -0.003956418578055421, "value": 28.0}, "2": {"effect": 0.03183707378444876, "value": 1.0}, "3": {"effect": -0.008962201947292973, "value": 0.0}, "4": {"effect": 0.003461650555014722, "value": 26.0}, "5": {"effect": -0.3315426629592721, "value": 0.0}}, "simIndex": 151.0}, {"outValue": 0.9304058822549214, "features": {"0": {"effect": 0.07469123069467049, "value": 3.0}, "1": {"effect": 0.030926410082176378, "value": 40.0}, "2": {"effect": 0.026492240860812005, "value": 1.0}, "3": {"effect": 0.008597424787852873, "value": 1.0}, "4": {"effect": 0.012772163051138208, "value": 15.5}, "5": {"effect": 0.16531634128924566, "value": 1.0}}, "simIndex": 6.0}, {"outValue": 0.1034820267109654, "features": {"0": {"effect": -0.16184325030156288, "value": 1.0}, "1": {"effect": -0.013660210154647773, "value": 25.0}, "2": {"effect": 0.02600120522000085, "value": 1.0}, "3": {"effect": -0.007311175026338018, "value": 0.0}, "4": {"effect": -0.039157490394213, "value": 82.1708}, "5": {"effect": -0.3121571241212996, "value": 0.0}}, "simIndex": 173.0}, {"outValue": 0.8535935757085678, "features": {"0": {"effect": 0.08747400677881124, "value": 3.0}, "1": {"effect": -0.012366097948707449, "value": 25.0}, "2": {"effect": -0.026894207179202205, "value": 0.0}, "3": {"effect": -0.007105512672432014, "value": 0.0}, "4": {"effect": 0.019558834393508827, "value": 8.05}, "5": {"effect": 0.18131648084756355, "value": 1.0}}, "simIndex": 53.0}, {"outValue": 0.8412204840255371, "features": {"0": {"effect": 0.08880277968085083, "value": 3.0}, "1": {"effect": -0.0325714115478429, "value": 19.0}, "2": {"effect": -0.027404361481318917, "value": 0.0}, "3": {"effect": -0.007217827372399822, "value": 0.0}, "4": {"effect": 0.025773308163934118, "value": 0.0}, "5": {"effect": 0.18222792509328795, "value": 1.0}}, "simIndex": 64.0}, {"outValue": 0.6717450559257547, "features": {"0": {"effect": -0.16336351494580387, "value": 1.0}, "1": {"effect": 0.10258792727598373, "value": 54.0}, "2": {"effect": -0.033106596593606014, "value": 0.0}, "3": {"effect": 0.013513983263611945, "value": 1.0}, "4": {"effect": -0.038666906899086834, "value": 77.2875}, "5": {"effect": 0.17917009233562997, "value": 1.0}}, "simIndex": 105.0}, {"outValue": 0.6999320545370031, "features": {"0": {"effect": -0.1592507888890433, "value": 1.0}, "1": {"effect": 0.10510100215896731, "value": 55.0}, "2": {"effect": -0.03236236806252707, "value": 0.0}, "3": {"effect": -0.008216221824101491, "value": 0.0}, "4": {"effect": 0.0027624928343336907, "value": 30.5}, "5": {"effect": 0.1802878668303482, "value": 1.0}}, "simIndex": 104.0}, {"outValue": 0.09719880754267052, "features": {"0": {"effect": -0.1600371981613901, "value": 1.0}, "1": {"effect": -0.023531012939553647, "value": 22.0}, "2": {"effect": -0.033236199355426654, "value": 0.0}, "3": {"effect": 0.027802995661433738, "value": 2.0}, "4": {"effect": -0.014884792580250744, "value": 49.5}, "5": {"effect": -0.3105250565711679, "value": 0.0}}, "simIndex": 177.0}, {"outValue": 0.33292052155169916, "features": {"0": {"effect": 0.08581318342794542, "value": 3.0}, "1": {"effect": -0.011999883487237777, "value": 26.0}, "2": {"effect": -0.03942373577897687, "value": 0.0}, "3": {"effect": -0.009368957648207204, "value": 0.0}, "4": {"effect": 0.02019043575696805, "value": 7.8542}, "5": {"effect": -0.32390059220781825, "value": 0.0}}, "simIndex": 138.0}, {"outValue": 0.5963161132881363, "features": {"0": {"effect": -0.17416287568980426, "value": 1.0}, "1": {"effect": -0.004256306884595856, "value": 28.0}, "2": {"effect": 0.04097166017317577, "value": 1.0}, "3": {"effect": -0.008734620929595954, "value": 0.0}, "4": {"effect": -0.04562730850100144, "value": 82.1708}, "5": {"effect": 0.17651549363093222, "value": 1.0}}, "simIndex": 115.0}, {"outValue": 0.07041265282944986, "features": {"0": {"effect": -0.1516047149572573, "value": 1.0}, "1": {"effect": 0.02110935927252275, "value": 36.0}, "2": {"effect": -0.031553386999348065, "value": 0.0}, "3": {"effect": -0.006730844132042585, "value": 0.0}, "4": {"effect": -0.07477686327285094, "value": 135.6333}, "5": {"effect": -0.2976409685705998, "value": 0.0}}, "simIndex": 169.0}, {"outValue": 0.9545330021583112, "features": {"0": {"effect": 0.0683190673992748, "value": 3.0}, "1": {"effect": -0.062283372959850314, "value": 4.0}, "2": {"effect": 0.16416347107757856, "value": 4.0}, "3": {"effect": 0.021603726326231296, "value": 2.0}, "4": {"effect": 0.0029086255415412265, "value": 31.275}, "5": {"effect": 0.1482114132845098, "value": 1.0}}, "simIndex": 87.0}, {"outValue": 0.08878819382021896, "features": {"0": {"effect": -0.15773064169002943, "value": 1.0}, "1": {"effect": 0.0028231046651396446, "value": 30.0}, "2": {"effect": -0.03276397970004738, "value": 0.0}, "3": {"effect": -0.007057646655353744, "value": 0.0}, "4": {"effect": -0.020056112101176415, "value": 56.9292}, "5": {"effect": -0.3080366021873395, "value": 0.0}}, "simIndex": 175.0}, {"outValue": 0.2944878919462983, "features": {"0": {"effect": 0.08330684314706899, "value": 3.0}, "1": {"effect": -0.04463874943353134, "value": 18.0}, "2": {"effect": -0.039168948326048156, "value": 0.0}, "3": {"effect": -0.009176381413020138, "value": 0.0}, "4": {"effect": 0.019907907046730438, "value": 7.4958}, "5": {"effect": -0.32735285056392727, "value": 0.0}}, "simIndex": 148.0}, {"outValue": 0.6553424309149063, "features": {"0": {"effect": 0.0960934429905432, "value": 3.0}, "1": {"effect": 0.018207501775359652, "value": 33.0}, "2": {"effect": 0.18745898490501553, "value": 3.0}, "3": {"effect": -0.009270389878661828, "value": 0.0}, "4": {"effect": 0.014925150884063265, "value": 15.85}, "5": {"effect": -0.26368233125043933, "value": 0.0}}, "simIndex": 127.0}, {"outValue": 0.32804330374971824, "features": {"0": {"effect": 0.08550761388456408, "value": 3.0}, "1": {"effect": -0.016122639415488478, "value": 25.0}, "2": {"effect": -0.03940314867986555, "value": 0.0}, "3": {"effect": -0.009347588064389494, "value": 0.0}, "4": {"effect": 0.020212149962918667, "value": 7.75}, "5": {"effect": -0.3244131554270468, "value": 0.0}}, "simIndex": 144.0}, {"outValue": 0.10036020045346272, "features": {"0": {"effect": -0.16102416513234197, "value": 1.0}, "1": {"effect": -0.013588398837456855, "value": 25.0}, "2": {"effect": 0.025845671328396325, "value": 1.0}, "3": {"effect": -0.007265195433850541, "value": 0.0}, "4": {"effect": -0.044131898733836245, "value": 89.1042}, "5": {"effect": -0.31108588422647376, "value": 0.0}}, "simIndex": 174.0}, {"outValue": 0.8919482823225954, "features": {"0": {"effect": 0.08215941441223755, "value": 3.0}, "1": {"effect": -0.011508335136081481, "value": 25.0}, "2": {"effect": 0.02981236515031277, "value": 1.0}, "3": {"effect": -0.0066880275043918955, "value": 0.0}, "4": {"effect": 0.01058835202050408, "value": 19.9667}, "5": {"effect": 0.17597444189098854, "value": 1.0}}, "simIndex": 3.0}, {"outValue": 0.8535935757085678, "features": {"0": {"effect": 0.08747400677881124, "value": 3.0}, "1": {"effect": -0.012366097948707449, "value": 25.0}, "2": {"effect": -0.026894207179202205, "value": 0.0}, "3": {"effect": -0.007105512672432014, "value": 0.0}, "4": {"effect": 0.019558834393508827, "value": 8.05}, "5": {"effect": 0.18131648084756355, "value": 1.0}}, "simIndex": 51.0}, {"outValue": 0.7242367936608112, "features": {"0": {"effect": -0.031367810201578604, "value": 2.0}, "1": {"effect": -0.09898966950138025, "value": 3.0}, "2": {"effect": 0.0384982822737981, "value": 1.0}, "3": {"effect": 0.012934014837128513, "value": 1.0}, "4": {"effect": 0.006502221763230481, "value": 26.0}, "5": {"effect": 0.18504968300058716, "value": 1.0}}, "simIndex": 74.0}, {"outValue": 0.13365092381339028, "features": {"0": {"effect": -0.1680981949958484, "value": 1.0}, "1": {"effect": 0.05044944446640183, "value": 44.0}, "2": {"effect": -0.03466455501975729, "value": 0.0}, "3": {"effect": -0.007654711494976069, "value": 0.0}, "4": {"effect": 0.0011481495756167082, "value": 27.7208}, "5": {"effect": -0.3191392802070723, "value": 0.0}}, "simIndex": 162.0}, {"outValue": 0.8440935299550355, "features": {"0": {"effect": 0.08855283475752367, "value": 3.0}, "1": {"effect": -0.02251897479941363, "value": 22.0}, "2": {"effect": -0.02729156331432045, "value": 0.0}, "3": {"effect": -0.00719460581144158, "value": 0.0}, "4": {"effect": 0.018790067536137996, "value": 9.35}, "5": {"effect": 0.1821457000975237, "value": 1.0}}, "simIndex": 63.0}, {"outValue": 0.8508485353035814, "features": {"0": {"effect": 0.08779236458962927, "value": 3.0}, "1": {"effect": -0.01571013603516424, "value": 24.0}, "2": {"effect": -0.027010273527130944, "value": 0.0}, "3": {"effect": -0.00713153342747027, "value": 0.0}, "4": {"effect": 0.01972765915416264, "value": 7.8958}, "5": {"effect": 0.18157038306052914, "value": 1.0}}, "simIndex": 58.0}, {"outValue": 0.07758039892679103, "features": {"0": {"effect": -0.15416442444841416, "value": 1.0}, "1": {"effect": -0.012998372300581318, "value": 25.0}, "2": {"effect": 0.024617007495397886, "value": 1.0}, "3": {"effect": -0.006890705843118655, "value": 0.0}, "4": {"effect": -0.08392972801355719, "value": 146.5208}, "5": {"effect": -0.30066344945196133, "value": 0.0}}, "simIndex": 170.0}, {"outValue": 0.26125143913263127, "features": {"0": {"effect": -0.04787955221183121, "value": 2.0}, "1": {"effect": -0.01586968828696415, "value": 25.0}, "2": {"effect": 0.03202508054156636, "value": 1.0}, "3": {"effect": 0.012284850476350957, "value": 1.0}, "4": {"effect": 5.9166506662053494e-05, "value": 30.0}, "5": {"effect": -0.33097848938217855, "value": 0.0}}, "simIndex": 152.0}, {"outValue": 0.32771806835242223, "features": {"0": {"effect": 0.08548804997282151, "value": 3.0}, "1": {"effect": -0.016121792299952087, "value": 25.0}, "2": {"effect": -0.039401822833600136, "value": 0.0}, "3": {"effect": -0.009346278685339323, "value": 0.0}, "4": {"effect": 0.01994052250789436, "value": 8.05}, "5": {"effect": -0.3244506817984279, "value": 0.0}}, "simIndex": 140.0}, {"outValue": 0.9215083381116239, "features": {"0": {"effect": 0.0766825657062676, "value": 3.0}, "1": {"effect": 0.014742815032975747, "value": 34.0}, "2": {"effect": 0.027343519795945168, "value": 1.0}, "3": {"effect": 0.008885847104733419, "value": 1.0}, "4": {"effect": 0.013682397251858346, "value": 14.4}, "5": {"effect": 0.16856112173081786, "value": 1.0}}, "simIndex": 7.0}, {"outValue": 0.5434515623773238, "features": {"0": {"effect": -0.18104736633807547, "value": 1.0}, "1": {"effect": -0.01758080110981361, "value": 25.0}, "2": {"effect": -0.03605069241506606, "value": 0.0}, "3": {"effect": -0.00887099852346035, "value": 0.0}, "4": {"effect": 0.002527141857311843, "value": 29.7}, "5": {"effect": 0.1728642074174017, "value": 1.0}}, "simIndex": 109.0}, {"outValue": 0.07685850012703077, "features": {"0": {"effect": -0.15231740917644077, "value": 1.0}, "1": {"effect": -0.025456390256828998, "value": 21.0}, "2": {"effect": 0.07822806456571169, "value": 2.0}, "3": {"effect": 0.02645833871368866, "value": 2.0}, "4": {"effect": -0.17026861599923065, "value": 262.375}, "5": {"effect": -0.2913955592088949, "value": 0.0}}, "simIndex": 158.0}, {"outValue": 0.9132840938089227, "features": {"0": {"effect": 0.07825714163633565, "value": 3.0}, "1": {"effect": 0.06517325838658386, "value": 51.0}, "2": {"effect": -0.023963525815429723, "value": 0.0}, "3": {"effect": -0.006421948366768959, "value": 0.0}, "4": {"effect": 0.01855620296660499, "value": 7.0542}, "5": {"effect": 0.1700728935125711, "value": 1.0}}, "simIndex": 21.0}, {"outValue": 0.35594837226645604, "features": {"0": {"effect": 0.08720415678261538, "value": 3.0}, "1": {"effect": 0.006684709180992208, "value": 30.5}, "2": {"effect": -0.039477153684440595, "value": 0.0}, "3": {"effect": -0.009457749935753289, "value": 0.0}, "4": {"effect": 0.020601699892425017, "value": 7.75}, "5": {"effect": -0.3212173614584085, "value": 0.0}}, "simIndex": 136.0}, {"outValue": 0.8564832166267902, "features": {"0": {"effect": 0.08712894018509143, "value": 3.0}, "1": {"effect": -0.009045130113886898, "value": 26.0}, "2": {"effect": -0.026770513478858737, "value": 0.0}, "3": {"effect": -0.007077654271610279, "value": 0.0}, "4": {"effect": 0.019609417781727426, "value": 7.8958}, "5": {"effect": 0.18102808503530143, "value": 1.0}}, "simIndex": 39.0}, {"outValue": 0.5424749420268371, "features": {"0": {"effect": -0.18115761940319544, "value": 1.0}, "1": {"effect": -0.017591177052798945, "value": 25.0}, "2": {"effect": -0.03606856315712759, "value": 0.0}, "3": {"effect": -0.008873506664193342, "value": 0.0}, "4": {"effect": 0.001763368458666273, "value": 30.5}, "5": {"effect": 0.17279236835646034, "value": 1.0}}, "simIndex": 110.0}, {"outValue": 0.7147519243248086, "features": {"0": {"effect": -0.15684607752622526, "value": 1.0}, "1": {"effect": 0.11611095884349695, "value": 58.0}, "2": {"effect": -0.03192686568523834, "value": 0.0}, "3": {"effect": -0.008124910397685148, "value": 0.0}, "4": {"effect": 0.0035261128883486564, "value": 29.7}, "5": {"effect": 0.18040263471308593, "value": 1.0}}, "simIndex": 102.0}, {"outValue": 0.8540847222957536, "features": {"0": {"effect": 0.08741247690335102, "value": 3.0}, "1": {"effect": -0.012355486046584208, "value": 25.0}, "2": {"effect": -0.026872936068282324, "value": 0.0}, "3": {"effect": -0.007100611746160397, "value": 0.0}, "4": {"effect": 0.020128846348069707, "value": 7.25}, "5": {"effect": 0.181262361416334, "value": 1.0}}, "simIndex": 42.0}, {"outValue": 0.8495835360446616, "features": {"0": {"effect": 0.08796764693924777, "value": 3.0}, "1": {"effect": -0.012451921364205608, "value": 25.0}, "2": {"effect": -0.027066408607817602, "value": 0.0}, "3": {"effect": -0.007145089576458159, "value": 0.0}, "4": {"effect": 0.014931721283976823, "value": 14.5}, "5": {"effect": 0.18173751588089262, "value": 1.0}}, "simIndex": 37.0}, {"outValue": 0.7285887088312183, "features": {"0": {"effect": -0.030798619559356064, "value": 2.0}, "1": {"effect": -0.014884349678388156, "value": 25.0}, "2": {"effect": -0.031569339715917424, "value": 0.0}, "3": {"effect": -0.00811959403548234, "value": 0.0}, "4": {"effect": 0.015990005238431172, "value": 15.05}, "5": {"effect": 0.18636053509290532, "value": 1.0}}, "simIndex": 85.0}, {"outValue": 0.8991018835537501, "features": {"0": {"effect": -0.020688979235918295, "value": 2.0}, "1": {"effect": 0.09644966235411234, "value": 60.0}, "2": {"effect": 0.029505026602455728, "value": 1.0}, "3": {"effect": 0.009651212114677371, "value": 1.0}, "4": {"effect": -0.0019474181986402286, "value": 39.0}, "5": {"effect": 0.17452230842803737, "value": 1.0}}, "simIndex": 33.0}, {"outValue": 0.8860264330748258, "features": {"0": {"effect": 0.08304503524141855, "value": 3.0}, "1": {"effect": 0.02245549630059157, "value": 36.0}, "2": {"effect": -0.025410511326481984, "value": 0.0}, "3": {"effect": -0.006765624880204185, "value": 0.0}, "4": {"effect": 0.024276100993069167, "value": 0.0}, "5": {"effect": 0.17681586525740683, "value": 1.0}}, "simIndex": 16.0}, {"outValue": 0.8922713091424823, "features": {"0": {"effect": 0.08205999673427547, "value": 3.0}, "1": {"effect": 0.03604242610571064, "value": 40.5}, "2": {"effect": -0.025101455465968772, "value": 0.0}, "3": {"effect": -0.006693687612376889, "value": 0.0}, "4": {"effect": 0.01880505969723922, "value": 7.75}, "5": {"effect": 0.17554889819457686, "value": 1.0}}, "simIndex": 23.0}, {"outValue": 0.8748856738919465, "features": {"0": {"effect": 0.08471856594368452, "value": 3.0}, "1": {"effect": 0.01342703559948101, "value": 33.0}, "2": {"effect": -0.02594702359304181, "value": 0.0}, "3": {"effect": -0.0068902011279864325, "value": 0.0}, "4": {"effect": 0.019265346566466897, "value": 7.775}, "5": {"effect": 0.17870187901431653, "value": 1.0}}, "simIndex": 14.0}, {"outValue": 0.6549458872565688, "features": {"0": {"effect": -0.1662598013534632, "value": 1.0}, "1": {"effect": 0.06632046698613599, "value": 45.0}, "2": {"effect": -0.03360311778905106, "value": 0.0}, "3": {"effect": -0.008464640400181564, "value": 0.0}, "4": {"effect": 0.0060611455664723035, "value": 26.55}, "5": {"effect": 0.17928176275763053, "value": 1.0}}, "simIndex": 106.0}, {"outValue": 0.08481427845063949, "features": {"0": {"effect": -0.1565128456493478, "value": 1.0}, "1": {"effect": 0.0091660993390182, "value": 32.0}, "2": {"effect": -0.03252351485291116, "value": 0.0}, "3": {"effect": -0.006991725399189824, "value": 0.0}, "4": {"effect": -0.03377515396785162, "value": 76.2917}, "5": {"effect": -0.3061586525081041, "value": 0.0}}, "simIndex": 176.0}, {"outValue": 0.9057851706066579, "features": {"0": {"effect": 0.07969640216672173, "value": 3.0}, "1": {"effect": 0.05441256170781496, "value": 47.0}, "2": {"effect": -0.02438668538656484, "value": 0.0}, "3": {"effect": -0.006524055748058921, "value": 0.0}, "4": {"effect": 0.01870172666846357, "value": 7.25}, "5": {"effect": 0.17227514970925556, "value": 1.0}}, "simIndex": 24.0}, {"outValue": 0.8744069867111816, "features": {"0": {"effect": 0.08478865349601522, "value": 3.0}, "1": {"effect": 0.01344051448993152, "value": 33.0}, "2": {"effect": -0.02596968499504714, "value": 0.0}, "3": {"effect": -0.006895479403753069, "value": 0.0}, "4": {"effect": 0.018657569582031235, "value": 8.6625}, "5": {"effect": 0.178775342052978, "value": 1.0}}, "simIndex": 15.0}, {"outValue": 0.31301092115500906, "features": {"0": {"effect": 0.08449591498266948, "value": 3.0}, "1": {"effect": -0.04498123815410836, "value": 18.0}, "2": {"effect": -0.03933870854599575, "value": 0.0}, "3": {"effect": 0.012700963988930464, "value": 1.0}, "4": {"effect": 0.014047864562515236, "value": 14.4542}, "5": {"effect": -0.3255239471680278, "value": 0.0}}, "simIndex": 146.0}, {"outValue": 0.8540847222957536, "features": {"0": {"effect": 0.08741247690335102, "value": 3.0}, "1": {"effect": -0.012355486046584208, "value": 25.0}, "2": {"effect": -0.026872936068282324, "value": 0.0}, "3": {"effect": -0.007100611746160397, "value": 0.0}, "4": {"effect": 0.020128846348069707, "value": 7.25}, "5": {"effect": 0.181262361416334, "value": 1.0}}, "simIndex": 43.0}, {"outValue": 0.7924832196253436, "features": {"0": {"effect": -0.027086401424320745, "value": 2.0}, "1": {"effect": -0.013763701386625107, "value": 25.0}, "2": {"effect": 0.036014169076879, "value": 1.0}, "3": {"effect": -0.007666238631797627, "value": 0.0}, "4": {"effect": 0.006776469916743452, "value": 26.0}, "5": {"effect": 0.18659885058543885, "value": 1.0}}, "simIndex": 97.0}, {"outValue": 0.8537779159877965, "features": {"0": {"effect": 0.08745094060247136, "value": 3.0}, "1": {"effect": -0.012362117616543093, "value": 25.0}, "2": {"effect": -0.026886228241055826, "value": 0.0}, "3": {"effect": -0.007103674615936001, "value": 0.0}, "4": {"effect": 0.019772689499711355, "value": 7.75}, "5": {"effect": 0.18129623487012292, "value": 1.0}}, "simIndex": 48.0}, {"outValue": 0.8541000485908506, "features": {"0": {"effect": 0.08741055306312039, "value": 3.0}, "1": {"effect": -0.012355154544311905, "value": 25.0}, "2": {"effect": -0.02687227165857214, "value": 0.0}, "3": {"effect": -0.007100458621038301, "value": 0.0}, "4": {"effect": 0.020146645386774742, "value": 7.225}, "5": {"effect": 0.18126066347585207, "value": 1.0}}, "simIndex": 41.0}, {"outValue": 0.8692529596114695, "features": {"0": {"effect": -0.02265876042788552, "value": 2.0}, "1": {"effect": -0.018543319343361585, "value": 23.0}, "2": {"effect": 0.09141924656615144, "value": 2.0}, "3": {"effect": 0.010417529406154633, "value": 1.0}, "4": {"effect": 0.01698899773540899, "value": 11.5}, "5": {"effect": 0.18001919418597578, "value": 1.0}}, "simIndex": 99.0}, {"outValue": 0.32804330374971824, "features": {"0": {"effect": 0.08550761388456408, "value": 3.0}, "1": {"effect": -0.016122639415488478, "value": 25.0}, "2": {"effect": -0.03940314867986555, "value": 0.0}, "3": {"effect": -0.009347588064389494, "value": 0.0}, "4": {"effect": 0.020212149962918667, "value": 7.75}, "5": {"effect": -0.3244131554270468, "value": 0.0}}, "simIndex": 142.0}, {"outValue": 0.4725132975810876, "features": {"0": {"effect": 0.09305576588030318, "value": 3.0}, "1": {"effect": 0.03092428315289099, "value": 36.0}, "2": {"effect": 0.037385316792870715, "value": 1.0}, "3": {"effect": -0.009746301650829542, "value": 0.0}, "4": {"effect": 0.012868949597752435, "value": 17.4}, "5": {"effect": -0.30358478768092595, "value": 0.0}}, "simIndex": 128.0}, {"outValue": 0.8536883506486206, "features": {"0": {"effect": 0.08746215189680547, "value": 3.0}, "1": {"effect": -0.012364051928016654, "value": 25.0}, "2": {"effect": -0.026890105664753212, "value": 0.0}, "3": {"effect": -0.007104567880135401, "value": 0.0}, "4": {"effect": 0.019668771000840433, "value": 7.8958}, "5": {"effect": 0.18130608173485416, "value": 1.0}}, "simIndex": 55.0}, {"outValue": 0.8329295166551044, "features": {"0": {"effect": 0.08969084280008666, "value": 3.0}, "1": {"effect": -0.036352387196252826, "value": 18.0}, "2": {"effect": -0.027739485479201804, "value": 0.0}, "3": {"effect": -0.007293470475577952, "value": 0.0}, "4": {"effect": 0.020159155187529865, "value": 7.775}, "5": {"effect": 0.18285479032949464, "value": 1.0}}, "simIndex": 69.0}, {"outValue": 0.4639983270695178, "features": {"0": {"effect": 0.09256978112235584, "value": 3.0}, "1": {"effect": -0.016014153703258302, "value": 25.0}, "2": {"effect": 0.03712942467835198, "value": 1.0}, "3": {"effect": 0.03629426971031645, "value": 2.0}, "4": {"effect": 0.007154290839872554, "value": 23.45}, "5": {"effect": -0.3047453570671465, "value": 0.0}}, "simIndex": 129.0}, {"outValue": 0.8432526689934234, "features": {"0": {"effect": 0.08880752183653783, "value": 3.0}, "1": {"effect": 0.010877367271613609, "value": 32.0}, "2": {"effect": -0.02734857760368746, "value": 0.0}, "3": {"effect": -0.00721107263956642, "value": 0.0}, "4": {"effect": -0.015754634557433346, "value": 56.4958}, "5": {"effect": 0.18227199319693344, "value": 1.0}}, "simIndex": 8.0}, {"outValue": 0.8367347759794133, "features": {"0": {"effect": 0.08930669902755801, "value": 3.0}, "1": {"effect": -0.032807827433713146, "value": 19.0}, "2": {"effect": -0.027588086048349413, "value": 0.0}, "3": {"effect": -0.007259876510140005, "value": 0.0}, "4": {"effect": 0.020855290397275284, "value": 6.75}, "5": {"effect": 0.18261850505775673, "value": 1.0}}, "simIndex": 71.0}, {"outValue": 0.4136683484392544, "features": {"0": {"effect": 0.09026021501272763, "value": 3.0}, "1": {"effect": -0.016110355987825717, "value": 25.0}, "2": {"effect": 0.03619661288954536, "value": 1.0}, "3": {"effect": -0.009682320672442749, "value": 0.0}, "4": {"effect": 0.014177818157101, "value": 15.5}, "5": {"effect": -0.31278369244887694, "value": 0.0}}, "simIndex": 133.0}, {"outValue": 0.8415536430731785, "features": {"0": {"effect": 0.08882062125159525, "value": 3.0}, "1": {"effect": -0.025932749750698404, "value": 21.0}, "2": {"effect": -0.027394924982510706, "value": 0.0}, "3": {"effect": -0.007217467043579253, "value": 0.0}, "4": {"effect": 0.019343226856362382, "value": 8.6625}, "5": {"effect": 0.18232486525298341, "value": 1.0}}, "simIndex": 59.0}, {"outValue": 0.3751351133169436, "features": {"0": {"effect": 0.08821156786044876, "value": 3.0}, "1": {"effect": -0.016208267332674067, "value": 25.0}, "2": {"effect": -0.03955265580530215, "value": 0.0}, "3": {"effect": 0.035436016211035704, "value": 2.0}, "4": {"effect": 0.014014749589801215, "value": 15.2458}, "5": {"effect": -0.31837636869539165, "value": 0.0}}, "simIndex": 135.0}, {"outValue": 0.6414848937585669, "features": {"0": {"effect": -0.035995203656404046, "value": 2.0}, "1": {"effect": -0.11414066541026038, "value": 1.0}, "2": {"effect": -0.03389868005330504, "value": 0.0}, "3": {"effect": 0.0363144123473694, "value": 2.0}, "4": {"effect": -0.0038459001882976615, "value": 37.0042}, "5": {"effect": 0.18144085923043882, "value": 1.0}}, "simIndex": 72.0}, {"outValue": 0.9350367014108375, "features": {"0": {"effect": 0.07345827875557905, "value": 3.0}, "1": {"effect": 0.0998164531636228, "value": 65.0}, "2": {"effect": -0.022595995432637683, "value": 0.0}, "3": {"effect": -0.006081715532738119, "value": 0.0}, "4": {"effect": 0.01720935714229909, "value": 7.75}, "5": {"effect": 0.16162025182568657, "value": 1.0}}, "simIndex": 27.0}, {"outValue": 0.41491673939784657, "features": {"0": {"effect": 0.09032061524467971, "value": 3.0}, "1": {"effect": -0.016108821994850676, "value": 25.0}, "2": {"effect": 0.036223894720016, "value": 1.0}, "3": {"effect": -0.009684457466870952, "value": 0.0}, "4": {"effect": 0.01515403298071459, "value": 14.4542}, "5": {"effect": -0.3125985955748679, "value": 0.0}}, "simIndex": 134.0}, {"outValue": 0.7487507653785292, "features": {"0": {"effect": -0.15112320990947145, "value": 1.0}, "1": {"effect": 0.14060536666280693, "value": 65.0}, "2": {"effect": -0.030873337577187028, "value": 0.0}, "3": {"effect": -0.007896618862797175, "value": 0.0}, "4": {"effect": 0.006284980919245585, "value": 26.55}, "5": {"effect": 0.1801435126569066, "value": 1.0}}, "simIndex": 100.0}, {"outValue": 0.703125528171719, "features": {"0": {"effect": -0.03231597648467336, "value": 2.0}, "1": {"effect": -0.03898723182124845, "value": 19.0}, "2": {"effect": -0.0323176045287936, "value": 0.0}, "3": {"effect": -0.008269026870518989, "value": 0.0}, "4": {"effect": 0.01787643782035923, "value": 13.0}, "5": {"effect": 0.1855288585675684, "value": 1.0}}, "simIndex": 76.0}, {"outValue": 0.8994946925388678, "features": {"0": {"effect": 0.08083219407646433, "value": 3.0}, "1": {"effect": 0.046103194895320006, "value": 44.0}, "2": {"effect": -0.02472671869804066, "value": 0.0}, "3": {"effect": -0.00660516029521116, "value": 0.0}, "4": {"effect": 0.01838439574397404, "value": 8.05}, "5": {"effect": 0.17389671532733547, "value": 1.0}}, "simIndex": 25.0}, {"outValue": 0.9264837613128519, "features": {"0": {"effect": 0.0754678420656669, "value": 3.0}, "1": {"effect": 0.08553039975449614, "value": 59.0}, "2": {"effect": -0.023162164258815357, "value": 0.0}, "3": {"effect": -0.006224633904163099, "value": 0.0}, "4": {"effect": 0.01789901132108672, "value": 7.25}, "5": {"effect": 0.16536323484555482, "value": 1.0}}, "simIndex": 29.0}, {"outValue": 0.5313447091781714, "features": {"0": {"effect": -0.18238890144915618, "value": 1.0}, "1": {"effect": -0.01770598247812251, "value": 25.0}, "2": {"effect": -0.036267081654915434, "value": 0.0}, "3": {"effect": -0.008900473619938926, "value": 0.0}, "4": {"effect": -0.006955669491000668, "value": 39.6}, "5": {"effect": 0.17195274638227936, "value": 1.0}}, "simIndex": 114.0}], "plot_cmap": "RdBu", "ordering_keys_time_format": null, "baseValue": 0.6116100714890258, "link": "identity"}),
    document.getElementById('iEZ21XTKJB28Y4UIKTO0F')
  );
</script>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Feature 5 = Sex</span>
<span class="c1"># Feature 0 = Class</span>
<span class="c1"># Feature 2 = Had siblings</span>
<span class="c1"># Feature 1 = age</span>
<span class="c1"># Feature 4 = Fare</span>
<span class="c1"># Feature 3 = Parent / child</span>
<span class="n">shap</span><span class="o">.</span><span class="n">summary_plot</span><span class="p">(</span><span class="n">shap_values</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhYAAAEHCAYAAADyJQ9wAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3XmcFdWd9/HPsbvBJHQji8QFBQVjojEa+fkkZFAxLtGJ
ZDCBaAZFXOMYHyUuY4ZFexR3xWVUNC5oXGJiEmE0JCouExMhzk99MEpQFIGoKDSBBoKoQD1/nGos
rre7b/ctupH+vl+v++quU6dOndpu/erUqVshSRJERERE8rBVe1dAREREthwKLERERCQ3CixEREQk
NwosREREJDcKLERERCQ3CixEREQkNwosREREJDcKLERERCQ3CixEREQkN5XtXYFPo4cffjgZMmRI
e1dDRESkrYRSM6rFQkRERHKjwEJERERyo8BCREREcqPAQkRERHKjwEJERERyo8BCREREcqPAQkRE
RHKjwEJERERyo8BCREREcqPAQkRERHKjwEJERERyo8BCREREcqPAQkRERHKjwEJERERyo8BCRERE
cqPAQkRERHKjwEJERERyo8BCREREchOSJGnvOnzq9LniHa00kRwd3K9ze1dBZItx57Aem6LYUGpG
tViIiIhIbhRYiIiISG4UWIiIiEhuFFiIiIhIbhRYiIiISG4UWIiIiEhuFFiIiIhIbiqby2BmTwMD
gY8yyQ+4+8nlztzMRgHj3L1/uWW1Yt59gTeB1UDD71Isd/febV0XERGRLUWzgUXqYnefsElr0kpm
FoAKd1/byiJ2d/e38qyTiIhIR1VqYNEoMxsKjAf6AYuACe5+XzquN3A7MADoBLwEjHb3581sIHAL
0MnMVqXFHZn+ne7ulZl51AKD3P2QdDgBRgPHAXsCBwEzzewU4CxgJ2AecL67P1buMoqIiEhpyupj
YWaHAncQT/LdgeOBG83sgEz5NwN9gO2AF4DfmFmVu88ATgPmuXuX9PN0C2Z/EnA00AV40cxOBc4H
RgDdgLHpvJq7zfJnM1tiZk+b2eAWzF9EREQKlNpiMdbMzs0MH+7uM4mtA9e7+zNp+nNmdi8wEviD
uy8EFjZMZGbjgDOB3YDZZdb9and/I/1/nZmdCVzk7rPStGlm9hRwDFDsNk4dse/IC0AVcCLwOzP7
mru/VGbdREREOqRSA4tLGuljsQtwkJmdnUmrAJ4BMLOewERgMLANsD7Ns22rarux+UXqcpOZ3ZBJ
qwSK9p9w91XAzHTwQ+C/zOw7wHDiLRsRERFpoXL7WCwA7nL3qxoZfxmwPfA1d19kZtXACj5+S9r6
ItOsAirMrLO7f5Cm7VAkX+G0C4AL3f3BFi3BJ8ss+Q1uIiIisrFyA4vrgMlmNhN4lthasRcQ3N2B
GuLjnMvMrAtwRcH07wK9zKzG3Vekaa8Sg4uTzWwS8A1gGPGWRVOuBWrNbC4wC9ia2Gm0zt3nFGY2
s6+n85lDXA8jgQOBMS1YfhEREckoq/Nm+sTFqcBVxD4Li4gn+C5plguBXsBS4u2FZ4F1mSKeBB4H
3jSz5WZ2oLuvBE4AzgHqif047i6hLrcBVwKTgWXEvh3jif0nitkFmJLO423iEyZD3P35UpZdRERE
PikkSdJ8LtlInyve0UoTydHB/Tq3dxVEthh3DuuxKYotuZuAftJbREREcqPAQkRERHKjwEJERERy
o8BCREREclP2u0I6ohv3eJ4hQ4a0dzVEREQ2O2qxEBERkdwosBAREZHcKLAQERGR3CiwEBERkdwo
sBAREZHcKLAQERGR3CiwEBERkdwosBAREZHc6O2mraC3m7Ytvfmy7W2ityOKyKeX3m4qIiIibU+B
hYiIiORGgYWIiIjkRoGFiIiI5EaBhYiIiORGgYWIiIjkprK5DGb2NDAQ+CiT/IC7n1zuzM1sFDDO
3fuXW1Yr52/AzcCXgUXAhe5+b3vURUREZEvQbGCRutjdJ2zSmrSSmQWgwt3XtnC6rsDvgKuB/YED
gIfM7A13n5F/TUVERLZ8pQYWjTKzocB4oB/xqn+Cu9+XjusN3A4MADoBLwGj3f15MxsI3AJ0MrNV
aXFHpn+nu3tlZh61wCB3PyQdToDRwHHAnsBBwEwzOwU4C9gJmAec7+6PNVL17wLvA1e6ewI8bmYP
AacCCixERERaoaw+FmZ2KHAH8STfHTgeuNHMDsiUfzPQB9gOeAH4jZlVpa0CpwHz3L1L+nm6BbM/
CTga6AK8aGanAucDI4BuwNh0Xo3dZtkbeCENKhq8kKaLiIhIK5TaYjHWzM7NDB/u7jOJrQPXu/sz
afpzZnYvMBL4g7svBBY2TGRm44Azgd2A2WXW/Wp3fyP9f52ZnQlc5O6z0rRpZvYUcAxQ7DZONVBf
kLYcqCmzXiIiIh1WqYHFJY30sdgFOMjMzs6kVQDPAJhZT2AiMBjYBlif5tm2VbXd2PwidbnJzG7I
pFUCbzUy/Uqgb0HaNsCKHOomIiLSIZXbx2IBcJe7X9XI+MuA7YGvufsiM6smnrgbXmayvsg0q4AK
M+vs7h+kaTsUyVc47QLiUx0Pllj3WcBRBWlfTdNFRESkFcoNLK4DJpvZTOBZYmvFXkBwdyfeVlgN
LDOzLsAVBdO/C/Qysxp3b2gpeJUYXJxsZpOAbwDDiP0fmnItUGtmc4nBwdbETqN17j6nSP6HgCvN
7DzgeuKTId8FDi156UVERGQjZXXeTJ+4OBW4CqgjPhVyLbFDJcCFQC9gKfGJkGeBdZkingQeB940
s+VmdqC7rwROAM4h9oE4C7i7hLrcBlwJTAaWEft2jAeqGsm/HPhnYHg6n9uA0/SoqYiISOuFJEma
zyUb6XPFO1ppbejgfp3buwodzp3DerR3FURk8xKazxLpJ71FREQkNwosREREJDcKLERERCQ3CixE
REQkN2W/K6QjunGP5xkyZEh7V0NERGSzoxYLERERyY0CCxEREcmNAgsRERHJjQILERERyY0CCxER
EcmNAgsRERHJjQILERERyY0CCxEREcmN3m7aCpvz20035zeB6o2ZIiKfWnq7qYiIiLQ9BRYiIiKS
GwUWIiIikhsFFiIiIpIbBRYiIiKSGwUWIiIikhsFFiIiIpKbyuYymNnTwEDgo0zyA+5+crkzN7NR
wDh3719uWa2Y99eB8YABWwOvAxe7+5S2rouIiMiWotnAInWxu0/YpDVpJTMLQIW7r23hpN2BXwCj
gKXAd4Cfm9kB7v6/+dZSRESkYyg1sGiUmQ0lXvn3AxYBE9z9vnRcb+B2YADQCXgJGO3uz5vZQOAW
oJOZrUqLOzL9O93dKzPzqAUGufsh6XACjAaOA/YEDgJmmtkpwFnATsA84Hx3f6xYvd19WkHSFDN7
GRgEKLAQERFphbL6WJjZocAdxJN8d+B44EYzOyBT/s1AH2A74AXgN2ZW5e4zgNOAee7eJf083YLZ
nwQcDXQBXjSzU4HzgRFAN2BsOq+SbrOY2XbEIOWlFtRBREREMkptsRhrZudmhg9395nE1oHr3f2Z
NP05M7sXGAn8wd0XAgsbJjKzccCZwG7A7DLrfrW7v5H+v87MzgQucvdZado0M3sKOAZo8jaOmX0O
+DXw3+7+RJn1EhER6bBKDSwuaaSPxS7AQWZ2diatAngGwMx6AhOBwcA2wPo0z7atqu3G5hepy01m
dkMmrRJ4q6lCzKwa+C2wmBgQiYiISCuV28diAXCXu1/VyPjLgO2Br7n7ovQkvoKP35K2vsg0q4AK
M+vs7h+kaTsUyVc47QLgQnd/sNTKm1l34PfE/hjHtqIDqIiIiGSUG1hcB0w2s5nAs8TWir2A4O4O
1ACrgWVm1gW4omD6d4FeZlbj7ivStFeJwcXJZjYJ+AYwjNg/oynXArVmNheYRXyEdABQ5+5zCjOn
fSoeT8s90d3XtWzRRUREpFBZnTfTJy5OBa4C6ohPhVxL7FAJcCHQi/g450vE4CN7An+SeHJ/08yW
m9mB7r4SOAE4B6gn9uO4u4S63AZcCUwGlhH7dowHqhqZ5IfAl4lBS72ZrUo/Y0pbehERESkUkiRp
7zp86vS54p3NdqUd3K9ze1ehUXcO69HeVRARkdYJzWeJ9JPeIiIikhsFFiIiIpIbBRYiIiKSm7J/
0rsjunGP5xkyZEh7V0NERGSzoxYLERERyY0CCxEREcmNAgsRERHJjQILERERyY0CCxEREcmNAgsR
ERHJjQILERERyY0CCxEREcmNXkLWCpvDS8g2h5eN6aViIiIdhl5CJiIiIm1PgYWIiIjkRoGFiIiI
5EaBhYiIiORGgYWIiIjkRoGFiIiI5EaBhYiIiOSmsrkMZvY0MBD4KJP8gLufXO7MzWwUMM7d+5db
Vivm/RngZ8A+QD/gAnef0Nb1EBER2ZI0G1ikLt5cT7pmFoAKd1/bwkkT4FngZuCy3CsmIiLSAZUa
WDTKzIYC44lX/YuACe5+XzquN3A7MADoBLwEjHb3581sIHAL0MnMVqXFHZn+ne7ulZl51AKD3P2Q
dDgBRgPHAXsCBwEzzewU4CxgJ2AecL67P1as3u6+Brg2LW9NuetBREREyuxjYWaHAncQT/LdgeOB
G83sgEz5NwN9gO2AF4DfmFmVu88ATgPmuXuX9PN0C2Z/EnA00AV40cxOBc4HRgDdgLHpvNr8NouI
iEhHVWqLxVgzOzczfLi7zyS2Dlzv7s+k6c+Z2b3ASOAP7r4QWNgwkZmNA84EdgNml1n3q939jfT/
dWZ2JnCRu89K06aZ2VPAMcBmeRtHRERkS1NqYHFJI30sdgEOMrOzM2kVwDMAZtYTmAgMBrYB1qd5
tm1VbTc2v0hdbjKzGzJplcBbOcxLRERESlBuH4sFwF3uflUj4y8Dtge+5u6LzKwaWMHHb0lbX2Sa
VUCFmXV29w/StB2K5CucdgFwobs/2KIlEBERkdyUG1hcB0w2s5nEJywqgL2A4O4O1ACrgWVm1gW4
omD6d4FeZlbj7ivStFeJwcXJZjYJ+AYwjNg/oynXArVmNheYBWxN7DRa5+5zik1gZp2JQc5WQKWZ
bQ2sc/ePiuUXERGRppXVeTN94uJU4CqgjvhUyLXEDpUAFwK9gKXEJ0KeBdZlingSeBx408yWm9mB
7r4SOAE4B6gn9uO4u4S63AZcCUwGlhH7dowHqpqY7FXgfWD/tK7vA7c1Ny8REREpLiRJ0t51+NTp
c8U77b7SDu7Xub2rwJ3DerR3FUREpG2E5rNE+klvERERyY0CCxEREcmNAgsRERHJjQILERERyU3Z
7wrpiG7c43mGDBnS3tUQERHZ7KjFQkRERHKjwEJERERyo8BCREREcqPAQkRERHKjwEJERERyo8BC
REREcqPAQkRERHKjwEJERERyo7ebtsKmeLvppnhbqd4+KiIiOdHbTUVERKTtKbAQERGR3CiwEBER
kdwosBAREZHcKLAQERGR3CiwEBERkdxUNpfBzJ4GBgIfZZIfcPeTy525mY0Cxrl7/3LLKrMeRwDT
gDvyWC4REZGOqtnAInWxu0/YpDVpJTMLQIW7r23l9F2B64E/5VoxERGRDqjUwKJRZjYUGA/0AxYB
E9z9vnRcb+B2YADQCXgJGO3uz5vZQOAWoJOZrUqLOzL9O93dKzPzqAUGufsh6XACjAaOA/YEDgJm
mtkpwFnATsA84Hx3f6yZRZgI3AF8qdUrQURERIAy+1iY2aHEk/JooDtwPHCjmR2QKf9moA+wHfAC
8Bszq3L3GcBpwDx375J+nm7B7E8Cjga6AC+a2anA+cAIoBswNp1Xo7dZzOxbwD7ANS2Yr4iIiDSi
1BaLsWZ2bmb4cHefSWwduN7dn0nTnzOze4GRwB/cfSGwsGEiMxsHnAnsBswus+5Xu/sb6f/rzOxM
4CJ3n5WmTTOzp4BjgE/cxjGzGmAS8D13X2tmZVZHRERESg0sLmmkj8UuwEFmdnYmrQJ4BsDMehJv
NQwGtgHWp3m2bVVtNza/SF1uMrMbMmmVwFuNTH818At3fzGHuoiIiAjl97FYANzl7lc1Mv4yYHvg
a+6+yMyqgRV8/DKT9UWmWQVUmFlnd/8gTduhSL7CaRcAF7r7gyXW/TCga9ovA+ItFczsEHfvW2IZ
IiIiklFuYHEdMNnMZgLPElsr9gKCuztQA6wGlplZF+CKgunfBXqZWY27r0jTXiUGFyeb2STgG8Aw
Yv+MplwL1JrZXGAWsDWx02idu88pkv/rbLz8E4G1wLlF8oqIiEgJyuq8mT5xcSpwFVBHfCrkWtKr
f+BCoBewlPhEyLPAukwRTwKPA2+a2XIzO9DdVwInAOcA9cR+HHeXUJfbgCuBycAyYt+O8UBVI/nf
dfe3Gj7EAGi1u79T+hoQERGRrJAkSXvX4VOnzxXv5L7SDu7XOe8iuXNYj9zLFBGRDik0nyXST3qL
iIhIbhRYiIiISG4UWIiIiEhuFFiIiIhIbsp+V0hHdOMezzNkyJD2roaIiMhmRy0WIiIikhsFFiIi
IpIbBRYiIiKSGwUWIiIikhsFFiIiIpIbBRYiIiKSGwUWIiIikhsFFiIiIm2otraWY489tr2rscno
B7Ja4YzZAzhjdvNvVy/1jaV6C6mISPn6XNH893I5Fpy/Q8l577//fiZOnMicOXOorq5mn332YezY
sQwaNGgT1rC48ePHM2XKFP76178ybtw4amtrN+n81GIhIiKSo4kTJzJ69GjGjBnDe++9x8KFCzn9
9NOZOnVqu9Snf//+XHnllXz7299uk/kpsBAREclJfX09F1xwATfddBPf/e53+dznPkdVVRVDhgzh
qquuKjrN8OHD2W677ejatSsHHHAAr7zyyoZx06ZNY4899qC6upodd9yRq6++GoC6ujqOPPJIttlm
G7p3787+++/P+vXri5Z//PHHc8QRR1BdXZ3/AhehwEJERCQnM2bMYM2aNRx11FElT3PEEUcwd+5c
Fi9ezL777suIESM2jDvppJO49dZbWblyJS+//DLf/OY3Abjmmmvo3bs3S5Ys4b333uPSSy8lhJD7
8rSG+liIiIjkZOnSpfTs2ZPKytJPryeeeOKG/2tra+nWrRv19fV07dqVqqoqZs+ezd577023bt3o
1q0bAFVVVSxatIgFCxbQv39/9t9//9yXpbXUYiEiIpKTHj16UFdXx9q1a0vKv27dOn7yk5/Qr18/
ampq6Nu3LxBvdQD8+te/Ztq0afTp04cDDzyQGTNmAHDeeefRv39/DjvsMHbddVcuv/zyTbI8raHA
QkREJCcDBw5k6623ZsqUKSXlv//++5k6dSrTp0+nvr6e+fPnA5AkCQD77bcfU6dOZfHixQwdOpTv
f//7AFRXV3PNNdcwb948Hn74YSZOnMgTTzyxSZappZptqzGzp4GBwEeZ5Afc/eRyZ25mo4Bx7t6/
3LJaMe/dgbuB3YAq4C3gOnf/aVvXRUREtgxdu3bloosu4kc/+hGVlZUcdthhVFVVMX36dJ566imu
vPLKjfKvXLmSzp0706NHD1avXs2YMWM2jPvwww958MEHOfLII+natSs1NTVUVFQA8Mgjj/DFL35x
Q0tHRUXFhnGFPvroI9atW8f69etZu3Yta9asoaqqqtH85Sr1JtDF7j5hk9SgTGYWgAp3L63d6WPv
AscB89x9nZl9BZhuZvPd/bHcKyoiIptUS35nYlM6++yz+fznP8+ECRMYMWIE1dXVDBgwgLFjx34i
78iRI3n00UfZcccd6d69OxdffDGTJk3aMP6ee+7hjDPOYN26dey+++7ce++9AMydO5czzjiDJUuW
0K1bN04//XQGDx5ctD6nnHIKd99994bhSy65hMmTJzNq1Khcl7tBaGhuaUzaYjG9scDCzIYC44F+
wCJggrvfl47rDdwODAA6AS8Bo939eTMbCDyVpq9Oizsy/Tvd3Ssz86gFBrn7IelwAowmBgZ7Age5
+0wzOwU4C9gJmAecX2qQYGZ7AdPT+v9XU3n7XPFO0ystpR/IEhGRLUTJj5yU1cfCzA4F7iCe5LsD
xwM3mtkBmfJvBvoA2wEvAL8xsyp3nwGcRmwx6JJ+nm7B7E8Cjga6AC+a2anA+cAIoBswNp1Xk7dZ
zOwlM/uAGPQsBn7egjqIiIhIRqm3Qsaa2bmZ4cPdfSaxdeB6d38mTX/OzO4FRgJ/cPeFwMKGicxs
HHAmsV/D7DLrfrW7v5H+v87MzgQucvdZado0M3sKOAZo9DaOu3/FzKqAA9LPP8qsl4iISIdVamBx
SSO3QnYBDjKzszNpFcAzAGbWE5gIDAa2ARp+FmzbVtV2Y/OL1OUmM7shk1ZJ7JTZJHf/CHjCzL4H
XAD8Rw71ExER6XDK/YGsBcBd7l78d0rhMmB74GvuvsjMqoEVfHyvptjvj64CKsyss7t/kKYV65FT
OO0C4EJ3f7BFS7CxSmJrioiIiLRCuYHFdcBkM5sJPEtsrdgLCO7uQA2xY+YyM+sCXFEw/btALzOr
cfcVadqrxODiZDObBHwDGEbsn9GUa4FaM5sLzAK2JnYarXP3OYWZzexbwHLgRSAB/hk4Fvi/LVh+
ERERySir82b6xMWpwFVAHfGpkGuJHSoBLgR6AUuJnSOfBdZlingSeBx408yWm9mB7r4SOAE4B6gn
9uO4m2a4+23AlcBkYBmxb8d44m9UFFMD3An8HVgC1ALnuPsdJSy6iIiIFNHs46bySXrcVEREOpi2
edxUREREJEuBhYiISBuqra3l2GOPbe9qbDJ6bbqIiGwRTvzV0k1afktuW99///1MnDiROXPmUF1d
zT777MPYsWMZNGjQJqxhcfPnz+eEE07gz3/+MzvvvDM33ngjhxxyyCabn1osREREcjRx4kRGjx7N
mDFjeO+991i4cCGnn346U6dObZf6/OAHP+CrX/0qS5cu5ZJLLmHYsGEsWbJkk81PnTdb4eGHH06G
DBnS3tUQEZGMzaHFor6+nh133JHJkyczfPjwonlqa2t5/fXXN7xQbPjw4TzzzDO8//777L333kya
NIk999wTgGnTpnHuuefyt7/9jZqaGn784x9z7rnnUldXx6hRo/jjH//IVlttxZ577sn//M//sNVW
G7cXvPbaa+y1117U1dVRXV0NwP7778+IESM47bTTWrL46rwpIiLS1mbMmMGaNWs46qijSp7miCOO
YO7cuSxevJh9992XESNGbBh30kknceutt7Jy5UpefvllvvnNbwJwzTXX0Lt3b5YsWcJ7773HpZde
SgifPPe/8sor7LrrrhuCCoC9996bV155pYylbJoCCxERkZwsXbqUnj17UllZehfGE088kerqajp3
7kxtbS2zZs2ivr4egKqqKmbPns2KFSvo1q0b++6774b0RYsWsWDBAqqqqth///2LBharVq2ia9eu
G6V17dqVlStXlrGUTVNgISIikpMePXpQV1fH2rVrS8q/bt06fvKTn9CvXz9qamro27cvAHV1dQD8
+te/Ztq0afTp04cDDzyQGTNmAHDeeefRv39/DjvsMHbddVcuv/zyouV36dKFFStWbJS2YsWKjVow
8qbAQkREJCcDBw5k6623ZsqUKSXlv//++5k6dSrTp0+nvr6e+fPnA9DQ/3G//fZj6tSpLF68mKFD
h/L9738fgOrqaq655hrmzZvHww8/zMSJE3niiSc+Uf6ee+7JvHnzNmqhmDVr1oY+HJuCAgsREZGc
dO3alYsuuogf/ehHTJkyhdWrV/PRRx/xu9/9jn//93//RP6VK1fSuXNnevTowerVqxkzZsyGcR9+
+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
hoPjRwAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[54]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[   3.    ,   28.5   ,    0.    ,    0.    ,    7.2292,    1.    ],
       [   2.    ,   27.    ,    0.    ,    0.    ,   10.5   ,    0.    ],
       [   3.    ,   25.    ,    1.    ,    0.    ,   16.1   ,    0.    ],
       ..., 
       [   1.    ,   25.    ,    0.    ,    0.    ,  221.7792,    1.    ],
       [   3.    ,   12.    ,    1.    ,    0.    ,   11.2417,    1.    ],
       [   2.    ,   36.    ,    0.    ,    0.    ,   10.5   ,    1.    ]])</pre>
</div>

</div>

</div>
</div>