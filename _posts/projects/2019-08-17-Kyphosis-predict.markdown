---
layout: project
title:  "Kyphosis Prediction"
permalink: /kyphosis-prediction-random-forest/
date: 2019-08-17
categories: project
tags: machine-learning case-study
author: Sage Elliott
published: true
github_url: https://github.com/sagecodes/kyphsis-classifier-random-forest
---

Use Random Forest model, sklearn, python and the Kyphosis dataset to predict if the Kyphosis would return after surgery.

## About:

This project / case study is for phase 1 of my [100 days of machine learning code](https://sageelliott.com/100daysofmlcode/) challenge.

This is a homework solution to a section in [Machine Learning Classification Bootcamp in Python](https://www.udemy.com/machine-learning-classification). 

#### Problem Statement:

Predict if Kyphosis will return to patient after corrective spinal surgery



## Technology used:

#### Model(s): 
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)

#### Dataset(s):

- [Kyphosis Dataset](https://www.kaggle.com/abbasit/kyphosis-dataset/metadata)

#### Libraries:

- [Scikit Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
<!--- [numpy](https://www.numpy.org/)-->
- [seaborn](https://seaborn.pydata.org/)

#### Resources:

- [Scikit Learn Random forest classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Scikit Feature Extraction - CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

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
<h3 id="Import-Data-and-libraries">Import Data and libraries<a class="anchor-link" href="#Import-Data-and-libraries">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import Libraries</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="c1"># import numpy as np</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import data</span>
<span class="n">kyphosis_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/kyphosis/kyphosis.csv&#39;</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>absent</td>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>absent</td>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>present</td>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>absent</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>absent</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
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
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">tail</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>



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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76</th>
      <td>present</td>
      <td>157</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>77</th>
      <td>absent</td>
      <td>26</td>
      <td>7</td>
      <td>13</td>
    </tr>
    <tr>
      <th>78</th>
      <td>absent</td>
      <td>120</td>
      <td>2</td>
      <td>13</td>
    </tr>
    <tr>
      <th>79</th>
      <td>present</td>
      <td>42</td>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>80</th>
      <td>absent</td>
      <td>36</td>
      <td>4</td>
      <td>13</td>
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
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(81, 4)</pre>
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
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Age in Months</span>
<span class="n">kyphosis_df</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>



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
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>83.654321</td>
      <td>4.049383</td>
      <td>11.493827</td>
    </tr>
    <tr>
      <th>std</th>
      <td>58.104251</td>
      <td>1.619423</td>
      <td>4.883962</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>26.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>87.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>130.000000</td>
      <td>5.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>206.000000</td>
      <td>10.000000</td>
      <td>18.000000</td>
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
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">],</span> <span class="n">label</span> <span class="o">=</span><span class="s1">&#39;Count&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a0caa9438&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="k">import</span> <span class="n">LabelEncoder</span><span class="p">,</span> <span class="n">OneHotEncoder</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># change Kyphosis Column into number output</span>
<span class="n">LabelEncoder_y</span> <span class="o">=</span> <span class="n">LabelEncoder</span><span class="p">()</span>
<span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">LabelEncoder_y</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>



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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
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
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_true</span> <span class="o">=</span> <span class="n">kyphosis_df</span><span class="p">[</span><span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_true</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[12]:</div>



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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>59</td>
      <td>6</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>82</td>
      <td>5</td>
      <td>14</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>105</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>96</td>
      <td>3</td>
      <td>12</td>
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
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># 0 is usually equal to false</span>
<span class="n">kyphosis_false</span> <span class="o">=</span> <span class="n">kyphosis_df</span><span class="p">[</span><span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">0</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_false</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>



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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>16</td>
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
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Disease present after operation percentage is&#39;</span><span class="p">,</span>
      <span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">kyphosis_true</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="p">))</span><span class="o">*</span><span class="mi">100</span><span class="p">,</span> <span class="s1">&#39;%&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Disease present after operation percentage is 20.98765432098765 %
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Disease not present after operation percentage is&#39;</span><span class="p">,</span>
      <span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">kyphosis_false</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="p">))</span><span class="o">*</span><span class="mi">100</span><span class="p">,</span> <span class="s1">&#39;%&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Disease not present after operation percentage is 79.01234567901234 %
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">corr</span><span class="p">(),</span> <span class="n">annot</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a151f24e0&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">pairplot</span><span class="p">(</span><span class="n">kyphosis_df</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">,</span>
             <span class="nb">vars</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="s1">&#39;Number&#39;</span><span class="p">,</span> <span class="s1">&#39;Start&#39;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;seaborn.axisgrid.PairGrid at 0x1a152e69b0&gt;</pre>
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
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Data-Prep">Data Prep<a class="anchor-link" href="#Data-Prep">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">kyphosis_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>Kyphosis</th>
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
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
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">kyphosis_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[21]:</div>



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
      <th>Age</th>
      <th>Number</th>
      <th>Start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>158</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>128</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>15</td>
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
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">kyphosis_df</span><span class="p">[</span><span class="s1">&#39;Kyphosis&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[23]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    0
1    0
2    1
3    0
4    0
Name: Kyphosis, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Create train Test Split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span>
                                                    <span class="n">y</span><span class="p">,</span>
                                                    <span class="n">test_size</span><span class="o">=</span><span class="mf">0.20</span><span class="p">,</span>
                                                    <span class="n">random_state</span> <span class="o">=</span> <span class="mi">2</span><span class="p">,</span>
                                                    <span class="n">stratify</span><span class="o">=</span><span class="n">y</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Train-Model">Train Model<a class="anchor-link" href="#Train-Model">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Decision tree</span>
<span class="kn">from</span> <span class="nn">sklearn.tree</span> <span class="k">import</span> <span class="n">DecisionTreeClassifier</span>
<span class="n">decision_tree</span> <span class="o">=</span> <span class="n">DecisionTreeClassifier</span><span class="p">()</span>
<span class="n">decision_tree</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>DecisionTreeClassifier(class_weight=None, criterion=&#39;gini&#39;, max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter=&#39;best&#39;)</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Find feature importance</span>
<span class="n">feature_importance</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">decision_tree</span><span class="o">.</span><span class="n">feature_importances_</span><span class="p">,</span>
                                  <span class="n">index</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">,</span>
                                  <span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;importance&#39;</span><span class="p">]</span> <span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">feature_importance</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[27]:</div>



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
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.476436</td>
    </tr>
    <tr>
      <th>Number</th>
      <td>0.254160</td>
    </tr>
    <tr>
      <th>Start</th>
      <td>0.269404</td>
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
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Sorted Feature Importance</span>
<span class="n">feature_importance</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span>
    <span class="n">decision_tree</span><span class="o">.</span><span class="n">feature_importances_</span><span class="p">,</span>
    <span class="n">index</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">,</span>
    <span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;importance&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="s1">&#39;importance&#39;</span><span class="p">,</span>
    <span class="n">ascending</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">feature_importance</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[29]:</div>



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
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0.476436</td>
    </tr>
    <tr>
      <th>Start</th>
      <td>0.269404</td>
    </tr>
    <tr>
      <th>Number</th>
      <td>0.254160</td>
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
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">classification_report</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict</span> <span class="o">=</span> <span class="n">decision_tree</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
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
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span> <span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[33]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a15d50898&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_predict</span><span class="p">))</span>
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

          0       0.85      0.85      0.85        13
          1       0.50      0.50      0.50         4

avg / total       0.76      0.76      0.76        17

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">randomforest_classifier</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span>
                            <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">500</span><span class="p">,</span>
                            <span class="n">criterion</span> <span class="o">=</span><span class="s1">&#39;entropy&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[62]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">randomforest_classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[62]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>RandomForestClassifier(bootstrap=True, class_weight=None, criterion=&#39;entropy&#39;,
            max_depth=None, max_features=&#39;auto&#39;, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[63]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict_forest</span> <span class="o">=</span> <span class="n">randomforest_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[64]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_predict_forest</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[65]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a162d40f0&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_predict_forest</span><span class="p">))</span>
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

          0       1.00      0.92      0.96        13
          1       0.80      1.00      0.89         4

avg / total       0.95      0.94      0.94        17

</pre>
</div>
</div>

</div>
</div>

</div>