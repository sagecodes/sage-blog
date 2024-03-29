---
layout: project
title:  "Crime Forecast"
permalink: /chicago-crime-prophet/
date: 2019-08-20
categories: project
tags: machine-learning case-study
author: Sage Elliott
img: img\projects\crime_forecast_fb_pro.png
published: true
github_url: https://github.com/sagecodes/chicago-crime-prediction-fbprophet
---

Use Facebook prophet, sklearn, python and the Chicago crime dataset to predict crime rates in Chicago.

## About:

This project / case study is for phase 1 of my [100 days of machine learning code](https://sageelliott.com/100daysofmlcode/) challenge.

This is a homework solution to a section in [Deep Learning and Machine Learning Practicakl Workouts](https://www.udemy.com/course/deep-learning-machine-learning-practical). 

#### Problem Statement:

Forecast Chicago crime rate 


## Technology used:

#### Dataset(s):

- [Crimes in chicago](https://www.kaggle.com/currie32/crimes-in-chicago)

#### Libraries:

- [Facebook Prophet](https://github.com/facebook/prophets)
<!--- [Scikit Learn](https://scikit-learn.org/stable/)
-->- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
<!--- [numpy](https://www.numpy.org/)-->
- [seaborn](https://seaborn.pydata.org/)

#### Resources:

- [Forecasting at Scale with FB prophet](https://facebook.github.io/prophet/)

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
<p>Use fbprophet to predict crime in chicago</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Import-data">Import data<a class="anchor-link" href="#Import-data">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#import libraries</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="kn">from</span> <span class="nn">fbprophet</span> <span class="k">import</span> <span class="n">Prophet</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import csv data into dataframes</span>
<span class="n">chicago_df_1</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/chicago/Chicago_Crimes_2005_to_2007.csv&#39;</span><span class="p">,</span>
                           <span class="n">error_bad_lines</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
<span class="n">chicago_df_2</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/chicago/Chicago_Crimes_2008_to_2011.csv&#39;</span><span class="p">,</span>
                           <span class="n">error_bad_lines</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
<span class="n">chicago_df_3</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/chicago/Chicago_Crimes_2012_to_2017.csv&#39;</span><span class="p">,</span>
                           <span class="n">error_bad_lines</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
                        
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>b&#39;Skipping line 533719: expected 23 fields, saw 24\n&#39;
b&#39;Skipping line 1149094: expected 23 fields, saw 41\n&#39;
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">chicago_df_1</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">chicago_df_2</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">chicago_df_3</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(1872343, 23)
(2688710, 23)
(1456714, 23)
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Concatinate Data Frames</span>
<span class="n">chicago_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">chicago_df_1</span><span class="p">,</span>
                        <span class="n">chicago_df_2</span><span class="p">,</span>
                        <span class="n">chicago_df_3</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(6017767, 23)</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Case Number</th>
      <th>Date</th>
      <th>Block</th>
      <th>IUCR</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>...</th>
      <th>Ward</th>
      <th>Community Area</th>
      <th>FBI Code</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Updated On</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4673626</td>
      <td>HM274058</td>
      <td>04/02/2006 01:00:00 PM</td>
      <td>055XX N MANGO AVE</td>
      <td>2825</td>
      <td>OTHER OFFENSE</td>
      <td>HARASSMENT BY TELEPHONE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>...</td>
      <td>45.0</td>
      <td>11.0</td>
      <td>26</td>
      <td>1136872.0</td>
      <td>1936499.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.981913</td>
      <td>-87.771996</td>
      <td>(41.981912692, -87.771996382)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4673627</td>
      <td>HM202199</td>
      <td>02/26/2006 01:40:48 PM</td>
      <td>065XX S RHODES AVE</td>
      <td>2017</td>
      <td>NARCOTICS</td>
      <td>MANU/DELIVER:CRACK</td>
      <td>SIDEWALK</td>
      <td>True</td>
      <td>...</td>
      <td>20.0</td>
      <td>42.0</td>
      <td>18</td>
      <td>1181027.0</td>
      <td>1861693.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.775733</td>
      <td>-87.611920</td>
      <td>(41.775732538, -87.611919814)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4673628</td>
      <td>HM113861</td>
      <td>01/08/2006 11:16:00 PM</td>
      <td>013XX E 69TH ST</td>
      <td>051A</td>
      <td>ASSAULT</td>
      <td>AGGRAVATED: HANDGUN</td>
      <td>OTHER</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>69.0</td>
      <td>04A</td>
      <td>1186023.0</td>
      <td>1859609.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.769897</td>
      <td>-87.593671</td>
      <td>(41.769897392, -87.593670899)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4673629</td>
      <td>HM274049</td>
      <td>04/05/2006 06:45:00 PM</td>
      <td>061XX W NEWPORT AVE</td>
      <td>0460</td>
      <td>BATTERY</td>
      <td>SIMPLE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>...</td>
      <td>38.0</td>
      <td>17.0</td>
      <td>08B</td>
      <td>1134772.0</td>
      <td>1922299.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.942984</td>
      <td>-87.780057</td>
      <td>(41.942984005, -87.780056951)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4673630</td>
      <td>HM187120</td>
      <td>02/17/2006 09:03:14 PM</td>
      <td>037XX W 60TH ST</td>
      <td>1811</td>
      <td>NARCOTICS</td>
      <td>POSS: CANNABIS 30GMS OR LESS</td>
      <td>ALLEY</td>
      <td>True</td>
      <td>...</td>
      <td>13.0</td>
      <td>65.0</td>
      <td>18</td>
      <td>1152412.0</td>
      <td>1864560.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.784211</td>
      <td>-87.716745</td>
      <td>(41.784210853, -87.71674491)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">tail</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[15]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Case Number</th>
      <th>Date</th>
      <th>Block</th>
      <th>IUCR</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>...</th>
      <th>Ward</th>
      <th>Community Area</th>
      <th>FBI Code</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Updated On</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1456709</th>
      <td>6250330</td>
      <td>10508679</td>
      <td>HZ250507</td>
      <td>05/03/2016 11:33:00 PM</td>
      <td>026XX W 23RD PL</td>
      <td>0486</td>
      <td>BATTERY</td>
      <td>DOMESTIC BATTERY SIMPLE</td>
      <td>APARTMENT</td>
      <td>True</td>
      <td>...</td>
      <td>28.0</td>
      <td>30.0</td>
      <td>08B</td>
      <td>1159105.0</td>
      <td>1888300.0</td>
      <td>2016</td>
      <td>05/10/2016 03:56:50 PM</td>
      <td>41.849222</td>
      <td>-87.691556</td>
      <td>(41.849222028, -87.69155551)</td>
    </tr>
    <tr>
      <th>1456710</th>
      <td>6251089</td>
      <td>10508680</td>
      <td>HZ250491</td>
      <td>05/03/2016 11:30:00 PM</td>
      <td>073XX S HARVARD AVE</td>
      <td>1310</td>
      <td>CRIMINAL DAMAGE</td>
      <td>TO PROPERTY</td>
      <td>APARTMENT</td>
      <td>True</td>
      <td>...</td>
      <td>17.0</td>
      <td>69.0</td>
      <td>14</td>
      <td>1175230.0</td>
      <td>1856183.0</td>
      <td>2016</td>
      <td>05/10/2016 03:56:50 PM</td>
      <td>41.760744</td>
      <td>-87.633335</td>
      <td>(41.760743949, -87.63333531)</td>
    </tr>
    <tr>
      <th>1456711</th>
      <td>6251349</td>
      <td>10508681</td>
      <td>HZ250479</td>
      <td>05/03/2016 12:15:00 AM</td>
      <td>024XX W 63RD ST</td>
      <td>041A</td>
      <td>BATTERY</td>
      <td>AGGRAVATED: HANDGUN</td>
      <td>SIDEWALK</td>
      <td>False</td>
      <td>...</td>
      <td>15.0</td>
      <td>66.0</td>
      <td>04B</td>
      <td>1161027.0</td>
      <td>1862810.0</td>
      <td>2016</td>
      <td>05/10/2016 03:56:50 PM</td>
      <td>41.779235</td>
      <td>-87.685207</td>
      <td>(41.779234743, -87.685207125)</td>
    </tr>
    <tr>
      <th>1456712</th>
      <td>6253257</td>
      <td>10508690</td>
      <td>HZ250370</td>
      <td>05/03/2016 09:07:00 PM</td>
      <td>082XX S EXCHANGE AVE</td>
      <td>0486</td>
      <td>BATTERY</td>
      <td>DOMESTIC BATTERY SIMPLE</td>
      <td>SIDEWALK</td>
      <td>False</td>
      <td>...</td>
      <td>7.0</td>
      <td>46.0</td>
      <td>08B</td>
      <td>1197261.0</td>
      <td>1850727.0</td>
      <td>2016</td>
      <td>05/10/2016 03:56:50 PM</td>
      <td>41.745252</td>
      <td>-87.552773</td>
      <td>(41.745251975, -87.552773464)</td>
    </tr>
    <tr>
      <th>1456713</th>
      <td>6253474</td>
      <td>10508692</td>
      <td>HZ250517</td>
      <td>05/03/2016 11:38:00 PM</td>
      <td>001XX E 75TH ST</td>
      <td>5007</td>
      <td>OTHER OFFENSE</td>
      <td>OTHER WEAPONS VIOLATION</td>
      <td>PARKING LOT/GARAGE(NON.RESID.)</td>
      <td>True</td>
      <td>...</td>
      <td>6.0</td>
      <td>69.0</td>
      <td>26</td>
      <td>1178696.0</td>
      <td>1855324.0</td>
      <td>2016</td>
      <td>05/10/2016 03:56:50 PM</td>
      <td>41.758309</td>
      <td>-87.620658</td>
      <td>(41.75830866, -87.620658418)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
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
<h3 id="Explore-Data">Explore Data<a class="anchor-link" href="#Explore-Data">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">20</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Case Number</th>
      <th>Date</th>
      <th>Block</th>
      <th>IUCR</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>...</th>
      <th>Ward</th>
      <th>Community Area</th>
      <th>FBI Code</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Updated On</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4673626</td>
      <td>HM274058</td>
      <td>04/02/2006 01:00:00 PM</td>
      <td>055XX N MANGO AVE</td>
      <td>2825</td>
      <td>OTHER OFFENSE</td>
      <td>HARASSMENT BY TELEPHONE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>...</td>
      <td>45.0</td>
      <td>11.0</td>
      <td>26</td>
      <td>1136872.0</td>
      <td>1936499.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.981913</td>
      <td>-87.771996</td>
      <td>(41.981912692, -87.771996382)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4673627</td>
      <td>HM202199</td>
      <td>02/26/2006 01:40:48 PM</td>
      <td>065XX S RHODES AVE</td>
      <td>2017</td>
      <td>NARCOTICS</td>
      <td>MANU/DELIVER:CRACK</td>
      <td>SIDEWALK</td>
      <td>True</td>
      <td>...</td>
      <td>20.0</td>
      <td>42.0</td>
      <td>18</td>
      <td>1181027.0</td>
      <td>1861693.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.775733</td>
      <td>-87.611920</td>
      <td>(41.775732538, -87.611919814)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4673628</td>
      <td>HM113861</td>
      <td>01/08/2006 11:16:00 PM</td>
      <td>013XX E 69TH ST</td>
      <td>051A</td>
      <td>ASSAULT</td>
      <td>AGGRAVATED: HANDGUN</td>
      <td>OTHER</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>69.0</td>
      <td>04A</td>
      <td>1186023.0</td>
      <td>1859609.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.769897</td>
      <td>-87.593671</td>
      <td>(41.769897392, -87.593670899)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4673629</td>
      <td>HM274049</td>
      <td>04/05/2006 06:45:00 PM</td>
      <td>061XX W NEWPORT AVE</td>
      <td>0460</td>
      <td>BATTERY</td>
      <td>SIMPLE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>...</td>
      <td>38.0</td>
      <td>17.0</td>
      <td>08B</td>
      <td>1134772.0</td>
      <td>1922299.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.942984</td>
      <td>-87.780057</td>
      <td>(41.942984005, -87.780056951)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4673630</td>
      <td>HM187120</td>
      <td>02/17/2006 09:03:14 PM</td>
      <td>037XX W 60TH ST</td>
      <td>1811</td>
      <td>NARCOTICS</td>
      <td>POSS: CANNABIS 30GMS OR LESS</td>
      <td>ALLEY</td>
      <td>True</td>
      <td>...</td>
      <td>13.0</td>
      <td>65.0</td>
      <td>18</td>
      <td>1152412.0</td>
      <td>1864560.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.784211</td>
      <td>-87.716745</td>
      <td>(41.784210853, -87.71674491)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>4673631</td>
      <td>HM263167</td>
      <td>03/30/2006 10:30:00 PM</td>
      <td>014XX W 73RD PL</td>
      <td>0560</td>
      <td>ASSAULT</td>
      <td>SIMPLE</td>
      <td>APARTMENT</td>
      <td>True</td>
      <td>...</td>
      <td>17.0</td>
      <td>67.0</td>
      <td>08A</td>
      <td>1167688.0</td>
      <td>1855998.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.760401</td>
      <td>-87.660982</td>
      <td>(41.760401372, -87.660982392)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>4673632</td>
      <td>HM273234</td>
      <td>04/05/2006 12:10:00 PM</td>
      <td>050XX N LARAMIE AVE</td>
      <td>0460</td>
      <td>BATTERY</td>
      <td>SIMPLE</td>
      <td>SCHOOL, PUBLIC, BUILDING</td>
      <td>True</td>
      <td>...</td>
      <td>45.0</td>
      <td>11.0</td>
      <td>08B</td>
      <td>1140791.0</td>
      <td>1932993.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.972221</td>
      <td>-87.757670</td>
      <td>(41.972220564, -87.75766982)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>4673633</td>
      <td>HM275105</td>
      <td>04/05/2006 03:00:00 PM</td>
      <td>067XX S ROCKWELL ST</td>
      <td>0820</td>
      <td>THEFT</td>
      <td>$500 AND UNDER</td>
      <td>STREET</td>
      <td>False</td>
      <td>...</td>
      <td>15.0</td>
      <td>66.0</td>
      <td>06</td>
      <td>1160205.0</td>
      <td>1859776.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.770926</td>
      <td>-87.688304</td>
      <td>(41.770925978, -87.688304107)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4673634</td>
      <td>HM275063</td>
      <td>04/05/2006 09:30:00 PM</td>
      <td>019XX W CHICAGO AVE</td>
      <td>0560</td>
      <td>ASSAULT</td>
      <td>SIMPLE</td>
      <td>PARKING LOT/GARAGE(NON.RESID.)</td>
      <td>False</td>
      <td>...</td>
      <td>32.0</td>
      <td>24.0</td>
      <td>08A</td>
      <td>1163122.0</td>
      <td>1905349.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.895923</td>
      <td>-87.676334</td>
      <td>(41.895922672, -87.676333733)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>4673635</td>
      <td>HM268513</td>
      <td>04/03/2006 03:00:00 AM</td>
      <td>063XX S EBERHART AVE</td>
      <td>0486</td>
      <td>BATTERY</td>
      <td>DOMESTIC BATTERY SIMPLE</td>
      <td>SIDEWALK</td>
      <td>False</td>
      <td>...</td>
      <td>20.0</td>
      <td>42.0</td>
      <td>08B</td>
      <td>1180669.0</td>
      <td>1863047.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.779456</td>
      <td>-87.613191</td>
      <td>(41.77945628, -87.613190628)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>4673637</td>
      <td>HM275073</td>
      <td>04/06/2006 11:15:00 AM</td>
      <td>0000X N LA SALLE ST</td>
      <td>0810</td>
      <td>THEFT</td>
      <td>OVER $500</td>
      <td>STREET</td>
      <td>False</td>
      <td>...</td>
      <td>42.0</td>
      <td>32.0</td>
      <td>06</td>
      <td>1175135.0</td>
      <td>1900412.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.882114</td>
      <td>-87.632361</td>
      <td>(41.882114362, -87.632361012)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>4673639</td>
      <td>HM272124</td>
      <td>04/04/2006 08:15:00 PM</td>
      <td>029XX S FEDERAL ST</td>
      <td>1350</td>
      <td>CRIMINAL TRESPASS</td>
      <td>TO STATE SUP LAND</td>
      <td>CHA HALLWAY/STAIRWELL/ELEVATOR</td>
      <td>True</td>
      <td>...</td>
      <td>3.0</td>
      <td>35.0</td>
      <td>26</td>
      <td>1176025.0</td>
      <td>1885766.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.841905</td>
      <td>-87.629534</td>
      <td>(41.841904764, -87.629533842)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>4673640</td>
      <td>HM275082</td>
      <td>04/06/2006 11:30:00 AM</td>
      <td>017XX E 86TH PL</td>
      <td>0935</td>
      <td>MOTOR VEHICLE THEFT</td>
      <td>THEFT/RECOVERY: TRUCK,BUS,MHOME</td>
      <td>STREET</td>
      <td>False</td>
      <td>...</td>
      <td>8.0</td>
      <td>45.0</td>
      <td>07</td>
      <td>1189375.0</td>
      <td>1847970.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.737879</td>
      <td>-87.581757</td>
      <td>(41.737879171, -87.581756795)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>4673642</td>
      <td>HM202299</td>
      <td>02/26/2006 02:47:21 PM</td>
      <td>002XX S LEAMINGTON AVE</td>
      <td>1811</td>
      <td>NARCOTICS</td>
      <td>POSS: CANNABIS 30GMS OR LESS</td>
      <td>SIDEWALK</td>
      <td>True</td>
      <td>...</td>
      <td>28.0</td>
      <td>25.0</td>
      <td>18</td>
      <td>1142168.0</td>
      <td>1898610.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.877845</td>
      <td>-87.753461</td>
      <td>(41.87784456, -87.753461293)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>4673643</td>
      <td>HM270077</td>
      <td>04/03/2006 08:09:00 PM</td>
      <td>073XX S WOODLAWN AVE</td>
      <td>0420</td>
      <td>BATTERY</td>
      <td>AGGRAVATED:KNIFE/CUTTING INSTR</td>
      <td>SIDEWALK</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>69.0</td>
      <td>04B</td>
      <td>1185483.0</td>
      <td>1856655.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.761804</td>
      <td>-87.595743</td>
      <td>(41.761804069, -87.595743133)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>4673644</td>
      <td>HM187135</td>
      <td>02/17/2006 09:26:33 PM</td>
      <td>052XX S FAIRFIELD AVE</td>
      <td>1811</td>
      <td>NARCOTICS</td>
      <td>POSS: CANNABIS 30GMS OR LESS</td>
      <td>STREET</td>
      <td>True</td>
      <td>...</td>
      <td>14.0</td>
      <td>63.0</td>
      <td>18</td>
      <td>1158926.0</td>
      <td>1869898.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.798728</td>
      <td>-87.692716</td>
      <td>(41.798728387, -87.692716037)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>4673645</td>
      <td>HM272962</td>
      <td>04/05/2006 08:00:00 AM</td>
      <td>024XX W HARRISON ST</td>
      <td>0935</td>
      <td>MOTOR VEHICLE THEFT</td>
      <td>THEFT/RECOVERY: TRUCK,BUS,MHOME</td>
      <td>STREET</td>
      <td>False</td>
      <td>...</td>
      <td>2.0</td>
      <td>28.0</td>
      <td>07</td>
      <td>1160214.0</td>
      <td>1897293.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.873877</td>
      <td>-87.687237</td>
      <td>(41.873876903, -87.687236966)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>4673646</td>
      <td>HM263178</td>
      <td>03/31/2006 08:20:00 AM</td>
      <td>067XX S PERRY AVE</td>
      <td>0486</td>
      <td>BATTERY</td>
      <td>DOMESTIC BATTERY SIMPLE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>...</td>
      <td>6.0</td>
      <td>69.0</td>
      <td>08B</td>
      <td>1176558.0</td>
      <td>1860293.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.771992</td>
      <td>-87.628345</td>
      <td>(41.771992493, -87.628344689)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>4673648</td>
      <td>HM273363</td>
      <td>04/05/2006 01:30:00 PM</td>
      <td>046XX N MILWAUKEE AVE</td>
      <td>1330</td>
      <td>CRIMINAL TRESPASS</td>
      <td>TO LAND</td>
      <td>PARK PROPERTY</td>
      <td>True</td>
      <td>...</td>
      <td>45.0</td>
      <td>15.0</td>
      <td>26</td>
      <td>1140652.0</td>
      <td>1930355.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.964984</td>
      <td>-87.758246</td>
      <td>(41.96498423, -87.758246066)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>4673649</td>
      <td>HM263200</td>
      <td>03/31/2006 05:00:00 AM</td>
      <td>062XX S RACINE AVE</td>
      <td>0810</td>
      <td>THEFT</td>
      <td>OVER $500</td>
      <td>SCHOOL, PUBLIC, GROUNDS</td>
      <td>False</td>
      <td>...</td>
      <td>16.0</td>
      <td>67.0</td>
      <td>06</td>
      <td>1169388.0</td>
      <td>1863488.0</td>
      <td>2006</td>
      <td>04/15/2016 08:55:02 AM</td>
      <td>41.780918</td>
      <td>-87.654535</td>
      <td>(41.780918241, -87.654535186)</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 23 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot missing information </span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span> <span class="n">cbar</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">cmap</span> <span class="o">=</span> <span class="s1">&#39;YlGnBu&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x27e7b510da0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnMAAAKfCAYAAAAFNk73AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3XvYpWPdx//3J2Mf2RdGjcomxGBM2mAamyQPKWLaoNJEFHoUk0p4+v2UyiZJHvuIZIwku/l5YiR7ZsaMmRDTGDM1SSVkM3x+f1znMpdlrfu+zbqldd+f13HMca/rvM7rXOty+OM8rvM6vx/ZJiIiIiK60+te6x8QEREREYsuk7mIiIiILpbJXEREREQXy2QuIiIiootlMhcRERHRxTKZi4iIiOhiXT2Zk7SjpN9LekDSEa/174mIiIj4d1O31pmTtBhwH7A9MAe4HRhj+97X9IdFRERE/Bt185O5kcADth+0/SxwEbDra/ybIiIiIv6tunkytybwcO14TmmLiIiIGDSGvNY/oANq0fayNWNJY4GxAD/5yTGbjx2756v4k9alWvnN+K/N+P+O78j4r/13ZPzX/ju6ffx/h27/b5T/T/8zxm8513mZbp7MzQHWqh0PBeY2d7J9OnB6dXRfd74gGBEREdFGNy+z3g6sI2ltSUsAewGXv8a/KSIiIuLfqmufzNleIOkg4BpgMeAs29Nf458VERER8W/VtZM5ANtXAle+1r8jIiIi4rXSzcusEREREYNeR5M5SWdJmi9pWq1tJUkTJd1f/q5Y2leUNEHSVEm3SdqoaazFJN0t6Ypa20El3cGSVmnx/VtIel7S7p3cR0RERES36vTJ3DnAjk1tRwDX2V4HuK4cA3wNmGx7Y2Bv4KSm6w4GZjS13QRsB/yx+YtLAsR3qN6Zi4iIiBiUOprM2Z4EPNbUvCtwbvl8LvDh8nkDqskdtmcCwyS9EUDSUOBDwBlN499te1abr/8iMB6Y38k9RERERHSzV+OduTfangdQ/q5W2qcAHwGQNBJ4C1VtOIATga8CL/TlCyStCewGnNZ/PzsiIiKi+/w7N0AcB6woaTLVU7W7gQWSdgbm277zFYx1InC47ed76yhprKQ7JN1x+uk/X6QfHhEREfGf6tUoTfJnSavbnidpdcoyqO3HgU8DSBLwUPm3F7CLpJ2ApYDlJZ1v+5M9fMcI4KJqGFYBdpK0wPZlzR2TABERERED2avxZO5yYJ/yeR/glwCSVihJDQD7AZNsP257nO2htodRTez+r5eJHLbXtj2sXHMJ8IVWE7mIiIiIga7T0iQXAjcD60maI+mzVMup20u6H9i+HAO8A5guaSbwQardq72N/yVJc6jerZsq6YzeromIiIgYTDpaZrU9ps2pbVv0vRlYp5fxrgeurx2fDJzcyzX79vIzIyIiIgasJEBEREREdLFOl1nXkvQbSTMkTZd0cGnfoxy/IGlErf/iks6VdE+5Zlzt3KHlmmmSLpS0VGm/QNLvS/tZkhYv7W+Q9CtJU8p1n+7kXiIiIiK6UadP5hYA/237HcCWwIGSNgCmUdWUm9TUfw9gSdvvBDYHPi9pWKkb9yVghO2NgMWoNkMAXACsD7wTWJpq8wTAgcC9tjcBRgHfr22wiIiIiBgUOn1nbh7QKBD8T0kzgDVtTwQopUNecgmwrKQhVBOzZ4HHy+chwNKSngOWAeaWca9sXCzpNhYWGjawXClz8nqqJIoFndxPRERERLfpt3fmJA0DNgVu7aHbJcCTVBPA2cD3bD9m+xHge6VtHvAP29c2jb848Cng6tJ0CtUO2bnAPcDBtvuUIBERERExUPTLZE7S66lyUg8pxYHbGQk8D6wBrA38t6S3SlqRKtN17XJuWUnNteZOpapNd2M5/gAwufQfDpwiafkWvy0JEBERETFgdZwAUZ6YjQcusH1pL90/Dlxt+zlgvqSbqNIcDDxk+y9lzEuB9wDnl+OjgFWBz9fG+jRwnG0DD0h6iOrdutvqX5gEiIiIiBjIOt3NKuBMYIbtH/ThktnAaFWWpdo0MbO0bylpmTLmtsCM8h37UT2FG9O0jDq79EPSG4H1gAc7uZ+IiIiIbtPpk7n3Ur3Hdo+kyaXta8CSwA+pnqb9WtJk2x8AfgScTbXbVcDZtqcCSLoEuItqE8PdvPg0jdOAPwI3lw0Vl9o+BjgWOEfSPWWsw20/2uH9RERERHSVTnez/pZqItXKhBb9n6AqT9JqrKOAo1q0t/yNtucCO/T5x0ZEREQMQEmAiIiIiOhinb4zt5Sk22opDEeX9leU2iBpuKSbS9tUSXu2+K4fSnqidvxlSfeW/tdJeksn9xIRERHRjTp9MvcMMLqkMAwHdpS0Ja88teEpYG/bGwI7AidKWqHxJSUS7MXj4m6qxIiNqerXfbfDe4mIiIjoOh1N5lxpPC1bvPyz7SvLOVOVCukxtcH2fbbvL2POBeZTbZ5A0mLA8cBXm777N7afKoe31L4jIiIiYtDo+J05SYuVnazzgYm2b62de8WpDZJGAksAfyhNBwGXl+iwdj4LXNXm96VocERERAxYHRcNtv08MLwsi06QtJHtaeV0u9SG0cDbgImSbmykRkhaHfgpsI/tFyStQbX7dVS77y9JESOAbdr8vhQNjoiIiAGr33az2v47cD3VO2/11IYv17p9mqpOnG0/ADRSGyhRXL8Gvm77ltJ/U+DtVAkPs4BlJD3QGEzSdsCRwC62n+mve4mIiIjoFp3uZl21sVFB0tLAdsDMV5raUDZBTADOs/2LRmfbv7b9JtvDbA8DnrL99nL9psBPqCZy8zu5j4iIiIhu1eky6+rAuWWTwuuAi21fIWkBryC1oSyVbg2sLGnfMva+tifT3vFUmyh+Ub5jtu1dOryfiIiIiK7SaQLEVKql0Ob2V5TaYPt84Pw+fN/ra5+3e0U/NiIiImIASgJERERERBfrr9Ikd0u6ohyfWRIepkq6RNLrS/sJkiaXf/dJ+nttjDdLulbSjJLqMKy0nyPpodp1w2vXjCpt0yXd0Ol9RERERHSjjkuTAAcDM4Dly/GhtVIjP6CqE3ec7UMbF0j6Ii9dnj0P+LbtiWXyV9808RXbl9S/sGy6OBXY0fZsSav1w31EREREdJ1Od7MOBT4EnNFoq03kRBXl1aq22xjgwtJvA2CI7Ynl+idqyQ7tfJxqU8Xsck12s0ZERMSg1Oky64lUMVvNKQ5nA3+iqiH3w6ZzbwHWBv6vNK0L/F3SpWW59viyO7bh22XJ9gRJS9auWVHS9ZLulLR3ux+YBIiIiIgYyBZ5MidpZ2C+7Tubz9n+NLAG1fLrnk2n9wIuKckRUC31bgUcBmwBvBXYt5wbRzUh3AJYCTi8ds3mVE8FPwB8Q9K6rX6n7dNtj7A9YuzY5p8SERER0d06eTL3XmCXksxwETBa0ovlRcpk7efAR5uu24uyxFrMAe62/aDtBcBlwGZljHklLeIZ4GxgZO2aq20/aftRYBKwSQf3EhEREdGVFnkyZ3uc7aElmWEvqmXTT0lqJDQI+C9gZuMaSesBKwI314a6nWrJdNVyPBq4t/RfvTbWh4FG5usvga0kDZG0DPAuqqeAEREREYNKf+xmrRNVIsTy5fMU4IDa+THARbZf3BRh+3lJhwHXlUnbncD/ltMXlEmegMnA/uWaGZKuBqZSva93hu1pRERERAwy/TKZs309cH05fG8P/b7Vpn0isHGL9tE9jHU8VaRXRERExKCVBIiIiIiILtYfCRCzJN1T0hjuaDp3mCRLWqUcf6WW5jBN0vOSVirndpT0e0kPSDqiNsaNtWvmSrqstEvSyaX/VEmbdXovEREREd2mv96Ze3/ZVfoiSWsB2wOzG231pVFJ/0WVFvFYqSv3o9J/DnC7pMtt32t7q9qY46k2PwB8EFin/HsX8OPyNyIiImLQeDWXWU+gKijcKgECaikQVCVHHijlSZ6lKnWya72zpOWodrpeVpp2Bc4rpUtuAVZo7H6NiIiIGCz6YzJn4NqSxDAWQNIuwCO2p7S6oJQT2REYX5rWBB6udZlT2up2A65rxIX18ZokQERERMSA1h/LrO+1PbeE3U+UNBM4Etihh2v+C7jJ9mPlWC36ND/RG0MtA7aP12D7dOD06ui+dk8JIyIiIrpSx0/mbM8tf+cDE4BtqLJXp5R0iKHAXZLeVLusVQrEWrXjocDcxoGklamWYn/d12siIiIiBoOOJnOSli3vsiFpWaqncbfbXs32sJIOMQfYzPafSr83UE34flkb6nZgHUlrS1qCarJ3ee38HsAVtp+utV0O7F12tW4J/MP2vE7uJyIiIqLbdLrM+kZgQhXcwBDgZ7av7uWa3YBrbT/ZaLC9QNJBwDXAYsBZtqfXrtkLOK5pnCuBnYAHgKeAT3dyIxERERHdqKPJnO0H6SXgvjydqx+fA5zTot+VVBO0VmOMatFm4MC+/taIiIiIgSgJEBERERFdrD8SIFaQdImkmZJmSHq3pG9JeqSW3LBTrf/Gkm6WNL0kRyzVNN7lkqbVjodLuqWRMCFpZFP/LUqSxO6d3ktEREREt+mP0iQnAVfb3r1sXlgG+ABwgu3v1TtKGgKcD3zK9pSyS/W52vmPAE80jf9d4GjbV5VJ4XeBUaX/YsB3qN61i4iIiBh0Ot3NujywNXAmgO1nbf+9h0t2AKY2ignb/qvt58tYrwe+DPxP0zUGli+f38BLy498karw8PxO7iMiIiKiW3W6zPpW4C/A2ZLulnRGKVECcJCkqZLOkrRiaVsXsKRrJN0l6au1sY4Fvk+1M7XuEOB4SQ8D3wPGAUhak2pn7Gk9/cAkQERERMRA1ulkbgiwGfBj25sCTwJHUIXevw0YDsyjmqQ1+r8P+ET5u5ukbSUNB95ue0KL7zgAONT2WsChlKeAwInA4Y0ne+3YPt32CNsjxo7ds4NbjYiIiPjP0+k7c3OAObZvLceXAEfY/nOjg6T/Ba6o9b/B9qPl3JVUk8EngM1LYsQQYDVJ15eSJPsAB5frf8HCSK8RwEWlxt0qwE6SFti+rMN7ioiIiOgaHT2ZK6kOD0tarzRtC9wrafVat92Axu7Ua4CNJS1TNkNsA9xr+8e21yg16d4H3FerLTe39AMYDdxfvnvtWsrEJcAXMpGLiIiIwaY/drN+Ebig7GR9kCqJ4eSydGpgFvB5ANt/k/QDqvguA1fa/nXLURf6HHBSmfw9DYzth98cERERMSB0PJmzPZlqybPuUz30P5+qPEm787OAjWrHvwU27+U37NuHnxoREREx4HRammS9WmHgyZIel3RIOfdFSb8vxYG/W9qGSfpXrf9ptbHGlCLCUyVdLWmV0r5JKTJ8j6RflXIoSFpc0rmlfYakcZ3cS0REREQ36jSb9fdUO1YbBXwfASZIej+wK7Cx7WckrVa77A+2h9fHKUuoJwEb2H60TP4OAr5FteHhMNs3SPoM8BXgG8AewJK23ylpGap39S4sT/YiIiIiBoX+zGbdlmqi9keqciLH2X4GwHZvRX1V/i2ranvq8iwsDrweMKl8ngh8tHx26T8EWBp4Fni8n+4lIiIioiv052RuL+DC8nldYCtJt0q6QdIWtX5rlwLDN0jaCsD2c1QTwHuoJnEbsLCe3DRgl/J5D2Ct8vkSqrp284DZwPdsP9aP9xMRERHxH69fJnNlJ+suVHXgoFq+XRHYkmpZ9OLyxG0e8OZSYPjLwM8kLS9pcarJ3KbAGsBUStID8BngQEl3AstRPYEDGAk8X/qvDfy3pLe2+G1JgIiIiIgBqz9KkwB8ELirVix4DnCpbQO3SXoBWMX2X4DG0uudkv5A9RRPpe0PAJIupkqSwPZMqkxXJK0LfKh8x8eBq8tTvfmSbqLaVftg/YfZPh04vTq6z/10vxERERH/EfprmXUMC5dYAS6jKvDbmIAtATwqadWyUYLyFG0dqsnXI8AGklYt128PzCj9Vit/Xwd8nYVZrLOB0aosS/UUcGY/3U9EREREV+j4yVzZSbo9pTBwcRZwlqRpVMui+9i2pK2BYyQtoFoi3b/xnpuko4FJkp4D/gjsW8YaI+nA8vlS4Ozy+Ufl8zSqJ3tn257a6f1EREREdJP+KBr8FLByU9uzwCdb9B0PjG8zzmksfOpWbz+JqmxJc/sTVBsiIiIiIgat/tzNGhERERH/Zp0mQBxaEh6mSbpQ0lKSbqwlPMyVdFnpO0rSP2rnvlnal5J0m6QpZayja+MfJOkBSW4kQpT2T5SkiKmSfidpk07uIyIiIqJbLfIyq6Q1gS9RpTb8q+xA3cv2VrU+44Ff1i670fbOTUM9A4y2/UQpUfJbSVfZvgW4CbgCuL7pmoeAbWz/TdIHqXarvmtR7yUiIiKiW3X6ztwQYOmyaWEZFqY2IGk5qh2tn+5pgFK+5IlyuHj553Lu7jJW8zW/qx3eAgzt5CYiIiIiutUiL7PafgT4HlWJkHnAP2xfW+uyG3Cd7XrE1rvLcupVkjZsNEpaTNJkYD4w0fatr+CnfBa4alHvIyIiIqKbLfJkTtKKwK5U6QtrUOWk1newNteeuwt4i+1NgB9S1aIDwPbztodTPWEbKWmjPv6G91NN5g7voU8SICIiImLA6mSZdTvgoZLqgKRLgfcA50tamSpua7dG5/oTOttXSjpV0iq2H621/13S9cCOVPXj2pK0MXAG8EHbf23XLwkQERERMZB1spt1NrClpGVK7uq2lNQGqvpvV9h+utFZ0ptKPySNLN/915IKsUJpX5pqkthjkoOkN1MVEP6U7fs6uIeIiIiIrtbJO3O3ApdQLZ/eU8YqT8DYi5cusQLsDkyTNAU4mWrnq4HVgd9ImgrcTvXO3BUAkr4kaQ7V8utUSWeUsb5JVaj41FLm5I5FvY+IiIiIbtbRblbbRwFHtWgf1aLtFOCUFu1TgU3bjH8y1cSvuX0/YL9X/osjIiIiBpYkQERERER0sU4TIA4u6Q/TJR1S2oZLuqWx/Fnej6tfs4Wk5yXtXo7fX0uFmCzpaUkfLucukPT78h1nlaLCjXFGlf7TJd3QyX1EREREdKtOSpNsBHyOatfqJsDOktYBvgscXUqNfLMcN65ZDPgOcE2jzfZvbA8v/UcDTwGNenUXAOsD7wSWpiytlg0TpwK72N6QasNFRERExKDTyZO5dwC32H7K9gLgBqpSJAaWL33eQC0VAvgiMJ6qOHAruwNX2X4KqhImLoDbWJj08HHgUtuzS79240VEREQMaJ1M5qYBW0taWdIywE7AWsAhwPGSHqZKiBgHL2a57gac1sOYrXbBUpZXPwVcXZrWBVaUdL2kOyXt3cF9RERERHStTkqTzKBaMp1INcmaAiwADgAOtb0WcChwZrnkROBw28+3Gk/S6lTLqde0OH0qMMn2jeV4CLA58CHgA8A3JK3bZtwkQERERMSA1WlpkjMpkzVJ/w8wB/h/gYNLl19QpTQAjAAuKnWDVwF2krTAdiPW62PABNvP1b9D0lHAqsDna81zgEdtPwk8KWkS1Xt7LysgnASIiIiIGMg63c26Wvn7ZuAjVEukc4FtSpfRwP0Atte2Pcz2MKpiw1+oTeTg5VmuSNqP6snbGNsv1E79EthK0pCyxPsuFqZPRERERAwaHT2ZA8aXHNbngANt/03S54CTJA0BngbG9jaIpGFU79s1lxg5DfgjcHN5onep7WNsz5B0NTAVeAE4w3aPWa4RERERA1Gny6xbtWj7LdX7bD1dt2/T8SxgzRb92v4+28cDx/fxp0ZEREQMSEmAiIiIiOhifZrMlfSF+ZKm1dpWkjRR0v3l74qlfUVJEyRNlXRbKS5cH2sxSXdLuqLWdpCkByRZ0iotvv8lqRGlbZ/y3fdL2mdRbj4iIiKi2/X1ydw5wI5NbUcA19leB7iuHAN8DZhse2Ngb+CkpusO5uWbFW4CtqN6P+4lWqVGSFoJOIpq48NI4KjGZDIiIiJiMOnTZM72JOCxpuZdgXPL53OBD5fPG1BN7rA9Exgm6Y0AkoZS1YY7oz6Q7bvLe3OttEqN+AAw0fZjtv9GVeuuebIZERERMeB18s7cG23PAyh/VyvtU6jKlCBpJPAWFsZwnQh8lWoHaq96SI1YE3i4djyHFhsoIiIiIga6V2MDxHFUUVuTqZ6q3Q0skLQzMN/2na9grHapEWrRt2VB4CRARERExEDWSWmSP0ta3fa8EsU1H8D248CnAVQVh3uo/NsL2EXSTsBSwPKSzrf9yR6+o2VqBNWTuFG1fkOB61sNkASIiIiIGMg6eTJ3OdDYRboPVSoDklaQtERp348qU/Vx2+NsDy0JEHsB/9fLRK6n1IhrgB3KztkVgR1onekaERERMaD1tTTJhcDNwHqS5kj6LNVy6vaS7ge2L8cA7wCmS5oJfJCFOa09jf8lSXOonrBNlXRGT/1tPwYcC9xe/h1T2iIiIiIGlT4ts9oe0+bUti363gys08t411NbFrV9MnByL9fs23R8FnBWT9dEREREDHRJgIiIiIjoYp0kQBwvaWZJepggaYXauXEl0eH3kj5Q2pYqiRBTJE2XdHSt/zmSHpI0ufwbXtpHSfpHrf2bTb/rZWkSEREREYNJJwkQE4GNStLDfcA4AEkbUG1w2LBcc2pJcXgGGG17E2A4sKOkLWvjfcX28PJvcq39xlr7MU2/oVWaRERERMSgscgJELavtb2gHN7CwsLAuwIX2X7G9kPAA8BIV54ofRYv/xa5VEi7NImIiIiIwaS/3pn7DHBV+dw2naEsi06mqkk30fattX7fLku2J0hastb+7rI0e5WkDWvtryhNIiIiImIg6ngyJ+lIYAFwQaOpRTcD2H7e9nCqp3gjJW1Uzo8D1ge2AFYCDi/tdwFvKUuzPwQuK9/Z5zSJJEBERETEQNZJAgSS9gF2Bra13VgynQOsVes2FJhbv8723yVdT/VO3bRGxivwjKSzgcNKv8dr11wp6VRJqwDvpY9pEkmAiIiIiIFskZ/MSdqR6gnaLrafqp26HNhL0pKS1qaqOXebpFUbO14lLQ1sB8wsx6uXvwI+DEwrx28qbUgaWX7vXxclTSIiIiJiIOrTk7mSADEKWKUkNRxFtTS6JDCxzLdusb2/7emSLgbupVp+PdD282XCdm7Z2fo64GLbjZIiF0halWqJdjKwf2nfHTig5LH+C9ir9gQwIiIiYtDrJAHizB76fxv4dlPbVGDTNv1Ht2k/BTill992PbU0iYiIiIjBJAkQEREREV2skwSIY0spkcmSrpW0RmlfX9LNkp6RdFit/1qSfiNpRkmAOLh2bpNyzT2SfiVp+dq5jcu56eX8UqV9TDmeKunqsjEiIiIiYlDpJAHieNsbl1IjVwCNqK3HgC8B32vqvwD4b9vvALYEDixpEVAV/j3C9juBCcBXACQNAc4H9re9IdV7e8+V9pOA95cEiqnAQX28l4iIiIgBo5MEiMdrh8uysJbcfNu3A8819Z9n+67y+Z9UMVxrltPrAZPK54nAR8vnHYCptqeU6/5q+3mqjRICli27XZenqfxJRERExGDQ0Ttzkr4t6WHgEyx8MteX64ZRbYZoJEBMA3Ypn/dgYZ26dQFLukbSXZK+CmD7OeAA4B6qSdwGtNmQkaLBERERMZB1NJmzfaTttajSH/q0zCnp9cB44JDa073PUC273gksBzxb2ocA76OaLL4P2E3StpIWp5rMbQqsQbXMOq7Nbzzd9gjbI8aO3XNRbjMiIiLiP1Z/7Wb9GQuXRtsqk7DxwAW2L220255pewfbmwMXAn8op+YAN9h+tBQmvhLYDBhervtDqTt3MfCefrqXiIiIiK7RSQLEOrXDXShpDj30F9VS6AzbP2g6t1r5+zrg68Bp5dQ1wMaSlimbHrahKkb8CLBBKTQMsD3VO3gRERERg0onCRA7SVoPeAH4IyW1QdKbgDuoNiW8IOkQqnfaNgY+BdwjaXIZ+mu2rwTGSDqwtF0KnA1g+2+SfgDcTrXB4krbvy7fczQwSdJz5fv3XdT/CBERERHdqt8TIGz/CRja4tRvqXagtrrmJKpSI63OnU9VnqS5/TQWPsGLiIiIGJSSABERERHRxRY5AaJ27jBJbiQwSPpESWWYKul3kjYp7T0lQLRLk3hDSYSYUq75dGkfXkuFmCop21QjIiJiUOokAQJJa1FtPphda34I2KYkMxwLnF7ae0qAaJcmcSBwr+1NqN7Z+76kJYCngL1LKsSOwImSVujjvUREREQMGIucAFGcAHyVkv5Q+v7O9t/K4S2U9+d6SoBolyZR/i5XdsK+vvyGBbbvs31/uXYuMB9YlYiIiIhBppPSJLsAjzSittr4LHBVi2uH8dIEiHZpEqcA76BKebgHONj2C01jjQSWYGFtuubvSgJEREREDFh92s3aTNIywJFU2ant+ryfajL3vqb2VgkQ2D4SOFLSOKo0iaOADwCTgdHA24CJkm5sXCdpdeCnwD7Nk7zauKfz4lLvfW7VJyIiIqJbLeqTubcBawNTJM2iWkq9q9SYQ9LGwBnArrb/2rioXQJEk3qaxKeBS115gOp9vPXLWMsDvwa+bvuWRbyPiIiIiK62SJM52/fYXs32MNvDqGK3NrP9J0lvpir8+ynb9zWu6SUBol2axGxg29LnjcB6wINlE8QE4Dzbv1iUe4iIiIgYCPpamuRC4GZgPUlzJH22h+7fBFYGTi2lRu4o7e+lSoAYXdonS9qpnDtO0jRJU6mWbhtlS44F3iPpHuA64HDbjwIfA7YG9q2NNbzvtx0RERExMHSSAFE/P6z2eT9gvxZ9ekqA+Gib9rm0eC+vXSpERERExGCTBIiIiIiILtbrZK5V+oOkb0l6pHm5VNLIWtsUSbv1NE5p/3ntmlmSJpf2JSSdLemeMtao2jVLSDpd0n2SZkpq+WQvIiIiYqDryzLrOVT13s5raj/B9vea2qYBI2wvKGVDpkj6le0F7cax/WIUl6TvA/8oh58r598paTXgKklblBIkRwLzba8r6XXASn24j4iIiIgBp9fJnO1Jpchvr2w/VTtcipcmQ/Q4Ttnt+jGqmnIAG1BtesD2fEl/B0YAtwGfoZQoKZO7R/vy+yIiIiIGmk7emTvYE+scAAAgAElEQVSohNyfJWnFRqOkd0maTpXYsH95KtcXWwF/bsR0AVOAXSUNkbQ2sDmwVi2D9VhJd0n6RSlb0lISICIiImIgW9TJ3I+pCgcPB+YB32+csH2r7Q2BLYBxkpbq45hjgAtrx2dR1a+7AzgR+B2wgOpp4lDgJtubUZVMaV7ufZHt022PsD1i7Ng923WLiIiI6EqLFOdl+8+Nz5L+F7iiRZ8Zkp4ENqKakLUlaQjwEaqnb43rFwCH1vr8Drgf+CvwFFXRYIBfUMWGRURERAw6i/RkrmxuaNiNauMDktYuEzMkvYUqsWFWH4bcDphpe07tO5aRtGz5vD2wwPa9tg38ChhVum4L3Lso9xERERHR7Xp9MlfSH0YBq0iaAxwFjCqJC6aarH2+dH8fcISk54AXgC+UxIaW49g+s1y3Fy9dYgVYDbhG0gvAI1TpEQ2HAz+VdCLwF6oM14iIiIhBpy+7WVulP5zZog3bPwV++grGaZzbt0XbLKone636/5EqzisiIiJiUOtrNmu7gr9flPR7SdMlfbfp3JslPSHpsD6Mc3wp/jtV0oTGjlVJK0v6TRnnlKZrNi8FhR+QdHIpbRIRERExqPT1nblzgB3rDZLeD+wKbFx2rzbvKD0BuKq3cYqJwEa2NwbuA8aV9qeBbwCHtbjmx8BYYJ3yr9W4EREREQNanyZzticBjzU1HwAcZ/uZ0md+44SkDwMPAtP7MA62r63Vo7uFqvQItp+0/VuqSd2LygaM5W3fXDZEnAd8uC/3EhERETGQdFI0eF1gK0m3SrpB0hYAZQfq4cDRizjuZ3j5E71ma1LVoGuYU9oiIiIiBpVOJnNDgBWBLYGvABeX99aOpsptfeKVDijpSKrCwBf01rVFm1u0JQEiIiIiBrRFKhpczAEuLcuct5USIqsA7wJ2LxsiVgBekPS07VN6GAtJ+wA7A9uWMXv77qG146HA3FYdbZ8OnF4d3dfbuBERERFdpZPJ3GXAaOB6SesCSwCP2t6q0UHSt4An+jCR25FqaXYb20/19sW250n6p6QtgVuBvYEfLvKdRERERHSpvpYmuZAqA3U9SXMkfZYqO/WtpczIRcA+vT1RazMOwCnAcsBESZMlnVa7ZhbwA2Dfcs0G5dQBwBnAA8Af6P09u4iIiIgBp09P5noo+PvJXq77Vl/Gsf32HsYY1qb9Dqrc14iIiIhBq5MNEBERERHxGlvkBAhJPy9LopMlzZI0ubQPk/Sv2rn6kmnb1IZWaRKStpd0Z7nmTkmjW/y2y5sTJSIiIiIGi75ugDiH6r228xoNtvdsfJb0feAftf5/sD28xTiN1IZbgCupUhuuakqTeEbSaqX/o8B/2Z4raSPgGmr15CR9BHjFJVAiIiIiBopOEiAAKE/XPgZc2NMYvaQ2tEyTsH237UbJkenAUpKWLOO9Hvgy8D99uYeIiIiIgag/3pnbCviz7ftrbWtLurskQzRKlfSU2tAyTaLJR4G7GxM+4Fjg+0CvpUwiIiIiBqr+mMyN4aVP5eYBb7a9KdWTs59JWp6eUxvapUkAIGlD4DvA58vxcODttif09uOSABEREREDWSdFg5E0BPgIsHmjrTw5ayyX3inpD1RP3npKbWiXJvEXSUOBCcDetv9Q+r8b2LzUoBsCrCbpetujmn9jEiAiIiJiIOv0ydx2wEzbLy6fSlpV0mLl81uBdYAHbc8D/ilpy/LUbW/gl+WyRpoE9TQJSSsAvwbG2b6p8R22f2x7jVKD7n3Afa0mchEREREDXScJEAB78fKND1sDUyVNAS4B9rfd2DzRLrWhXZrEQcDbgW/USp2sRkREREQAHSZA2N63Rdt4YHyb/i1TG2w/S4s0Cdv/Qy+7VW3PajVmRERExGCQBIiIiIiILtZJAsRwSbeUpc87JI2snRtV2qdLuqHWvmNJeXhA0hG19rVLWZL7S7LEEqV9/5L+MFnSbyVtUNoXl3RuOTdD0rj++I8RERER0W36+mTuHKq0hrrvAkeXpIdvlmPKpoVTgV1sbwjsUdoXA34EfBDYABjTmJxRlR05wfY6wN+Axjt5P7P9zvId3wV+UNr3AJa0/U6qnbSflzSsj/cSERERMWB0kgBhYPny+Q0sLDPycaoyI7PLtfNL+0jgAdsPlnfkLgJ2LTtbR1NtlgA4l5IMYfvx2vcty8K6dAaWLaVRlgaeBep9IyIiIgaFTurMHQJcI+l7VJPC95T2dYHFJV0PLAecZPs8qrSHh2vXzwHeBawM/N32glp7PX/1QKriw0tQypdQTfx2pSpQvAxwaG3HbERERMSg0clk7gCqSdR4SR8DzqSqOzeEaulzW6qnZjdLuoX2CRA9JUNg+0fAjyR9HPg6sA/VU77ngTWokiNulPT/2X6weSBJY4GxAD/5yTGMHbvnIt5uREQMFEu/+ahXdfx/zb7wVf2Of83uMQ69a7za/426ffy+6mQytw9wcPn8C6r6cVA9WXvU9pPAk5ImAZuU9rVq1zcSIB4FVpA0pDydqydD1F0E/Lh8/jhwte3ngPmSbgJGAC+bzCUBIiIiIgayTiZzc4FtgOuplj/vL+2/BE4p77MtQbWUegIwE1hH0trAI1QFhz9u25J+A+xOKRhcxkDSOrYb436o9h2zgdGSzqdaZt0SOLGDe4mIiEHkX7OPHhDf0e1e7f9G3T5+X/VpMlcSIEYBq0iaAxwFfA44qUzanqYsZdqeIelqYCrwAnCG7WllnIOAa4DFgLNsTy9fcThwkaT/Ae6mWrIFOEjSdsBzVLtc9yntPwLOBqZRLdOebXvqIv0XiIiIiOhiHSVAUL0b16r/8cDxLdqvBK5s0f4g1Xtwze0HN7eV9icoJU8iIiIiBrMkQERERER0sVdtMifpYEnTSgrEIaXt5yXNYbKkWZIml/aVJf1G0hOSTmkaZ/OS9PCApJNLXTokbSLp5nLuV5KWf/mviIiIiBjYXpXJnKSNqN6pG0m1k3XnsplhT9vDS6LDeODScsnTwDeAw1oM92Oq9/HWKf8aSRRnAEeUFIgJwFdejXuJiIiI+E/2aj2Zewdwi+2nSrmRG4DdGifL07WPARcC2H7S9m+pJnXU+q0OLG/7ZtsGzqOkQwDrAZPK54nAR1+le4mIiIj4j/VqTeamAVuX5dNlgJ14aY25rYA/18qOtLMmVX26hno6xDRgl/J5j6bxIyIiIgaFV2UyZ3sG8B2qJ2ZXA1OABbUuYyhP5XrRUzrEZ4ADJd1JFRv2bMsBpLGS7pB0x+mn/7yPdxARERHRHTopGtwj22dS6sVJ+n8oT9hKXbqP0KasSZM5VIkQDS+mQ9ieCexQxlyXqqhwq9+RBIiIiIgYsF7N3ayrlb9vppq8NZ7EbQfMtD2n3bUNtucB/5S0ZXnPbm8WpkM0xn8dVWbraf1+ExERERH/4V61J3PAeEkrU6U3HGj7b6V9L1ossUqaBSwPLCHpw8AOtu8FDgDOAZYGrir/AMZIOrB8vpQqESIiIiJiUHk1l1m3atO+b5v2YW3a7wA2atF+EnDSov/CiIiIiO6XBIiIiIiILtbRZE7SWiW5YUZJeji4tK8kaaKk+8vfFWvXjCoJENMl3VBrn1XSHCZLuqPWfrykmZKmSpogaYXSPrKWJjFF0m5EREREDDKdPplbAPy37XcAW1KVCtkAOAK4zvY6wHXlmDIROxXYxfaGVPXh6t5fEiJG1NomAhvZ3hi4DxhX2qcBI0qaxI7AT8pO2YiIiIhBo6PJnO15tu8qn/8JzKAq6rsrcG7pdi4LUxs+Dlxqe3a5Zn4fvuPakiIBcAulVEktXQJgKRbWn4uIiIgYNPrtnTlJw4BNgVuBN5ayIo3yIquVbusCK0q6XtKdkvauDWHg2tI+ts3XfIaFu1mR9C5J04F7gP1rk7uIiIiIQaFfJnOSXg+MBw6x/XgPXYdQFQv+EPAB4Bul4C/Ae21vBnyQarl266bvOJJqWfeCRpvtW8ty7RbAOElLtfhtSYCIiIiIAavjd8wkLU41kbvA9qWl+c+SVrc9T9LqQGM5dQ7wqO0ngSclTQI2Ae6z3Uh2mC9pAjASmFS+Yx9gZ2Bb2y9bTrU9Q9KTVCVM7mg6lwSIiIiIGLA63c0qqsiuGbZ/UDt1ObBP+bwPJbWh/N1K0hBJywDvAmZIWlbScmXMZaliuqaV4x2Bw6k2TTxV++61GxseJL0FWA+Y1cn9RERERHSbTp/MvRf4FHCPpMml7WvAccDFkj4LzKbsWi1P0K4GpgIvAGfYnibprcCEam7IEOBntq8u450CLAlMLOdvsb0/8D7gCEnPlbG+YPvRDu8nIiIioqt0NJmz/VtAbU5v2+aa44Hjm9oepFpubdX/7W3afwr8tM8/NiIiImIASgJERERERBfr9J25pSTdVhIYpks6urQfJOkBSZa0Sq3/KEn/qCU3fLN27mBJ08o4h9Tajy3pD5MlXStpjdL+ldo40yQ9L2mlTu4nIiIiott0+mTuGWC07U2A4cCOkrYEbgK2A/7Y4pobS8rDcNvHAEjaCPgc1Q7WTYCdJa1T+h9ve+OS9HAF8E2olmsb41ClQtxg+7EO7yciIiKiq3SaAGHbT5TDxcs/277b9qxXMNQ7qDY2NFIdbgB2K99Rr1u3LK2THsYAF77S3x8RERHR7Tp+Z07SYmUn63xgou1be7nk3WVZ9ipJG5a2acDWklYuJUt2Ataqfce3JT0MfILyZK52bhmqbNbxbX5figZHRETEgNVx0WDbzwPDJa1AVV5kI9vT2nS/C3iL7Sck7QRcBqxTSpZ8B5gIPAFMoUp7aHzHkcCRksYBBwFH1cb8L+CmdkusKRocERERA1m/7Wa1/XfgeqqnZO36PN5YlrV9JbB4Y4OE7TNtb2Z7a+Ax4P4WQ/wM+GhT215kiTUiIiIGqU53s65ansghaWmqTQ8ze+j/ppIagaSR5fv/Wo5XK3/fDHyEMkGrbYQA2KU+vqQ3ANuwMGEiIiIiYlDpdJl1deBcSYtRTcwutn2FpC8BXwXeBEyVdKXt/YDdgQMkLQD+BexVy1odL2ll4DngQNt/K+3HSVqPKuXhj8D+te/fDbi2ZL1GREREDDqdJkBMBTZt0X4ycHKL9lOo4rlajbVVm/bmZdX6uXOAc/r2ayMiIiIGniRARERERHSx/ipNcrekK8rxBZJ+X1IZzpK0eGlvmf7QLkWinDtH0kO1a4aX9vUl3SzpGUmHdXoPEREREd2q49IkwMHADGD5cnwB8Mny+WfAfsCPy/GNtnduur6RIvFEmfj9VtJVtm8p579i+5Kmax4DvgR8uB9+f0RERETX6nQ361DgQ8AZjTbbV5ZkCAO3AUN7GqNdikQv18y3fTvVZomIiIiIQavTZdYTqXatvtB8ojxl+xRwda25VfpDbykS35Y0VdIJkpZ8pT8wCRARERExkC3yMquknYH5tu+UNKpFl1OBSbZvLMct0x+gxxSJccCfgCWoUhwOB455Jb8zCRARERExkHXyZO69wC6SZgEXAaMlnQ8g6ShgVeDLjc49pT/U+rwkRcL2vLIM+wxwNjCyg98bERERMeAs8mTO9jjbQ20Po4rU+j/bn5S0H/ABYIztF5df26U/9JQiIWn18ldUmx3aZb5GREREDEr9sZu12WlUSQ03l7nbpbaPoU36Q5mwvSxFoox1gaRVAQGTKekPkt4E3EG1g/YFSYcAG9h+/FW4n4iIiIj/WP0ymbN9PdXyKLZbjtku/aFdikQ5N7pN+5/oZZdsRERExGCQBIiIiIiILtYfCRCzJN1TEhruKG3HlnIikyVdK2mNpmu2kPS8pN3L8ftrKQ+TJT0t6cPlnCR9W9J9kmZI+lJpb5koERERETGY9Nc7c++3/Wjt+Hjb3wAok69vsvB9t8WA7wDXNDrb/g3QiOpaCXgAuLac3hdYC1jf9guSVqt9T6tEiYiIiIhB49XYAEHTRoRleWmiwxeB8cAWbS7fHbjK9lPl+ADg442dsbbn9/PPjYiIiOha/fHOnIFrJd0paWyjsSyNPgx8gurJHJLWBHaj2vHazl7AhbXjtwF7lhSHqyStUzvXMlGiLgkQERERMZD1x5O599qeW5Y/J0qaaXuS7SOBIyWNAw4CjqKK/zrc9vOlbMlLlDIl76S2BAssCTxte4SkjwBnAVvRQ6JEXRIgIiIiYiDr+Mmc7bnl73xgAi9PafgZ8NHyeQRwUUmN2B04tbHRofgYMMH2c7W2OVTLspTxNy7f12uiRERERMRA19FkTtKykpZrfAZ2AKY1LYXuQkl0sL227WElNeIS4Au2L6v1HcNLl1iheuLWqDe3DXBf+b6WiRKd3E9EREREt+l0mfWNwIQypxoC/Mz21ZLGS1oPeIEqDWL/3gaSNIxq1+oNTaeOo0qCOBR4AtivtLdMlOjwfiIiIiK6SkeTOdsPApu0aP9oi+7NffZtOp4FrNmi39+BD7Vob5koERERETGYJAEiIiIioov1RwLECpIukTSzJDS8u3buMElu3pjQnABR2q6W9HdJVzT1vbGW8jBX0mWlfUVJE0rSxG2SNur0XiIiIiK6TX+UJjkJuNr27pKWAJYBkLQWsD0wu965VQJEcXy59vP1Rttb1a4dD/yyHH4NmGx7N0nrAz8Ctu2H+4mIiIjoGp3uZl0e2Bo4E8D2s+UdN4ATgK/y0vQHWJgA8ZIkB9vXAf/s4buWo9rV2tj9ugFwXbl2JjBM0hs7uZ+IiIiIbtPpMutbgb8AZ0u6W9IZpVzJLsAjtqfUO/cxAaKd3YDralFhU4CPlHFHAm8BhjZflASIiIiIGMg6XWYdAmwGfNH2rZJOAr5F9bRuhxb9e0yA6MUY4Iza8XHASZImA/cAdwMLmi9KAkREREQMZJ1O5uYAc2zfWo4voZrMrQ1MKRO2ocBd5elZIwECYBVgJ0kLmgoHv4yklamSJXZrtJUndJ8u5wU8VP5FREREDBqd1pn7k6SHJa1n+/dUGxDusv3iRoQS3TXC9qNUk7xG+znAFb1N5Io9St+na9evADxl+1mqQsKTakuwEREREYNCf+xm/SJVQsMSwIOUp2WvlKQbgfWB10uaA3zWdmPH615Uy6p17wDOk/Q8cC/w2UX53oiIiIhu1vFkzvZkquXTdueHtWnft+l4q1b9yrlRLdpuBtZ5ee+IiIiIwaPT0iTr1Qr6Tpb0uKRDJH1L0iO19p1K/5G1timSdquNNUvSPeXcHbX2TSTdXM79qpRD6XGsiIiIiMGi03fmfg8MhxeLAT8CTKBaaj3B9veaLplG9f7cAkmrU22S+JXtxi7U95d36+rOAA6zfYOkzwBfAb7Rh7EiIiIiBrz+zGbdFviD7T+262D7qdpkayleXlC4lfWASeXzROCjHYwVERERMaD052RuL+DC2vFBJTf1LEkrNholvUvSdKracPvXJmQGrpV0p6SxtXGmAbuUz3sAa/VhrIiIiIhBoV8mc2Un6y7AL0rTj4G3US3BzgO+3+hr+1bbGwJbAOMkLVVOvdf2ZsAHgQMlbV3aP1OO7wSWA57tw1j135YEiIiIiBiw+qM0CVQTsLts/xmg8RdA0v8CVzRfYHuGpCeBjYA7bM8t7fMlTaAqEjyp5K7uUMZaF/hQb2M1nUsCRERERAxY/bXMOobaEmvZkNCwG9VSKZLWljSkfH4L1ftws0qe63KlfVmqyVvjmtXK39cBX6fkurYbq5/uJyIiIqIrdPxkTtIywPbA52vN35U0nOo9uFm1c+8DjpD0HPAC8AXbj0p6KzChxHwNAX5m++pyzRhJB5bPlwJn9zRWp/cTERER0U36o2jwU8DKTW2fatP3p8BPW7Q/CGzS5pqTgJP6OlZERETEYNKfu1kjIiIi4t+s48mcpEMlTZc0TdKFkpZS5duS7pM0Q9KXSt9dS7mSyWWH6ftq4+wj6f7yb59a++Yl/eEBSSerrMVK+nktAWKWpMmd3ktEREREt+lomVXSmsCXgA1s/0vSxVT15kRVD2592y80NjEA1wGX27akjYGLgfUlrQQcRZXxauBOSZfb/htVmZOxwC3AlcCOwFW296z9ju8D/+jkXiIiIiK6UX8ssw4Bli47S5cB5gIHAMfYfgGqciPl7xO2G+VBlmVhasMHgIm2HysTuInAjmVX7PK2by7XnQd8uP7l5Undx3hpweKIiIiIQaGjyZztR4DvAbOpigP/w/a1VAWD9yxLqVdJWqdxjaTdJM0Efk1VEBhgTeDh2tBzStua5XNze91WwJ9t39/JvURERER0o44mcyWma1dgbWANYFlJnwSWBJ62PQL4X+CsxjW2J9hen+oJ27GNoVoM7x7a615S467Fb0wCRERERAxYnZYm2Q54yPZfACRdCryH6gna+NJnAgtrw73I9iRJb5O0Suk/qnZ6KHB9aR/a1D63cVCWdj8CbN7uByYBIiIiIgayTt+Zmw1sKWmZ8u7atsAM4DJgdOmzDXAfgKS313ajbgYsAfwVuAbYQdKK5WnfDsA1tucB/5S0Zblub+CXte/fDphpu74UGxERETFodPRkzvatki4B7gIWAHdTPQVbGrhA0qHAE8B+5ZKPAnuX1IZ/AXuWjQ2PSToWuL30O8b2Y+XzAcA5Zcyryr+GvcjGh4iIiBjE+iMB4iiqsiJ1zwAfatH3O8B32oxzFrV362rtdwAbtblm31f4cyMiIiIGlCRARERERPz/7N173GZj3f7xz5FpyC4TkYyMst8OxmhHNpFUNsmuDUVJUaof4pHUI89DSJ5HaWObxzZMIWEe2aRQhhlmzNjExKCGkOyGMcfvj3VezXLNdd33Nfc18/t1X/fxfr3mNes617nOtdb91/laa53fYxDrdjXrwSX5YYqkr5S2lskMkraVNKGkOUyQtHWL8S6XNLn2+wRJ00pqxDhJy5T2ZSVdL+k5Sad2cw8RERERg9mAJ3OS1gM+B4wFNgQ+LGl123vYHm17NNWK1svKIU8CH7G9PrAPcG7TeB+l+r6ubjywnu0NqBZRHFHaXwKOAg4Z6PVHRERE9IJunsytDdxq+wXbs4EbgV0aO5uTGWzfabtRVmQKsJikRUvfJYGvAd+pn8D2tWVsqOK8Rpb2523fTDWpi4iIiBiyupnMTQa2KK88Fwd2oMpjbegrmWFX4E7bs8rvY4CTgBf6ON++vHYla0RERMSQN+DJnO2pVCtTxwNXA5OoypM0tExmkLRuOe7z5fdoYDXb49qdS9KRZezz5vc6kwARERERvazbOnNnAGcASPoPSo5qu2QGSSOpEiH2tv2n0vwuYBNJ08v1LC/pBttblmP2AT4MbFNq0s3vNSYBIiIiInpWt6tZly//v41q8tZ4EjdPMkNZifor4Ajbv2u02z7N9lttjwLeC9xXm8htD3wd2NF2X69gIyIiIoakbosGXyppWeAV4EDbT5f2VskMBwGrAUdJOqq0bWd7Zh/jnwosCowvKWC32j4AoDzJWxoYLmnnMtY9Xd5PRERExKDS7WvWzdu0f7pF23doWq3aos90amkPtlfro++oDi8zIiIiomclASIiIiJiEOtoMifpTEkzm9IZ3iRpvKT7y/8jSvsbJV0haVJJhvhM7ZirJT0j6cqm8beRdEdJjbhZ0mql/YCSGNFoX6e0j5L0Yi1p4kcL4o8RERERMdh0+mTubGD7prbDgetsrw5cV34DHAjcY3tDYEvgJEnDy74TgE+1GP804BMlNeJ84Bul/Xzb65f27wLfqx3zp0bSROM7uoiIiIihpqPJnO2bgKeamncCzinb5wA7N7oDS5UEiCXLcbPLONcB/2h1CqrFDABvBB4r/Z+t9Vmi9IuIiIiIopsFECvYfhzA9uONMiVUK1Avp5qQLQXsYXtOP2N9FrhK0ovAs8A7GzskHUgV9TUc2Lp2zKqS7iz9v2H7t13cS0RERMSgtDAWQHwAmAi8FRgNnCpp6b4P4avADrZHAmdRe51q+we230FVb67x+vVx4G22N6Ka6J3f7hxJgIiIiIhe1s2Tub9KWrE8lVsRaNSL+wxwXElreEDSQ8BawB9aDSLpzcCGtm8rTRdRxYM1u5Dq2zpKpuussj1B0p+ANYDbmw9KAkRERET0sm6ezF0O7FO29wF+WbYfBrYBkLQCsCbwYB/jPA28UdIa5fe2wNRy/Oq1fh8C7i/tb5a0SNl+O7B6P+eIiIiI6EkdPZmTdAHVytTlJM0AjgaOAy6WtB/VBG630v0Y4GxJdwMCvm77yTLOb6me0i1ZxtnP9jWSPkeVJjGHanK3bxnrIEnvp0qYeJq5k8ctgH+XNBt4FTjAdvMCjYiIiIie19FkzvZebXZt06LvY8B2bcZplxgxDhjXov3gNv0vBS5td70RERERQ0W32awRERGDzhvedvRCHf/Fhy9YqOd48eHm+PPBaWH/jQb7+J3qdzIn6Uzgw8BM2+uVtt2AbwFrA2Nt317ahwM/BsYAc4CDbd9Q9h0L7A2MsL1kbfyvUZUmmQ08Aexr+8+1/UtTfUM3zvZBpW0P4EhgEeBXtg/r+I4jImLIe/Hhb/fEOQa7hf03Guzjd6qTBRBnM2/6w2Tgo8BNTe2fA7C9PtVChpMkNc5xBTC2xfh3AmNsbwBcQpX0UHcMcGPjh6RlqZIktrG9LrCCpHle90ZEREQMBf1O5lqlP9ieavveFt3XoYr2wvZM4Bmqp3TYvrVRZLhprOttv1B+3gqMbOyTtAmwAnBt7ZC3A/fZfqL8/l9g1/7uIyIiIqIXLeiiwZOAnSQNk7QqsAmw8nwcvx/wa4DyRO8k4NCmPg8Aa0kaJWkYVYzY/JwjIiIiomcs6MncmcAMquK93wd+T8ll7Y+kT1I9xTuhNH0RuMr2I/V+tp8GvkBVXPi3wPS+zpEEiIiIiOhlC3Q1q+3ZVNFcAEj6PaXQb19KLbkjgfeVdAeAdwGbS/oisCQwXNJztg+3fQXVN3hI2p+q1ly7a3TCWFEAACAASURBVEoCRERERPSsBTqZk7Q4INvPS9oWmG37nn6O2YhqBez25Ts7AGx/otbn01SLJA4vv5e3PVPSCKoneLsvyPuIiIiIGCz6fc1a0h9uAdaUNEPSfpJ2KQkO7wJ+Jema0n154A5JU4GvA5+qjfPdcsziZZxvlV0nUD15+7mkiZIu7+C6T5F0D/A7qhzY+zq73YiIiIje0u+TuT7SH1olNkynymJtNc5hwDz14Gy/v4NrOJuqREp/1xQRERExpCzoBRARERER8f9QR5M5SWdKmilpcq3tGEl3lVej10p6a2mXpP+S9EDZv3HtmH0k3V/+7VNrHy7pJ5LukzRN0q61fbtLukfSFEnnl7ZVJE0o554i6YAF8ceIiIiIGGw6XQBxNnAq8LNa2wm2jwKQ9GXgm8ABwAeB1cu/zYDTgM0kvQk4mqr8iIEJki4vpUaOpIoLW6PUl3tTGXd14AjgPbaflrR8OffjwLttz5K0JDC5jPXYgP4KEREREYNUR5M52zdJGtXU9mzt5xJUEzSAnYCf2TZwq6RlJK0IbAmMt/0UgKTxVDFhFwD7AmuVcecAT5axPgf8oEz4GqkS2H65du5FyeviiIiIGKK6mgRJOlbSI8AnqJ7MAawE1Av9zihtLdslLVN+HyPpDkk/l7RCaVsDWEPS7yTdKumfGbGSVpZ0Vxnz+HZP5VI0OCIiInpZV5M520faXhk4DzioNKtV1z7ah1Hlsf7O9sZUZVBOLPuHUb2u3RLYCzi9Mfmz/YjtDYDVgH1qE8Dma/yJ7TG2x+y//x4DuMuIiIiIf10L6vXk+cwNu5/Ba7NSRwKP9dH+N+AF5pY6+TnQWDQxA/il7VdsPwTcSzW5+6fyRG4KsPkCupeIiIiIQWPAk7myOKFhR2Ba2b4c2Lusan0n8HfbjwPXANtJGlGSG7YDrinf1l1B9fQNYBugkRrxC2Crcr7lqF67PihppKQ3lPYRwHuoJnoRERERQ0pHCyBKCsSWwHIlxeFoYAdJawJzgD9TrWQFuArYAXiA6onbZwBsPyXpGOCPpd+/NxZDUKVFnCvp+8ATjWOYOwG8hyp/9VDbfytRYSdJary+PdH23QP5A0REREQMZp2uZm2VuHBGm74GDmyz70zgzBbtfwa2aDPW18q/evt4YIN+LzwiIiKix6WkR0RERMQgNuAEiNq+QyS5fNOGpLUk3SJplqRDmvpuL+nekg5xeK19VUm3lWSIiyQNL+0nl5SHiSUd4pnSvlWtfaKklyTt3M0fIiIiImIw6vTJ3NlUBX5fQ9LKwLbAw7Xmp4AvM7e8SKPvIsAPqBIi1gH2krRO2X08cLLt1YGngf0AbH/V9mjbo4H/Bi4r7dfX2rem+jbv2g7vJSIiIqJndDSZs30T1SSt2cnAYcxNf8D2TNt/BF5p6jsWeMD2gyXB4UJgJ0mimpBdUvqdA7R6yrYXVVpEs48Bv7b9Qif3EhEREdFLuilNsiPwqO1JHR7SLhliWeAZ27Ob2uvnWgVYFfhNi3H3pPUkr3FsEiAiIiKiZ3W0mrWZpMWBI6lqxXV8WIu2vpIh6vYELrH9atN1rAisT1XCpCXbPwF+Uv26r3nciIiIiEFtoE/m3kH1pGySpOlUaQ53SHpLH8e0S4B4ElhG0rCm9rp2T992B8bZbn6lGxERETEkDGgyZ/tu28vbHmV7FNVEbWPbf+njsD8Cq5eVq8OpJmiXl1py11N9+wawD/DLxkGlMPEIqszWZu2+o4uIiIgYEjotTXIB1WRqTUkzJO3XR9+3lJSIrwHfKP2XLt/EHUT1SnQqcLHtKeWwrwNfk/QA1Td09YLEewEXlklf/TyjqJ703djJPURERET0om4SIOr7R9W2/0L1qrRVv6uo4r6a2x+kWu3a6phvtWmfTtNCiYiIiIihJgkQEREREYNYv5O5VukPkr4l6dFaAsMOpX1ZSddLek7SqU3j7CHpLklTJH23xXk+VpIkxpTfwyWdJeluSZMkbVnre6ykRyQ918W9R0RERAx6nTyZO5sW6Q9UiQ2jy7/Gq9OXgKOA5hivZYETgG1srwusIGmb2v6lqFIjbqsd9jkA2+tTpUycJKlxvVfQ5rVsRERExFDS72Suj/SHVn2ft30z1aSu7u3AfbafKL//F9i1tv8Y4LtNx60DXFfGnQk8A4wpv2+1/Xgn1xQRERHRy7r5Zu6g8tr0TEkj+un7ALCWpFGlntzOlJpzkjYCVrZ9ZdMxk6jivoZJWhXYhNfWqetIEiAiIiKilw0oAQI4jeppmsv/JwH7tuts+2lJXwAuAuYAvwfeXl6bngx8usVhZwJrA7cDfy7HzG7Rr09JgIiIiIheNqDJnO2/NrYl/RRofqrW6pgrqL51Q9L+wKvAUsB6wA2SAN4CXC5pR9u3A1+tnef3wP0Dud6IiIiIXjWg16wlE7VhF2Byu761Y5Yv/48AvgicbvvvtperJUncCuxo+3ZJi0taohyzLTDb9j0Dud6IiIiIXtXvk7mS/rAlsFxJdjga2FLSaKrXrNOBz9f6TweWBoZL2hnYrkzCTpG0Yen277bv6+fUywPXSJoDPAp8qnaO7wIfBxYv13R6u+LCEREREb2s38lcm/SHM1q0NfqPmo9xmvtsWdueDqzZpt9hwGH9jRcRERHR6zrNZp2ncHBp/5Kke+uFgCVtK2lCKfY7QdLWtf57lfa7JF0tabnSvqGkW8q+KyQtXTvmCEkPlPN8oNY+vfSfKOn2bv8QEREREYNRp9/MnU1T4WBJWwE7ARuUQsAnll1PAh8pxX73Ac4t/YcBpwBb2d4AuAs4qBxzOnB4OWYccGg5Zh1gT2Ddcv4fSlqkdhlblaLFYzq+44iIiIge0tFkrk3h4C8Ax9meVfrMLP/fafux0mcKsJikRQGVf0uoWrq6NNDotyZwU9kez9yCwjsBF9qeZfshqnp1SX6IiIiIKLopGrwGsLmk2yTdKGnTFn12Be4sk7FXqCaAd1NN4tZh7rd3k4Edy/ZuzC0OvBLwSG28GaUNqsUX15ZXuft3cR8RERERg1Y3k7lhwAjgnVSvRS8uT9wAkLQucDxlpauk11NN5jYC3kr1mvWI0n1f4EBJE6hqz73cGKbFeRuFf99je2Pgg+XYLVpdZBIgIiIiopcNNAECqqdkl9k28IdSQmQ54AlJI6m+fdvb9p9K/9EAjd+SLgYOL23TgO1K+xrAh2rnqEd4jaS8mm28yrU9U9I4qtevN9EkCRARERHRy7p5MvcLYGv45wRsOPCkpGWAXwFH2P5drf+jwDqS3lx+bwtMLcc3Cgq/DvgG8KPS53JgT0mLlnzW1akmjktIWqocswTVRLDfwsURERERvaajJ3NtCgefCZxZypW8DOxj25IOAlYDjpJ0VBliO9uPSfo2cJOkV6jyVj9d9u8l6cCyfRlwFoDtKeUJ3j1UuawH2n5V0grAuPJWdxhwvu2rB/xXiIiIiBikOprM9VHw95Mt+n4H+E6bcX7E3Kdu9fZTqMqWtDrmWODYprYHgQ1b9Y+IiIgYSrp5zRoRERER/58NOAFC0kUlfWFiSWOYWNrH1tonSdqldsxXS1rEZEkXSFqstJ9R+t4l6RJJS5b2LSTdIWm2pI81XdPxZZzJkvZYEH+MiIiIiMFmwAkQtvco6QujgUupvnWDaiHCmNK+PfBjScMkrQR8uexbD1iEKt0B4Ku2NyzJEA8zNxniYarv6s6vn1vSh4CNqVbIbgYcWo8Ai4iIiBgqukmAAKDUltsduKD0fcH27LJ7MebWhYPqG703lGivxZlbZuTZ2lhvaBxje7rtu4A5TaddB7jR9mzbzwOTaJpsRkRERAwFC+Kbuc2Bv9q+v9EgaTNJU6jSHg4ok65HqfJbHwYeB/5u+9raMWcBfwHWAv67n3NOAj4oaXFJywFb8dp6dBERERFDwoKYzO1FeSrXYPs22+sCmwJHSFpM0giqrNVVqRIglpD0ydoxnyntU4E+v4Erk8CrgN+Xc99CVbpkHkmAiIiIiF7WTQIE5XXpR4FNWu23PVXS88B6VJO4h2w/UY69DHg38D+1/q9KuogqHuysvs5dL1ki6Xzg/jb9kgARERERPavbJ3PvB6bZntFokLRqmeQhaRVgTWA61evVd5ZXowK2AaaqslrpL+AjwLS+TippEUnLlu0NgA2Aa/s6JiIiIqIXDTgBwvYZVKtRL2jq/l7g8JLyMAf4ou0nqaK+LgHuoHoleifVEzMB55TVqKL6Hu4L5bybUmW8jgA+Iunb5fXt64HflgSIZ4FP1hZdRERERAwZXSVA2P50i7ZzgXPb9D+aKgqs2Xva9P8jMLJF+0tUK1ojIiIihrQkQEREREQMYt0kQIyWdGtJerhd0timYzaV9GojuUHSVrVkiImSXpK0c9n321r7Y5J+Udp3KqkQjXO8t7SvImlCaZ8i6YAF9QeJiIiIGEw6Xc16NnAq8LNa23eBb9v+taQdyu8toVqgABwPXNPobPt6qsQGJL0JeICyaMH25o1+ki4Ffll+XgdcbttlocPFVHXoHgfebXtWif6aLOly2491fusRERERg183CRAGGhFab6SkORRfoor4mtlmyI8Bv7b9Qr1R0lLA1sAvynmfs90oJ7IEc5MhXrY9q7Qv2ul9RERERPSaburMfQW4RtKJVJOpdwOUDNZdqCZlm7Y5dk/gey3adwGua8R7lfF2Af4TWB74UK19ZeBXwGrAoXkqFxEREUNRN0+0vgB81fbKwFeBM0r794Gv23611UGSVgTWp/YKtqZVmsQ422sBOwPH1Nofsb0B1WRuH0krtDlfEiAiIiKiZ3XzZG4f4OCy/XPg9LI9Briw1IBbDthB0mzbvyj7dwfG2X6lPlgpAjyW6uncPGzfJOkdkpYrdesa7Y+VHNjNgUtaHJcEiIiIiOhZ3TyZewx4X9nemhKnZXtV26Nsj6KaXH2xNpGDFk/fit2AK0sNOQAkrVZSIZC0MTAc+JukkZLeUNpHUNWpu7eLe4mIiIgYlAacAAF8DjilRHe9BOzfwTijgJWBG1vs3hM4rqltV2DvkibxIrBHWdm6NnCSJFOlRpxo++5O7iUiIiKil3SVAAFs0s9xn276PR1YqU3fLVu0HU9V4qS5fTxVHmtERETEkJaSHhERERGDWL+TuTbpDxtKukXS3ZKukLR0aR8l6cVamsOPSvtSTekPT0r6ftm3haQ7JM1upEU0nX9pSY9KOrW/sSIiIiKGmk5es57NvOkPpwOH2L5R0r7AocBRZd+fbI+uD2D7H5T0BwBJE4DLys+HgU8Dh7Q5/zHUvrHrZ6yIiIiIIaXfJ3Nt0h/WBG4q2+OpFip0RNLqVAWAf1vGn277LmBOi76bACtQYr/6GysiIiJiqBnoN3OTgR3L9m5UK1QbVpV0p6QbJW0+76HsBVxUi+lqSdLrgJOonvq109FYEREREb1qoJO5fYEDyyvOpYCXS/vjwNtsbwR8DTi/8T1dzZ60rjPX7IvAVbYf6aNPv2MlASIiIiJ62YASIGxPA7YDkLQGJTPV9ixgVtmeIOlPwBrA7aXvhsAw2xM6OM27gM0lfRFYEhgu6Tnbh8/PWEmAiIiIiF42oMmcpOVtzyyvQr8BNFatvhl4yvarkt4OrA48WDu0XfrDPGx/ona+TwNjGhO5+R0rIiIiold1UprkAuAWYE1JMyTtB+wl6T5gGlWs11ml+xbAXZImUUV5HWC7vnhid5omYJI2LakSuwE/LjmrnZhnrIiIiIihpt8nc32kP5zSou+lwKV9jPX2Fm1/BEb2cw1nU5VI6XOsiIiIiKEmCRARERERg1gnr1lXlnS9pKmSpkg6uLS/SdJ4SfeX/0fUjtmypDNMkXRjrX0ZSZdImlbGe1dp/1ZJeWikOuxQO2aDkjYxpSROLFbaj5X0iKTnFuQfJCIiImIw6eTJ3Gzg/9heG3gnVUmSdYDDgetsrw5cV34jaRngh8COttel+hau4RTgattrARsCU2v7TrY9uvy7qow1DPgfqm/v1gW2BF4p/a8Axg7gniMiIiJ6RicJEI/bvqNs/4NqArYSsBNwTul2DrBz2f44cJnth8sxM6HKWKVaIHFGaX/Z9jP9nH474C7bk8oxf7P9atm+1fbjnd5oRERERC+ar2/mJI0CNgJuA1ZoTKbK/8uXbmsAIyTdIGmCpL1L+9uBJ4CzSkLE6ZKWqA1/kKS7JJ1Ze2W7BmBJ10i6Q9JhA7nJiIiIiF7V8WRO0pJUK1W/YvvZProOAzahKiT8AeCoUlh4GLAxcFpJiHie8moWOA14BzCaKkXipNpY7wU+Uf7fRdI2nV5zue4kQERERETP6qhosKTXU03kzrN9WWn+q6QVbT8uaUVgZmmfATxp+3ngeUk3UX0f91tghu3bSr9LKJM523+tneunwJW1sW60/WTZdxXVhPC6Tm8wCRARERHRyzpZzSqq79ym2v5ebdflwD5lex/gl2X7l1QxXMMkLQ5sVo79C/CIpDVLv22Ae8o5VqyNuwswuWxfA2wgafGyGOJ9jWMiIiIiorMnc+8BPgXcLWliafs34Djg4pII8TBl1artqZKuBu4C5gCn225Mzr4EnCdpOFXM12dK+3cljQYMTAc+X8Z6WtL3gD+WfVfZ/hWApO9SLbZYvCRInG77WwP6K0REREQMUp0kQNwMqM3ult+v2T4BOKFF+0RgTIv2T/Vx/v+hKk/S3H4YkAURERERMaQlASIiIiJiEOsmAWK38nuOpDG1/p+oJTlMLPtHl303SLq3tm/50r6KpOtKaZIbJI0s7Vs1jfWSpJ3LvrMlPVTbN3ph/IEiIiIi/pV18s1cIwHiDklLARMkjadapPBR4Mf1zrbPA84DkLQ+8MvyerXhE7ZvbzrHicDPbJ8jaWvgP4FP2b6eqlwJkt4EPABcWzvuUNuXdHivERERET1nwAkQtqfavrefw/cCLujgOtZhbrmR66nSJZp9DPi17Rc6GC8iIiJiSOgmAaITezDvZO6s8lr0qFL2BGASsGvZ3gVYStKyTcft2WKsY8ur2ZMlLdrhNUVERET0jIWRANHovxnwQq0sCVSvWNcHNi//GqtYDwHeJ+lOqlpyj1K93m2MtSKwPlXduYYjgLWATYE3AV9vcx1JgIiIiIie1U0CRH/meZJm+9Hy/z8knQ+MpfpW7jGq7+8ak8Zdbf+9dujuwDjbr9TGerxszpJ0FtWEcB5JgIiIiIhe1k0CRF/HvI6qiPCFtbZhkpYr268HPkxJepC0XDkGqiduZzYNOc+3d43UiHJ9OzM3NSIiIiJiyOgmAWJR4L+BNwO/kjTR9gfK/i2oclgfrI2zKHBNmcgtAvwv8NOyb0vgPyUZuAk4sHFQ+U5vZeDGpus6T9KbqQoaTwQO6OBeIiIiInpKtwkQ49occwPwzqa254FN2vS/BGhZYsT2dGClFu1bt7vmiIiIiKEiCRARERERg1g3CRAnSJpWSoOMk7RMaX+9pHMk3V2OOaI21vTSPlHS7bX2Y8o4EyVdK+mtpf2Nkq6QNKmc+zO1Y46XNLn822NB/lEiIiIiBotOnsw1EiDWpnp1eqCkdYDxwHq2NwDuo1q4ANXCh0VLCZJNgM+X794atrI92vaYWtsJtjewPRq4EvhmaT8QuMf2hlTf1Z0kabikDwEbU6VDbAYcKmnp+bz3iIiIiEGvmwSIa203asHdCoxsHAIsIWkY8AbgZaDPunRNdeuWKGM0xlqqrFhdEniKanK5DnCj7dnlW7xJwPb93UtEREREr1lQCRD7Ar8u25cAzwOPAw8DJ9p+quwzcK2kCZL2bxr7WEmPAJ9g7pO5U4G1gceAu4GDbc+hmrx9UNLipdzJVlQrXltdc4oGR0RERM/qqGgwtE+AkHQk1dOy80rTWOBV4K3ACOC3kv63lCl5j+3HJC0PjJc0zfZNALaPBI4s39gdBBwNfICq7MjWwDvKMb+1fa2kTYHfA08At1BLjKhL0eCIiIjoZR09mWuXACFpH6riv5+w3ZgofRy42vYrtmcCvwPGAJSkB0r7OKqJX7PzmZvT+hngMlceAB6iivDC9rHl27ttqUqn3N/5bUdERET0hgEnQEjanioPdUfbL9QOeRjYWpUlqBZNTJO0hKSlyrFLANsxNwFi9drxOwLTamNtU/qsAKwJPChpEUnLlvYNgA2Aa+f35iMiIiIGu24SIP6LKtVhfDXf41bbBwA/AM6imqgJOMv2XZLeDowrfYcB59u+uox3nKQ1gTnAn5mb5nAMcLaku8tYX7f9pKTFqF7fQrW44pO1xRgRERERQ0Y3CRBXten/HFV5kub2B4EN2xyza5v2x6ie4DW3v0S1ojUiIiJiSEsCRERERMQg1ukCiHYpEO2SG0aUVIi7JP1B0npN4y0i6U5JV9batpF0RxnrZkmrlfZFJV0k6QFJtzUKEEsaW/pOLAkRuyyYP0lERETE4NHpk7l2KRDtkhv+DZhY0iH2Bk5pGu9gquLDdadRrYodTbWi9RulfT/gadurAScDx5f2ycCY0n974MelUHFERETEkNHRZK6PFIh2yQ3rANeV/tOAUWU1KpJGAh8CTm8+DdCI5HojVaFggJ2Ac8r2JcA2kmT7hdqih8Vq546IiIgYMub7m7nmFIg2yQ2TgI+W/WOBVZgb9/V94DCqlat1nwWukjSDavXscaV9JeARgDJ5+zvQKEuymaQpVOkQB7Ra0ZoEiIiIiOhl8xvnNU8KhO0jba9MlQBxUOl6HDCilDL5EnAnMFvSh4GZtie0GP6rwA62R1KVNmnUtGu1ktbl3LfZXhfYFDiilCx5bUf7J7bH2B6z//57zM/tRkRERPzL63gy1y4FouafyQ22n7X9mfI9297Am6nSG94D7ChpOnAhVXHh/5H0ZmBD243M14uAd5ftGZTc1fJN3BuBRtYr5XxTqfJgX7PQIiIiIqLXdbqatV0KRMvkBknLSBpe2j8L3FQmeEfYHml7FLAn8BvbnwSeBt4oaY1yzLbMXSBxObBP2f5YOcaSVm0seJC0ClU6xPTObz0iIiJi8Ot09We7FIj92iQ3rA38TNKrwD1UK1Lbsj1b0ueASyXNoZrc7Vt2nwGcK+kBqidye5b29wKHS3qlnP+Ltp/s8H4iIiIiekJHk7kBpEDcAqzeal+tzw3ADbXf44BxLfq9ROtEiXOBc/s6R0RERESvSwJERERExCDW72Suj/SHb0l6tJbCsEPtmCNKYsO9kj7QNF6r9IfzSt/Jks4siy2QtJakWyTNknRIrf+atfNOlPSspK8siD9IRERExGDSyWvWRvrDHZKWAiZIGl/2nWz7xHrnkgyxJ7Au8FbgfyWtYfvV0qWR/rB07bDzgE+W7fOpFk2cRvWN3JeBnevnsH0vMLqcbxHgUVq8oo2IiIjodf0+mWuX/tDHITsBF9qeZfsh4AFgLLRPf7B9lQvgD5QCw7Zn2v4j8Eof59sG+JPtP/d3LxERERG9Zn6LBo+ilv4AHCTprvJqdERp+2diQzGDuZO/dukPjfFfT7Vq9ur5uKw9gQv6uOYkQERERETPmp+iwc3pD6cB76B63fk4cFKja4vD3U/6Q8MPqWrS/bbDaxpOVd/u5+36JAEiIiIielmnRYPnSX+w/Vfbr9qeA/yU8iqVWmJDMRJ4jDbpD7VzHE2VFPG1+bj+DwJ32P7rfBwTERER0TM6Wc3aLv1hxVq3XYDJZftyYE9Ji0palare3B/6SH9A0meBDwB7lclhp/aij1esEREREb2uk9Ws7dIf9pI0mir0fjrweQDbUyRdTJX8MBs4sLaStZ0fUSVI3FLNHbnM9r9LegtwO9XK1zml/Mg6tp+VtDhV7NfnO77biIiIiB7T72RuftMfyjHHAsf2sf8GXpv+0PI6bP+FsrK1xb4XgGXbnSMiIiJiKBhw0eCy70ul2O8USd8tbcuW/s9JOrVprKslTSr9f1RqxPU11thaYeBJknap9T9T0kxJk4mIiIgYoropGrwCVU25DWzPkrR86f8ScBSwXvlXt3t5RSrgEqrM1QslbdVmrMnAGNuzyzd6kyRdYXs2cDZwKvCzAd57RERExKDXTdHgLwDH2Z5V9s0s/z9fXs2+1GKsZ8vmMGA41fd29DHWC2XiBrBYrT+2b6JKiIiIiIgYsropGrwGsLmk2yTdKGnTDse4BpgJ/IPq6Rx9jSVpM0lTgLuBA2qTu4iIiIghr5uiwcOAEcA7gUOBi8vr0z7Z/gCwIrAosHVpbjuW7dtsrwtsChwhabFOr7lcdxIgIiIiomd18s1cy6LBVMWBL2vkqUqaAywHPNHfeLZfknQ51Xdy4zsZy/ZUSc9TfYd3e6c3aPsnwE+qX/e5z84RERERg8yAiwYDv6A8WZO0BtU3cE/2Mc6SjULDkoYBOwDT+hpL0qqlL5JWAdakqmkXEREREXRXNPhM4MxSGuRlYJ/yZI0S2bU0MFzSzsB2wN+AyyUtCiwC/IaqWDDtxpL0XuBwSa8Ac4Av2n6ynOMCYEtgOUkzgKNtnzHwP0VERETE4NNN0WCAT7Y5ZlSb/i0XSdh+udVYts8Fzm1zzF5tzhERERExZMzXataIiIiI+NeyUCZzkhaT9Ida2sO3m/b/t6Tnar8PkHR3SXq4WdI6pb2vBIjptWM6XhARERER0Us6Ws06ALOArW0/V1bC3izp17ZvlTQGWKap//m2fwQgaUfge8D29J0AAbBV4xu6iIiIiKFooUzmykKIxpO315d/LlmsJwAfB3ap9X+2dvgSlKQH2y/U2l+TABERETFQb3jb0Qt1/BcfvmChnuPFhy9YaGP/v7Sw/0aDffxOLawnc5SJ2wRgNeAHtm+TdDBwue3Hm+sLSzoQ+BpVWZKta+2bUa12XQX4VO2pnIFrJRn4caknFxER0a8XH/52/50GwTkGu4X9Nxrs43dqoS2AsP2q7dHASGCspC2A3YD/pGAnzwAAIABJREFUbtP/B7bfAXwd+EatvV0CxHtsbwx8EDiwjD+PJEBEREREL1toT+YabD8j6QZgK6qndA+Up3KLS3rA9mpNh1wInNZinNckQNh+rLTPlDQOGAvc1OK4JEBEREREz1pYq1nfLGmZsv0G4P3ABNtvsT2q1KF7oTGRk7R67fAPAfeX9pYJEJKWkLRUaV+Cqijx5IVxLxERERH/yhbWk7kVgXPKd3OvAy62fWUf/Q+S9H7gFeBpYJ/S3jIBQtLbgXHlCd8wqtWwVy+ke4mIiIj4l7WwVrPeBWzUT58la9sHt+nTMgHC9oPAhl1eZkRERMSglwSIiIiIiEGsq8mcpDMlzZQ0z/dqkg6RZEnLld87Sbqrkdgg6b2lfataysNESS9J2rlprObEiFUkXVfGu0HSyG7uIyIiImKw6vbJ3NlUSQ2vIWllYFvg4VrzdcCGpVzJvsDpALavtz26tG8NvABcWxurVWLEicDPbG8A/Dvwn13eR0RERMSg1NVkzvZNwFMtdp0MHEYtscH2cyUZAmopD00+Bvy6kfxQS4w4rKnfOlSTQ4DrgZ0Geg8RERERg9kC/2auZKs+antSi327SJoG/Irq6VyzPYF6fsVBlMSIpn6TgF3L9i7AUpKW7friIyIiIgaZBTqZk7Q4cCTwzVb7bY+zvRawM3BM07ErAusD15Tfb6V9YsQhwPsk3Qm8D3gUmN2iXxIgIiIioqct6NIk7wBWBSaVGnAjgTskjbX9l0Yn2zdJeoek5Ww/WZp3B8bZfqX83og2iREl/eGjAJKWBHa1/fdWF5QEiIiIiOhlC3QyZ/tuYPnGb0nTgTGl0O9qwJ9sW9LGwHDgb7XD9wKOqI31K+AttbGeqyVGLAc8ZXtOOebMBXkfEREREYNFt6VJLgBuAdaUNEPSfn103xWYLGki8ANgj8aCCEmjgJWBGzs89ZbAvZLuA1YAjh3QDUREREQMcl09mbO9Vz/7R9W2jweOb9NvOrBSP2PVEyMuAS6Zj0uNiIiI6ElJgIiIiIgYxLp9zbqypOslTZU0RdLBpf1bkh6tpTrsUNrH1tomSdqlNlZfaRJfknRvOcd3+xsrIiIiYqjodgHEbOD/2L5D0lLABEnjy76TbZ/Y1H8y1YKI2aUUySRJV9ieTZUmcSrws/oBkraiKgq8ge1ZkpbvYKyIiIiIIaHbb+YeBx4v2/+QNJU+vn1rJDsUi/HahIibykKIZl8AjrM9q/Sb2d9YEREREUPFAvtmrkzENgJuK00HSbqrvD4dUeu3maQpwN3AAR08SVsD2FzSbZJulLRpF2NFRERE9JQFMpkrhXsvBb5i+1ngNKoCwqOpntyd1Ohr+zbb6wKbAkdIWqyf4YcBI4B3AocCF6tUEe5krCRARERERC/rumiwpNdTTeTOs30ZgO2/1vb/FLiy+TjbUyU9D6wH3N7HKWYAl5WadH+QNAdYDniik7GSABERERG9rNvVrALOAKba/l6tfcVat12oFisgaVVJw8r2KsCawPR+TvMLYOtyzBpUyRFPDnCsiIiIiJ7S7ZO59wCfAu4uyQ4A/wbsJWk01aKE6cDny773AodLegWYA3yxkc1a0iS2BJaTNAM42vYZVFFdZ5aSJS8D+5RIsLZjRURERAwV3a5mvRlQi11Xtel/LnBum30t0yRsvwx8cn7GioiIiBgqkgARERERMYgNeDInaTFJfyjpC1Mkfbu0S9Kxku4ryRBfLu2fKKVK7pL0e0kb1saaLunukuZwe639olrKw/TGq9wy1sTavznltW5ERETEkNLNa9ZZwNa2nysrWm+W9GtgbWBlYC3bc2qJDQ8B77P9tKQPUq0w3aw23lbN37zZ3qOxLekk4O+l/TzgvNK+PvBL2xOJiIiIGGIGPJkrpUKeKz9fX/6ZKrHh47bnlH6NxIbf1w6/FRjZ6bnKqtndKatam+wFXDC/1x8RERHRC7otTbJIefU5Exhv+zaqYsF7lEK9v5a0eotD9wN+Xftt4FpJEyTt36L/5sBfbd/fYt8eZDIXERERQ1RXkznbr9oeTfWUbayk9YBFgZdsjwF+SlVa5J8kbUU1mft6rfk9tjcGPggcKGmLplO1fPomaTPgBduT211jEiAiIiKil3WdAAFg+xlJNwDbUyU2XFp2jQPOavSTtAFwOvBB23+rHf9Y+X+mpHHAWOCmcsww4KPAJi1OvSf9PJVLAkRERET0sm5Ws75Z0jJl+w3A+4Fp1BIbgPcB95U+bwMuAz5l+77aOEtIWqqxDWxHSYwo3g9Msz2j6fyvA3YDLhzoPUREREQMdt08mVsROEfSIlSTwottXynpZuA8SV+lWiDx2dL/m8CywA+r9QzMLq9iVwDGlbZhwPm2r66dp93Tty2AGbYf7OIeIiIiIga1blaz3gVs1KL9GeBDLdo/y9yJXb39QWDD5vba/k+3ab8BeGfHFxwRERHRg5IAERERETGIdT2ZK+VJ7pR0Zfn921oyw2OSflHa15J0i6RZkg5pGuNgSZNLksRXau0blmPulnSFpKVL+7KSrpf0nKRTu72HiIiIiMFqQTyZOxiY2vhhe3Pbo0vJkluoFj0APAV8GTixfnApZ/I5qhWsGwIfrtWmOx043Pb6VCtjDy3tLwFHAa+ZFEZEREQMNd0WDR5J9X3c6S32LUW1qvUXUJUdsf1H4JWmrmsDt9p+wfZs4EZgl7JvTUqJEmA8sGsZ63nbN1NN6iIiIiKGrG6fzH0fOAyY02LfLsB1tp/tZ4zJwBbl1eniwA5U2a6NfTuW7d1q7R1L0eCIiIjoZQNezSrpw8BM2xMkbdmiy160eGLXzPZUScdTPXl7DpgEzC679wX+S9I3gcuBl+f3OlM0OCIiInpZN0/m3gPsKGk6VeHerSX9D1QLFKi+gftVJwPZPsP2xra3oPq27v7SPs32drY3oao196curjciIiKi5wx4Mmf7CNsjbY+iKuz7G9ufLLt3A6603dE3bZKWL/+/jSq664Km9tcB3wB+NNDrjYiIiOhFCySbtYU9gePqDZLeAtwOLA3MKSVI1inf1F1anua9Ahxo++ly2F6SDizbl/HanNfpZazhknYGtrN9z0K6n4iIiIh/SQtkMlfSGG6o/d6yRZ+/ACPbHL95m/ZTgFPa7Bs13xcaERER0WOSABERERExiC2IBIjpJaFhoqTbS9toSbc22iSNLe1tUyDK/tekSZS2VSXdJul+SRdJGl7aPy3piVraxDy5rxERERG9bkE9mduqpD6MKb+/C3y7pEB8s/yGNikQNa9JkyiOB062vTrwNLBfbd9FjbQJ2/2WQYmIiIjoNQvrNaupFicAvBF4DPpMgWiZJiFJVCkSl5Smc4CdF9I1R0RERAw6C2IyZ+BaSRMk7V/avgKcIOkRqqdwR3QwTqs0iWWBZ0rMF8AMYKXa/l0l3SXpEkkt0yGSABERERG9bEGsZn2P7cdKTbjxkqYBHwO+avtSSbsDZwDvbzdAH2kSatG9keJwBXCB7VmSDqB6arf1PJ2TABERERE9rOsnc7b/+QoVGEeV/LAPVV04gJ+Xtr60S5N4ElhGUmPSOZK5r2z/ZntWaf8psEm39xIREREx2HQ1mZO0hKSlGtvAdsBkqgnX+0q3rSnxXO20S5OwbeB6qid9UE0Sf1nOt2JtiB2Zd+FERERERM/r9jXrCsC4ap0Cw4DzbV8t6TnglPJE7SVgf+g3BaKdrwMXSvoOcCfVK1uAL0vaEZhNtUr2013eS0RERMSg09VkzvaDwIYt2m+mxWvPvlIgan1u4LVpEg/S4jWt7SPobGFFRERERM9KAkRERETEILYgEiCWKaVBpkmaKuldkjYsSQ93S7pC0tK1/kdIekDSvZI+UGv/qqQpkiZLukDSYqX9oNLfkpar9R8haVwpTfIHSet1ey8RERERg82CeDJ3CnC17bWoXrlOpSr8e7jt9alWuB4KIGkdqgUO6wLbAz8sEV4rUSVDjLG9HrBI6QfwO6qyJn9uOu+/ARNtbwDsXa4jIiIiYkjpdjXr0sAWlEUJtl+2/QywJnBT6TYe2LVs7wRcaHuW7YeAB5j7Pdww4A1l0cTizC1Bcqft6S1Ovw5wXekzDRglaYVu7iciIiJisOn2ydzbgSeAsyTdKen0UqJkMlW5EIDdgEY6w0rAI7XjZwAr2X6UKiniYeBx4O+2r+3n3JOAjwJIGgusQovFFUmAiIiIiF7W7WRuGLAxcJrtjYDngcOBfYEDJU0AlgJeLv1bJjpIGkH11G5V4K3AEpI+2c+5jwNGSJoIfImqbMns5k62f2J7jO0x+++/x3zfYERERMS/sm4nczOAGbZvK78vATa2Pc32drY3AS4A/lTrX89QbSQ6vB94yPYTtl+hSo94d18ntv2s7c/YHk31zdybgYe6vJ+IiIiIQaWryVypG/eIpDVL0zbAPSWnFUmvA74B/KjsvxzYU9KiklYFVgf+QPV69Z2SFldVgXgb+kl0KKtoh5efnwVu6qf4cERERETPWRCrWb8EnCfpLmA08B/AXpLuA6ZRPXk7C8D2FOBi4B7gauBA26+WJ3uXAHcAd5fr+gmApC9LmkH1FO8uSaeX864NTJE0DfggcPACuJeIiIiIQaXbOC9sTwTGNDWfQptSIbaPBY5t0X40cHSL9v8C/qtF+y1UT/YiIiIihqwFUTR4nmK/klaVdJuk+yVd1HgdWl6vXlSKAN8maVRp31bShFJkeIKkrWvjXy1pUjnHjyQtUtqPKQWDJ0q6VtJbu72XiIiIiMGm2zpz7Yr9Hg+cbHt14Glgv3LIfsDTtlcDTi79AJ4EPlKKDO8DnFs7ze62NwTWo1rksFtpP8H2BmUBxJXAN7u5l4iIiIjBaEF8M9dc7PdxYGuqb+AAzgF2Lts7ld+U/dtIUikM/FhpnwIsJmlRqFat1s4zHHBTO8ASjfaIiIiIoaTb1azzFPsFJgDP2G7UfJtBVSwYakWDy/6/A8s2DbsrcKftWY0GSdcAM4F/MHeSiKRjJT0CfII8mYuIiIghqNvXrPMU+6VaWdqs8dSsZdHg2njrUr16/fxrOtgfAFYEFqV66tdoP9L2ysB5wEFtrjEJEBEREdGzul3N+s9ivwCSGsV+l5E0rDx9axQGhrlFg2eU17JvBJ4qx44ExgF72/4TTWy/JOlyqsnj+Kbd5wO/ovVq2J9QypzAfXkVGxERET2l22/mWhX7vQe4HvhY6bMP8MuyfXn5Tdn/G9uWtAzVZOwI279rDC5pSUkrlu1hwA5UteuQVC9LsmOjPSIiImIo6erJnO3bJDWK/c6mykf9CdXE7EJJ3yltZ5RDzgDOlfQA1RO5PUv7QcBqwFGSjipt21G9lr28LIZYBPgNc9MkjivJE3OAPwMHdHMvEREREYPRgiga3KrY74PA2BZ9X2JuaZF6+3eA77Q5xaZtzrvr/F1pRERERO9ZEKVJIiIiIuL/k25Xsx5ckh+mSPpKaXuTpPEl/WF8WfGKpBGSxpXUhj9IWq82zjKSLpE0TdJUSe9qOs8hkixpufL70JL8MLGc/1VJb+rmXiIiIiIGowFP5spk7HNUr1M3BD5cFiUcDlxX0h+uK78B/g2YaHsDYG9em916CnC17bXKWFNr51kZ2JZqsQUAtk+wPbqkPxwB3Gj7qYHeS0RERMRg1c2TubWBW22/UEqQ3AjswmtTHurpD+tQTe6wPQ0YJWkFSUsDW1AWSdh+2fYztfOcDBxG+4SHvYALuriPiIiIiEGrm8ncZGALSctKWpyqbMjKwAq2Hwco/y9f+k8CPgogaSywClUNurcDTwBnSbpT0umSlij9dgQetT2p1QWU824PXNrFfUREREQMWgOezNmeSpXWMB64mmqyNruPQ44DRkiaCHyJqmTJbKoVtRsDp9neCHgeOLxM1I6k75iujwC/6+sVaxIgIiIiopd1W2fuDMrrUUn/QZXw8FdJK9p+vBT8nVn6Pgt8pvQV8FD5tzgww/ZtZdhLqL6zewdVTNikqjsjgTskjbX9l9J3T/p5xZoEiIiIiOhl3a5mXb78/zaqV6gX8NqUh3+mP5QVq8NL+2eBm2w/WyZmj5QCwFBSJGzfbXt526Nsj6KaKG7cmMhJeiPwPuamS0REREQMOd0WDb5U0rLAK8CBtp+WdBxwsaT9qFagNooErw38TNKrVJFf+9XG+RJwXpnsPUh5gtePXYBrbT/f5T1ERMQQ84a3zRPlvUC9+PAFC/UcLz7cG+v+FvbfaLCP36luX7Nu3qLtb1RP15rbbwFWb24v+yYCY/o516im32cDZ3d8sREREcWLD3+7J84x2C3sv9FgH79TSYCIiIiIGMQ6msxJOlPSTEmTa23tkh7WknSLpFmSDmkaZ57EiL7GKvu2LEkPUyTd2Nc1RURERAw1nT6ZO5uqnltdu6SHp4AvAyfWO/eRGNF2LEnLAD8EdrS9LnO/v2t3TRERERFDSkeTOds3UU3S6lomPdieafuPVIsi6tolRrQdC/g4cJnthxtj93NNEREREUNKN9/MtUt6aKddYkRfY61BVWj4BkkTJO3dxfVGRERE9Jz/ZwsgBpAYAdVq202ADwEfAI6StMb8nDcJEBEREdHLuilN0jLpoS9tEiP6GmsG8GSpJfe8pJuovre7r9OLTAJERERE9LJunsy1THroS5vEiL7G+iWwuaRh5dXsZsDULq45IiIioqd09GRO0gXAlsBykmb83/bOPN7Wsfz/74+ZMqSkTBFSlCMhoUQp+pWSJNEgGaI4lEoqQ/UVmvWVFCeJBqGvBnMyzxyzUopKJUlEMn1+f1z3OnvtddbeZ3jutdZe+1zv12u/9n7utdd133vttZ7neu5r+AAHAV2VHiQ9B7gGWAJ4qrQgWbNos86kGFGm6GrL9m2SzgJuBJ4Cvm375rHWVHb+kiRJkiRJ5hlmy5mzvcMYD3VTevgrsMIYdmZSjCjjXVUjymNHAkfOwZqSJEmSJEnmGVIBIkmSJEmSZIhpogCxXVFleErSem3jW5Q2IjeV75u3PbZDGb9R0lmSntUxz0ckuTUuaf+i/jC9KEc8KWnp8tgfiq3pkq5p+kIkSZIkSZIMI00UIG4mihgu6hi/D3iT7ZcQxQwnAkhaAPgqsJnttYk8uA+2niRpRWALImcOiBCr7XVsrwMcAFxou71R8Gbl8fVIkiRJkiSZB5lrBQjbt9n+dZffvd72PeXwFmARSQsDKl9PkySiQOKetqd+GfgoMFb7kB0YqX5NkiRJkiRJ6H3O3LbA9bb/a/tx4APATYQTtyYjPee2Bv5s+4ZuRkpbki2BU9uGDZxTQrm79fBvSJIkSZIkmbD0zJmTtBah+LB7OV6QcOZeCixHhFkPKI7agcCnxzH3JuDSjhDrxrbXBbYC9pL0qjHWkQoQSZIkSZJMWpooQIyJpBWA04F32/5dGV4HoHUs6UfAx4nGwKsAN0T0lRWA6yRtUNqcALyDjhBrK5Rr+15JpwMbMHP+XipAJEmSJEkyqam+MydpKeDnwAG2L2176M/AmpKWKcdbALfZvsn2s22vbHtlQsJr3ZYjJ2lJYFPaFCYkPU3S4q2fgdcRBRlJkiRJkiTzFE0UIO4HjgKWAX4uabrt1xMVqqsBn5L0qWLidbbvkXQIcJGkx4G7gPfOxvTbAOcUfdYWywKnl528BYCTbZ81O39LkiRJkiTJZKKpAsTpXX73s8Bnx7BzDHDMLOZaueP4O0RrlPaxO4Ep49lJkiRJkiSZF0gFiCRJkiRJkiGmiQLEkZJuL2oOp5dcOSRt0KbacIOkbcr4Gm3j0yU9KGlqeWwdSVe01BwkbVDGl5T002LnFkk7t81/eFGFuFnS9jVflCRJkiRJkmGhiQLEucCLi5rDbwiFBohChPWKasOWwDclLWD7121qDi8DHmEkTHsEcEh57NPlGGAv4FbbU4icvS9KWkjS/wPWJSpkXw7sL2mJOfi7kyRJkiRJJgVNFCDOsf1EObyCaCmC7Ufaxhehu6LDa4Df2b6rZY5QhABYkhFlCAOLF8WIp5c1PEE0HL7Q9hOlMOIGZnY2kyRJkiRJJj21cubeB5zZOpD0ckm3EGoPe7Q5dy06+8ZNBY6U9EfgC4zs8n0deBHh3N0E7GP7KcJ520rSYpKeBWwGrFjpb0mSJEmSJBkaGjtzkg4kdstOao3ZvtL2WsD6hMrDIm2/vxCwNXBKm5kPAPvaXhHYlyLzBbwemE4oRqwDfF3SErbPAX4BXEY4hZeXNXRbXypAJEmSJEkyaWnkzEl6D/BGYEfbM4VTbd8GPAy8uG14K+A6239rG3sPcFr5+RRCzQFgZ+A0B78Ffg+8sNj+XMnB2wIQcEe3Ndo+1vZ6ttfbbbesk0iSJEmSZHIx186cpC2BjwFb236kbXwVSQuUn58HrAH8oe2pO9AhzUWEUTctP2/OiGN2N5Ffh6Rli607Jc0v6ZllfG1gbeCcuf1bkiRJkiRJhpUmChAHAAsD5xYlhits7wFsAny8qDw8Bexp+75iZzFCxmv3jil2Bb5anMBHgd3K+GeA70i6idh9+5jt+0rY9uIy74PATl3y8pIkSZIkSSY9TRQgjusyhu0TgRPHeOwR4Jldxi8h2pV0jt9D6K52jj9KVLQmSZIkSZLM06QCRJIkSZIkyRDTRAHiM0X9YbqkcyQt1/Gc9SU9Kelt5XizDgWIRyW9pTx2XFF5uFHSjyU9vYwvLOmHkn4r6UpJK5fxrioTSZIkSZIk8xpNFCCOtL12UW34GaHcAICk+YHDgbNbY7YvaFOA2JxQgGgVLexre0pRk7gb+GAZ3wX4p+3VgC8XmzCGysRs/i1JkiRJkiSThiYKEA+2HT6N0UoPHwJOBe4dw+TbgDNbVbAtW0XpYdE2W28GTig//xh4jSTNpspEkiRJkiTJpKdpn7nPFdWGHSk7c5KWB7YBjhnnqZ0KEEiaBvyV6CN3VBleHvgjQHHe/kUpoJgNlYkkSZIkSZJJTyNnzvaBRbXhJEZCo18hWog82e05kp4LvIS2EGyxtTOh9HAb0Oruq27Tlt8fU2WiY75UgEiSJEmSZNJSK8/sZODnRP+59YAflB5wzwLeIOkJ2z8pv/t24HTbj3casf2kpB8C+wPTgD8Rmqt/KjlxSzJzuPc2SS2ViWu62DwWODaOfpPh2CRJkiRJJhVNFCBWbzvcGrgdwPYqtle2vTKR57ZnmyMHHQoQClZr/Qy8qWULOIOQ+oLIs/ulbc+GykSSJEmSJMk8QRMFiDdIWoNQebgL2GM27KxM7LRd2D4MnCBpifLzDcAHymPHASdK+i2xI/eOMj6mykSSJEmSJMm8RHUFiI7nvbfj+A9EUUP72FPAxmM8/1Fguy7jY6pMJEmSJEmSzEukAkSSJEmSJMkQk85ckiRJkiTJEJPOXJIkSZIkyRCTElhJkiTJPMeiKx3UU/v/ufv7PZ3jP3d/f9a/NAT0+jUadvuzje38GuML2C3tT+6/IV+jyW9/MvwNw25/MvwNw25/MvwNw26/l3NkmHV8dkv7A59j2O33Y460P/g50v7g50j7g58j7Q9ojnTmkiRJkiRJhph05pIkSZIkSYaYdObG59i0P/A5ht1+P+ZI+4OfI+0Pfo60P/g50v6A5lBJyEuSJEmSJEmGkNyZS5IkSZIkGWLSmUuSJEmSJBli0pmbJEiaX9K+g15HkiRJMnjKNWHbQa8j6Q+ZM1eQJGADYHnAwD3AVR6iF0jSr2y/etDrSEDSO4BVbX9O0orAs21fO+h1zQ6SFga2BVamTSXG9qGV7K9i+/ezGkuSWSFpE2B129MkLQM8veb7SNKiwEq2f13LZof95xHrP6/MtYDthyrav9j2K2vZSyYuuTMHSHodcAdwMPAG4P8BhwB3lMdqzfMeSddJerh8XSPp3bXsA5dK+rqkV0pat/VV0f4MJD1P0mvLz4tKWryi7V26jH2+lv0utpeSdGBFe18HNgN2KkMPA8fUst82z/ySlpO0Uuurkun/A94MPEGsvfVVi1O7jP24on0AJG3Y/r6UtLikl1e0f+LsjDWwf/7sjM2F3aMkfW2sr6b22+bZUNLVkv4t6TFJT0p6sKL9g4CPAQeUoQWB71W0/yZgOnBWOV5H0hkV7e9KvO+/WYZWAH5Sy37hbElTJT1X0hKtr1rGJW0s6VxJv5F0p6TfS7qzlv0yhyTtJOnT5XglSRtUnqNn17O2OTaS9E5J72591bSf2qzBV4HX2v5D+6CkVYBfAC9qOkH5x00F9gOuAwSsCxwpCdvfbToHsFH53r6DYmDzCrZnUE5CuwFLA6sSJ6FjgNdUmuJtkh61fVKZ72hg4aZGyw7Zp4DliJPmycBngHcBNYUON7K9rqTrAWzfL2mhivaR9CHgIOBvwFNl2MDaFcyvYHvLCnZGIemFwFrAkpLe2vbQEsAitecDvkF8xlo83GWsCWu1H0iaH3hZU6OSFgEWA54l6RnEuQLidVquqX3gmvJ9Y2BN4IfleDug5u7x14F3AKcA6wHvBlaraH8b4KXE+RTb91S+CB9MRGt+VexPl7RyRft7FftXFvt3SHp2RfsAu5fvH24bM1Drxu84YF/iffNkJZudHE2c4zYnrm0PETeE69cw3ofrWesmb1Xi5qD1Ohmocd0H0plrsQDwpy7jfybu9mqwJ7BNh8P4y5LT8AMq/FNtb9bUxmzS65PQW4EzJD0FbAXcb3vPCna/C1xInAi2BK4AbgHWtv3XCvZbPC5pPuLDiqRnMuJw1WIfYA3b/6hsF+AySS+xfVNlu2sAbwSWAt7UNv4QsGvluSDSSGakSdh+SlLjc56kA4BPAIu27TQJeIw6PaR2J278liMuki1n7kHgf5sat30CgKT3ApvZfrwcHwOc09R+x1y/lTS/7SeBaZIuq2j+MduW1PqcPa2ibYAnbP9L0qx/c+74r+3HWvbLe7NqWo/tFWva68LHfZ6sAAAgAElEQVS/bJ/Z4zle3nFz/M/KN8f9cKrXA9bsZdpWOnPB8cDVkn4A/LGMrUjcVR5XaY4lOnf+AGz/oda2t6Rlgf8BlrO9laQ1gVfYrvU3tOjJSUjS0m2H7yd2zy4FDpW0tO37G06xtO2Dy89nS/obsL7t/za028n/Eg7jMpIOAd5OhO1r8kfgX5VtttgEeK+k3wP/JZwJ226062f7/4D/k/QK25dXWOesuFPS3sRuHMQNVeMQkO3DgMMkHWb7gFk+Yc7tfxX4qqQP2T6qtv02lgMWB1qfq6dTZ+evxSPlojtd0hHAX4CaDtePJH0TWKrsrrwP+FZF+zdLeicwv6TVgb2Bms7ohZJaNwVbEO/Pn1a0D8zYEV+Ttt1v2ydXMn+BpCOB04hzRcv+dZXsQ9wcz8/IzfEy1L057rlTDdwMPIf4DPSELIAoSHoRkSe0PHHx+hNwhu1bK9m/1nbXEMx4j83hHGcC04ADbU8pb8rrbb+kqe2OeY4AHiDCJh8iTkK32m6Ud1acB1OcB0Z2JCCciec3tH8D8Oo2uxe0H1dwFtvnWgt4bTk83/bNtWwX+8cRO10/Z/RJ9EsVbD+v27jtu5raLvaPAD4L/IfIR5oCTLVdLd+pzPNs4GtEeMbA+WWeeyvO8QxgdUZfKC+qZHsv4CTbD7TNtYPtoyvZ35kIJV5QhjYFDm7t3FWw/zwiDWAhIhS3JHC07d/WsF/m2AJ4HfEZPtv2uRVtLwYcWOwDnA18ptbNX9m934XR66/pjCLpk8X+C4n1vx64xPZbx33i7Nu/oMuwbVdL7ZG0I7A9kR5xAvA24JO2T6lkvyfXs445LgDWAa5i9Pl662pzpDPXHyQ9AnQ7iQl4vu3Gd6ySrra9vqTrbb+0jE23vU5T2x3z9Pwk1Ask/YG4o+sWN2nsLHbMtTaxw2XgUts31rJd7B/Ubdx2lR1ASVOAVhXcxbZvqGG32J5uex1J2wBvIS70F9ieUmuOfiDp/US4ewUiF2ZD4PJaF7Jun932z3ZD2yLW/TjQKgq5snK6Qc+rQXuJpO06HYZuYw3s71N2YccdazjHTYQTcV25wX8u8M2aTkQ/KLuLryHO3efbvq2i7ZmuZ8C3a4ZEJW3abdz2hdXmSGeuP4y129Gixq6HpF8RLSXOLTkGGwKH2+76RmowT09PQr3ekeg1isrYdwKnEyeHNxN/z2E9mGtxwhH9d0Wb+xA5bKeVoW2AY2uF/CTdYnstSd8CTrV9lqQbajlzkj5q+whJR9ElXGJ770rz3EQkYV9RnNMXAofY3r6S/RuBKa2LSgk13Wh7rfGfOdv2q0QExrH/JuALwEK2V5G0DnBoU0dC0kOMEwazXStt5Trb685qrLL9Ks56m72rbG8g6VoiCvFv4CbbL65g+8XA/kQhkIFbgS/UyrXtSLuZiZqRlH5Q0qBaRRtX1YwQQObM9Y1aIapZsB9wBrCqpEuBZYgt6dq8h6gAbue9Xcbmll1tz0j0LgmvuxJVTXONpNcDi9v+ccf4O4G/VwzR7AS8zPYjxf7niET2as5cOZGeSFRgIek+4N22b6lgfhci6fjhYvtw4HKgVv7WTyXdToRZ9yw5MI9Wsg3Qumu/Ztzfas6jth+VhKSFbd8uaY2K9s8m8sKOIS6We1DaZFTiCknr2766os12DqYH1aC2FweQdCjwV+JzIGBHIgewEZK2IlpULa/RrVqWINr1NLW/A3Gzt4pGtzpZHKhd0HS9pKWIvPBriCKaxvlskt5MOOqHAV8kXv+XAadJ+kjJj23KtYyk26wE/LP8vBRwN7BKE+PlZmy8m4IanQFac70dOJL4LAg4StL+ndeiJqQz1yfGuZtsJZc3vpu0fV3Zzl2j2P21S6VaDfp4EppP0oxKxLIjUaN66RBGV1G2+CWxi1bLmbuL0Z+tBaiQeN/BscB+ti8AkPRqIvl7o/GeNJuI0W0GnqR7aHqusP3x4iA+aPvJkoLw5or2W0nkj3QLk9WaB/hTuVD+BDhX0j+JZuO1+BhR2foB4vU/B/h2RfubAbtLuoto21Kl0KWNXleDvt52e9/Ab0i6Ejiiod17CMdna0a3anmISAloymVEIvyzCEeo3X7VdAzbrdYk/yvpbKIQr0ZxwqHAFh5d1HeDpF8SfSobO3O2V4EZVdZn2P5FOd6KkXzkJryxfN+rfG/1iNwReKSC/XYOJIrt7oUZRRznUbG/ZoZZO5C0m+1jxzqeyCj6U+3JSK7WxcAxtqvsepRQ8SrE3djH2x56iAj/NL5rLfMcSagPtO9I/NH2h8d73mzYvXGsC9V4j83FPKcR2+lnE+t/HXAJkQyO7f0qzDFTWLJWqFLSfsTu6+ll6C3Ad2x/pantYn8xYhd5Jdu7KSoF17D9sxr22+bpaZisw+6mRIL/WbYfq22/F4yV+lEriqAo0jmfOFdsS1SDLmh7j0r2LyMqx39AfM52APayXeOGBkkL1rwZHhTqgRqNpFttrzmnj83lXDOlA0i6xvZ6lexfanvjWY01nOMmtxUiljy9G1yxODF35mam8zayZ7eVPeC7hGPVCoftQNxtVNmNKCf5u4BX1LA3Dr3akVhE0gKdTqekBYFFK9hv8fPy1eKKirZb3CnpU4zcTe4EVJExsv2lkn+5CfH672z7+hq2C9OIHY/WRfdPRGPZKs5cr8NkHXN1ykktT8P/g6Qf2X77WGGgWjcdLadNUfXbi6bNHyJ2JP5LNOg+m6hirsU7idSOVnrHJWWsFitLOoyZ23pUKZQqOc1HEU3pFwLmBx6ulfNX5vg60Sv1VcDnGFGjadpw93FJK9m+u2O+51H5Mwbcp6jK/R7xediJupGgp0naxPYlAJI2om4LHYCzys5oqzn99oQgQTVyZ24S0cvdmg6b/TgJLUSEi02lcLFCEmxZ4INt+WBPI9pX3Gf7Y03nKDa3JCp8e/bhUhSFHMKIw3UR0Vbinw1sLmH7wbESj2slHLfuqjW66rpmAcQUooLvUODTbQ89RFTNzvVr1DHPQUQz0DVsv0DScsApTe/oJT3X9l/6sHO2NRHmWw64F3gecFuNAouSGvF52/s3tTUoJF1CqKx8mUjP2Jm4ZnatJJ8L+9fQRSHDdVtiXOfScLfmZ03SW4hw9v8wktu2PrEL+zHb1WTJyvnoIMIhhTjXHVLxfPQyIqdwyTL0APC+SuHo9nm2JVRXBFxk+/RZPGWOyJ05ZoSVxsQVenf1ieslbWj7CgCFDuWlPZinpzI9Jf/rBOAPxBt/RUnvcfP+XZ8kdgbuKnlCIppDH0fIfNXivcDXJf0ImGb7joq2gSgKIcJWNTmZyCNpnZxbtPr+1Wrd8piiZUUrJ3JV2novNcXRRuUGSSd7RN3gGcCKtRy5Qk/kpIojNz9wnO0auUFj8Rmincp5tl8qaTNiN78xJReyZ5WyAJJWIG4qNybeS5cA+9jupuYzNyxq+/ySv3sXcLCkiwnHogrurUIG9EiNxvZPFH1BP0zswIpojPt2V2xjVOa6n2gB1BNKyHmKonm/bPekGbvtU+muS12FdOaC1gl4DeLuopXc/ybiLmBC0xaOWRB4t6TW1vdKRLl4dXp8Evoi8DqX3lSSXkBsTze6OJTw6scVqgwt5/O3tv/TxG6Xed5REuN3BL4v6T9EaPGHrR3BuUXSV2xPlfRTuofg5rrtg+03lu+NqsRmg4OIqswVJZ1EXIzf24N5zi27TwsQfeD+LunCGjmLhZ7JSbUKQyQt2auLC/C47X9Imk/SfLYvUBSm1OJ6RaHUKUR4DwDbp439lDliGnED0koj2amMbVHJ/qPFEbpD0gcJeceaMk+9VsiAHqrRFKetqlh8NxQNd7ud62r1c/x0x3HL/qFdnzBnti+xvYlmLoCsVvjYIp05RhqtSjoHWNf2Q+X4YOJENNF546x/pSq9Pgkt6LYmo7Z/U/LaGqHR4u4QH66lFM1ZH2pqf5Rh+wFJJxMf2v2JHY9PSPqSm/XLa+XIfaHpGsdC0vm2XzOrsbm0LeB2Qn93Q+L12cf2fU1td2HJEjZ+P7FDepCid1stei0n9Shwk6RzGe0M1dqRfUDS04lCqZMk3UvdfKelidym9ouuGelf2JRlbE9rO/6OpKmVbEPo4y5G7IB/hvg73lPR/ruA+YAPElWyKxKFIo2R9AtgT9vfVfSYey3xWdvOldVo+sBH2n5ehHiNar5P22+wFyGup1WaEtvepHxvvGM/KzJnrg1F76spLnItkhYmKk5eONiVzT6tcBJtjnoPYv89lemRdDxx0m8vFV/A9s4N7U7rMrw0sDawi+1fNrHfNs9WxIX9RcBJRCXoX8rOza22x20gPZtzVG/crKiGXowOmTOicOBM2y+aW9sd8/S0WW3bPDcRlcQnEBJ3V6ti1XKZo5dyUt0cB9v+bkO7U4n0i9uIFgzzEZ+xJYnm1rV7nbXPXa2vnaTzgO8wklS+A1Gs0/imY9hR9DX7LPHeP6JGzvFEouywV22G32Z7YaIVyusr2jzR9rtmNdaE3JkbzYnAVZJOJ5yJbYgK0aFA0meIcNXvGNnSNaPvjBtj+y5F5V41+agOPkD0/tmbkeT+xuoPYzmDxTn9ESOyRk15F/CNTufQ9sNlB6cGvWjcvDuxG7EcoxuLPkiEa2rR62a1LQ4lKigvLY7c84Gq+YvFeTtX0rOo3/B1qW4OewW7KxDvkxcSfc0uI5y7n9ZKKm9H0ppEju0OwL+IPNsavI/I3/0ycZ67rIxVoaR37E8UhrTfHNcK721MNFbutN84N9X2jyT9nCgAukbSibTlyg1RHnirAKLFfES6zXN6OOVi1MsPbjGqqEihm171hjZ35jqQtC4jmpQXuW5Lhp4i6dfAS9yjPlclRHYQERYQ8cF6AjiqRn7BIFGF/mOSzrH9uln/ZqM5Wo2bNyHCYy0WB56skTAv6UOuJN01hv1bgRcQbW560ay2pyiquT8P3E+E304kGsDOR6hwVFFp6PaeVEW5p5IqsR7RIuYV5esBV+gRVm6QdihfTxAOy3oe3WR2QiPpBqKNx7W0NdF2wx5tbfZvJyIbnfar3BSU/+/HifPFDxntzDW6CdcYUnlt9qsVZ5VCi5YSxBNE659DXVqJVLDf3gJofkI56TM1zoGSDgA+QbS+ajUiFvAYIZF4QNM5WuTO3MwsRnSmnyZpGUmr2K7Sv6sP3ExInVTVfGtjKpGsvn7rNSm7Hd+QtK/tLzcxrj7Kq3TMuwZ1qimXqWBjVvSje/y3FRXePWk+DWxVyc64lJ2VbwDL2n6xpLWBrW037XX2deIEvSShHrKV7SsU2qzfp6HklsZWWlmCurt/ixabS5ave4DGupqlGGpJopnv22zfIen3tRy5kqd7p+1jOsb3BZ7jSi2GCAWLb1Sy1Y1/2T6zF4YV7ZG+RBTzresiLViRllTexkQfvh+W4+0YrZpRgxd1nntKKLQW7TnnTwB/c6UG+A497sMkHVbTcetG7sy1oR71jeoXktYjZFRups05aVLh2GH/ekLC5b6O8WWAc5ruGGiMvlot3LC/lrpXgC4NPBfYyfblDe3fyehk3VFUrOJrzfccQvvSwNW2/1rJ7o8I5/B7ZWgH4Bm2GzefLtWBN7qC0PdszHUhESb7pkd6bN3cdO5SMLNO+fm29lzCGjtn6rHSiqRjibDPQ8CVRFPrK1yv/97/ES1bzgBOtn2ZpDtrhA+L/VuBF9t+qmO86nurFMDdSyihtJ9Pa/U3+zyxE3Rah/0a2qkXA3u4jlbzePNcQHQeaLUAWpC4FmxWcY6eKrn0I5+t2HwGsDqjG1BX65aRO3Oj6UnfqD5yAnA4cXfduJdQFxbsVnVo+++qUG3azVlr5SK5zl1HZwWoiZ2OOyqFppck7vK6qYbUrOJD0i5EyPuXZb6jJB1q+/gK5tfw6KaiF5SQU2NsPyXpBnXpHt8DFrN9lUZrg9a4427/bHW2tWn8Pi2fg7skvRb4T3nNXkDkuDXeOSNaFi1M5A/+mVDgeKCCXQBsv1nSkkTV4SGSViMqfjewfVWdKTzT+a28TjUVe1oFKO2Nj2v2W2zl6LbnEFbJcbb9yln/VhWWI1I8Wg7u08tYY8rN6vLAopJeyuiCrMVqzFHofT5bVNTvQ+SrTicq+S+nYj57OnOj6VnfqD5xn+2vzfrX5prxHJ7GztB4uUiSGuci2b6w6RpnwV22qyVgz4KPAi9t5dcomoFeRnQyb0qvm08/F7hF0lWMtAWw7TdXnANCBmhVRhqmvo0IUTdliqQHiYvLouVnynFNWayLgFeWO/rzidDW9kTl6Vxje8vi9KxF5Mt9GHixpPuBy11B4cDRG+944HiFXNj2wFckrWh7xYbmH5G0ujuacSs0fqv1jHSP+y3W3L0aIJ8nzhcXlONNiaKOGryeKOpagQgZt3iISHNoRHs+W8dn+DGgth77PkQP2ytsb1ZSMqoWD2aYtQ1JHyG2QbcgQhzvI8IEPUsGr4mkLxHb9WdQedu+2H+S0T15ZjwELGK70e6cQt6mlYt0LB25SBXCV52NG2c8RIUGjjWT02djrvOJ1+excrwQ8ItKBRC3EQ2025tP30bsSDUuVFCI0s84JHLzdnAFGamOeZ5PvI82Av5JJE7vNCxJ+BqRYvoQoUZwRO33mEJFYWPiNXoj8EzbS9Wy32W+51VIl9iKUH74LCP5WesBBwBTbTfSvJS0ue1faua+lEDzdAlJO9n+nsZQHvIQVZrCjB201i7jlbXSPdrsb+tQT+gJfclnk662vb6k6cDLbf+3PV2jBrkz14btLyj6Rj1IXMw+7Yp9o/pA6yS/YdtYtdYktuevYWccFrB9DkAJGV5R5r29RvTEvW/cWDXHYhb8Gbiy5CcZeDPRVmc/aHxB2LLC+sbE9oWS1iGS/N9OOFnHjP+suZrnTuC1ZYd9PlduDN0HJOkVxE7cLmWs8Tlb0t6E87Yx8Dix63o5sZNWI4w7Jk0duWLjTIU26P6ElBREnvC2tmusf1MifeFN3aanebpEK+IzTCk8XSk7vK8Fnm/7UEkr1Qqnt5xeYOVujm9Tp1fSC23fDpyi6GLRab9mf9Y/KVSBfkK0MvonUXBUjdyZ64JCo62970/13kvJzLQntXYmuNZMeJ0MKIp1xsRz0XpA0hIOxYSluz3e9HNQ8r5a/cb+QVTAfcQVmiiPMd9ShNzQyoz+PNfWtO0JZQfzw0SfvMPLTuPUpusvO/iXFbs1ws7JPIqkbxA79pvbflFJCTjH9voVbO9u+5tjnOvshu2wJB1re7e2EHGn/ar9Wdvm3ZSIPp1VKVc77KYzN4Kk3YlGo/8h3qCt8FvtBoI9QR0acy2avun7RVsYV8zcl6dxGDcZH0k/s/1Gje7r1KLx50DSU0Sbk11c1EJUscqxy3yXEZWaowqCbJ/Qi/mS0UhaehhvhMcKf7aosCM0bl7zsNxswKhUgBnhf0k3dBRQNZ1jY9uXzmpsIlPywW/xiFTo4sCatq+sNUeGWUfzEWAt90Ynsh/0TGOuH/QhjNsXJL2RyF+rXlEs6Su2p6p7m5VGbWiKIydgU/em0nRbYmfuAklnEX3IalYfdrKI7XEvzBORXv6P+8yVJUdoGiEHNyw7B63w5xpE0nqr19+biKKUprTy/PrRo63XPC5pfkaKjJahfieFo4DOqEy3sblijNzIfwE32a7Vs/UbjF7vw13GGpE7c22UC8xbXb/B4kBQDzTmklkj6XtEN/1TCYH3ag61pJfZvrajiGAGrlCxqx5rp5YctrcQ4dbNiZY6p7fyJSvOsy/wb+Bn9KBPWK/ox/+4H7TlU72P6If4Q0Kn+DcDXdhsIukcIg+vfTflFNtVckrVhx5tvUbSjkSl8rrE5/htwKds/6iC7VcQuZ1TCcm2FksA29Ta/VPInr2C0KSG0KW+glCpOdT2iWM8dU7mmKnYQZV1onNnbjQHAJdJupLRJ/+h2fbuoBcac8kssL1TybvcAZimaHUzjajIbZSEXy7y8wO72t6pwnK70VPtVNsPAycBJ5X8vO2I5rhVnTmixcCRwIGM1iqe0J8JF7moUijS0kD++2BXNeeUnbiWdu1mRBPqPRU9Cz/uuWzSrf5JSa3E6JZLjxH5l7XoWY+2fmH7JEnXAq8hdtnfUvHmdSHiNVmA0cUiDxJOYy2eIlQm/gYgaVli1+zlxE5sY2cOuLMUHrUURfYE7qxgdwbpzI3mm0QVU6+a7vYUddeYG4p8uclGKSQ4lcj9m0o0pN5f0tfcsNWN7ScVUnML1UygbWMzYHdJPddOLbtk3yxftdkPWG3Y0ibKjtYoDWRJQ6eBrOh9uBNR5f03ovL0DGAd4BRC5WJuuGbWv1KFE4kK8dOJ8+o2xO5TLXrZo60vaEQp4fYuY40oO9AXSvpOjSrocVi55cgV7gVeYPt+SY9XmmMP4GvAJ8vxecBulWwDGWYdhaTLbG806HXMLRoth1VVYy6ZfSRtDewMrEpcEE6wfa+kxYDbalRvSvomEdo4g7ZcyabJ2cV21/X1+IRaHYWu6TuGLW2ihIffAOzmDg1kogKukQZyv5D0G+L9P832nzoe+5jtwwezstmntKxoqSlcZPv6SnZFNMN9nB72aOs1XboOzE/kmq1ZcY5liCbpazFaCqtKtamko4ld2FPK0LaEKsr+wM+GJeydzlwbkj4H3AX8lCHKsUkmFpJOAI5zF909Sa+xfX6FObq2JvFctCTpYrvnlVf9oOyorEXkwgxN2oR6rIHcLyS9vTN3StJ2tk8Z6zmzafdZwF5EI+jjiVD6K4HfAR9uVUo3nKPnGsK9zk3tJWpTTyC6DrQKmR4DjnXFJrwld/GHRIHiHoTM2t9tf6ySfREO3MbE33EJcGrNgh1Fc+6jyhwuc+zTeZPTaI505kYoLRk6mfCtSTRa2aD1oTIRRl/IdobT+0S5Mz3bFZQYBkVxJtZtnczKhe0aD1mfP0nv6TY+0VuTSLp5LCdivMcmGp27NmONzYXdc4hQ6+JErtY04gb8lcCOtl/dxH7bPCcBB/SoshtJ/0sUhPQkN7UfqD/qCdfafll7wYCkC213LRCaiEg6FziZkfy7nYj36ha15siLfBvusRZfr3CHskHZSdkT2B04fSCLmkcp+WyPSFrSoU/ZE8rJYTvbD5TjZwA/qFS5rPa7UoeA+dCdK2yfoJA5e0EZ+nWrcnCC01MN5F6jkNt6A7C8RvdUW4JI/2jKsrY/UXZU7rJ9ZBm/XdJeFey36KYhXLM1TN9yU3uF7QPKuWd1RodAa7RwadH6zP5F0v8jlBNWqGW8tCY5HHg28T+oIu/YwTK2p7Udf0fS1Ir205nrRNKLid4/7W/M7w5uRbOPouP9VKLr/cnA+i5C7ElfeRS4qThc7ReBmuG9ZVqOXLH9T4WgeQ16XnnVDyS9mkhY/wNxgl5R0nsqX2h6wRSNCH+3I9rOSxOYe4ids60Z3TftIWDfCvafhLjaSuosbqlZuFZVCL0LW/XYfs+R9H5CRH4FYDohJXk5lSQkC5+VtCShhnIUcVNQ0xE6AnhTxSrcbtwnaSfg++W4pYJTjQyztlHykF5NOHO/ID5sl9iuWQZdnZJD8mGi38/xRNVbz3aFkvHpR3ivtAPYphUCKkULp9cIhRan8GvECdnA+YSMVK0Gmn2hvEbvtP3rcvwCoj3MUOYpDRuSFuhFAZakB4iWESJCqy3nXMAmtp9Rca5licbBAFfV/gxImsJIgcXFtm+oab/XlA4K6wNX2F5H0guBQ2xv3+N5p9r+SiVbl9reuIatceZYCfg60c/OhJze3jVD+OnMtVHemFOA621PKR/kb9vuJrg8YZD0MPB3Indkpj5mNSock4mFpC2BY4FWA9lXEdWPZw9uVRMLdWnK2W0sqYukH9l+e0erpBk0ff01RjPlNvtVmipLejtRXPErRhzH/W3/uJL9fYBdgdPK0DZE8UCj1kX9RNLVttdXKH283PZ/1aVBbg/mvdv2SpVsfRV4DvATRhdKnTbmk+rMW80hhXTmRiHpKtsblDv6zQjH6Gbbaw14aeMi6WDGb6LZ63BB0oak1YHDmDlcX7WQpuzIbkhcaC7vrH5sYPcI4LOERvFZxA3OVNvfq2G/X0g6nvhctJKOdwQWsL3z4FY1+ZH0XNt/6VWLG0kr9aoooWOeG4iq4nvL8TLAea6nPHAj8ApHE+2WMsrlw3SzUSrGdybCnpsTFcYL2n5Dj+f9o+0VK9ma1mXYtt9Xw/4481ZzSCFz5jq5puSdfYvI9fg3cNVglzRrbB886DUko5hGNH39MnFTsDPU1SCVtDEw3fbPSi7GJyR9temFsvA62x+VtA3Rb2k7or3HUDlzwAeIFhZ7E6//RcDRA13RPIDtv5TvvepL+BOKpqWkU21v26N55usIq/4DmK+ifVHy/wpPUvk80Wtsb1N+PFjR/HhJ4gaw51NXMzS4m7uq/+t05tqwvWf58RiFTusStm8c5JqSoWRR2+dLUrmgHSzpYsLBq8U3iET5KURzy+OB7xJd5JuyYPn+BiLH7P4oHBwuSsjnROBED6Ec1rDTwyrB9jdjL9tGnSXpbEaS1rcncqlrMQ24suxuCXgzcFxF+z1DIcPXyU3l+9MZkShrMkd7y61RDxH97arQjx5wY1A1LJrOXAeSlgeeR3ltJL1qCKrfkonFo6U32x2SPgj8mbig1eSJUs33ZuBrto8bq/BiLvippNuJMOueJbz0aCXbPae0rGiXw5KkJxkyOaxJQK+qBD3Gz3UnsfcvDukmxPvoWNvVWj3Z/pKkXxX7ADu7ksJEH7iWeO1FqCf8s/y8FHA3cy/VNoPOlls9ZBrR/WG7crxTGWvcA65fDimkMzcKSYcTd1+3MrL9bUaqpZJkdpgKLEaE9z5D5JLUcrRaPKTowv4u4JWKZsULzuI5s4Xtj5fPwoOtvnnErsGwMJW4y17fHXJYkvb1kMhhTQL+1qN2D3IEImsAAAzqSURBVK3WLQIWbWvj0ov+YJcSfc5Mb1Juniy2zRDpgbd6sko6BjjD9i/K8VbAsDVM71kPuD46pFkA0Y6kXwNr2/7vLH95AlKqb/8HWM72VpLWJBJsh2LrPpl9JD0HeCdwte2LS+n7q2v0RFRoyO4HrGR7t1LQsYbtnzW13Q80SeSwhp1BVQnWoo/VrKcW+8NYzTqTJJmka2yvN6g1zSmSzgO+w+gecDvbfs3AFjUXpDPXhqQzia76/x70WuaGsv5pwIGltcoCRJuVlwx4afMUktYDDqQtXA/NWzJ0med5wOq2zysO2PwueqoN7f6QCKO82/aLJS1KVNn1tN1ALTRJ5LCGnUFVCdYiq1lnTckpvJgojjIRonyV6yjR9IV+9IDrBxlmHc0jwHRJ5zNEwtxtPMv2j0r4DdtPlFyhpL+cRBQl3ESPQieSdgV2A5YGVgWWB44htCqbsqrt7SXtAGD7PxquCoihlsOaLEyCFjBZzTprdiDyU1u5hBeVsaGhOG2jJNpKmLVaD7h+kM7caM4oX8PKw5KeSUm4lLQhkEoQ/efvtnv9PtoL2AC4EsD2Haon5/VY2Y1rvY9Wpe3mZggYdjmsSUHZmevWNHgoduboXs16ZkX77dWsAG9hSKpZW9i+H9hHIbf1VI3IwARhP9KZG15cUW5pQOxHOKOrSroUWAaY0FJkk5SDJH2bkMHqVa7Qf20/1towKyH1WjkTBxG9olaUdBJRTPDeSrZ7ju35B72GBID2HMtFiJywewa0ljmmz9WsYriqWQGQtD7RFmnxcvwv4H22rx33iROfYdshzZy5dkoj1oMZyXVqVUf1spdRVcpFfQ1i7b+2/fiAlzTPIel7wAuBWxgJs1bNFSoqDQ8A7wY+BOwJ3Gr7wEr2n8mIusQVtdQlknmX0q7nPNs1RdirI2k1YFnbl3aMvwr4s+3fNbS/PpESc2bH+NbF/tA4QiXvby/bF5fjTYCjhynvrxu11Rn6Qe7MjeY4YF8i+Xvocs0kbQecZfsWSZ8E1pX0WdvXDXpt8xhT+lB08nFgFyIvb3eimem3mxotNwNbEc4owG2E05gkTVmd6Ek20fkK8Iku44+Ux5pqdR9J953uWwm95Qnt7HbwUMuRA7B9SemtNuHpZw+4fpA7c21IutL2ywe9jrlFRUS83B0dBnwB+MQw/03DiKRvAV+2fWuP51kGoJa6gaTlCNmuvwDXEye1lxLtJTazPTQhsmTwtF0sVb7/FTjA9qkDXdgsmEU19E1Nb9TGsyHphlrVsv1A0peJnprfJ/7H2xMNhE8FyI2E/pHOXBuSPg/MD5zG6FynoXhDSrre9kslHQbcZPvk1tig1zYvIek2osL098T7qBWubxx66KZuQOwiN1Y3kPQdQu/1Kx3jewMvs1278XGSTDgk/db2anP62ESx308Ueqxj4YkeUp9MpDPXxhhvzKF5Q0r6GSEd9VrgZYQc01XDdKc3GSj932bCFYTHJe1LaKbu1qluQITY51rdQNLttl84xmO/tr3G3NpO5k0krQ2szOh+ixO6abCk7wO/tP2tjvFdgNfZ3r6h/WOINiefdNsFWNIhwHNt79bEfjJvks7cJKI0jt2S2JW7Q9JzgZfYPmfAS5snkLSE7QfVXYS6VcbfdI6eqRuMt4ubO7zJnCLpeGBtelgI1AuKks7pRE/CVjHCesBCwDa2/9rQ/tOI/NYNgOlleApwDfD+YWhaL2m/jiED9wGXtG4yk/6SzhyT741Z+o3N6Kc1bJ2shxVJP7P9Rkm/ZyRXqEWVquheqhtIuhP4SLeHgCNsrzq3tpN5D0m32l5z0OuYWyRtBrQ+T7fY/mVl+88H1mqzf2dN+71E0kFdhpcGXg8cbPsHfV7SPE86c0yeN2Ypbf8isBxwL1E5drvttcZ9YlKNktO2Yq8caEnX2V53Th+bTdvd5JdmMAk6+id9RNJxwBd7XQiUTBxKVOK8JuehZO5IZ24chu2NWbQENyfW/NJyZ7lD5mD0F3URn65o+0ng4W4PAYvYXrAX8ybJnFL6sv2UqGKtWgiUTFwyJWMwZJ+5cbB9/5BpUj5u+x+S5pM0n+0LJB0+6EXNg1whaX3bV9c2nOoGyRBxPPAueqhRPIxI+gWwp+0/DHottZG0OdGaJOkz6cyNwxC+MR+Q9HRC7PgkSfcCTwx4TfMimwF7SPoDsYuWOxLJvMjd7r1GcXUkrWj7j2M89sr2JrlzyXeAcySdQOSiDp1Kj6SbmLnh7tKEXNu7+7+iJMOszPqNafv2/q9qzilVUv8B5gN2BJYETrL9j4EubB6jl61JkmRYkHQ0sBQRau2VRnF1SiHQMcCXbD9RxpYl8pHXsL1+hTmeBnya6D5wIm07l7a/1NR+r+lyjjPwD9vdUkCSPpA7c8EbO46H6o3ZRUvwKeCEkrOyFNHTKOkxkhYB9gBWI0JLx7UuBsOGpI2YuT/Ydwe2oGQYWZRw4l7XNmaiKftE5mXA54HrJe0DvATYDziCertOjxO79gsTIvVDFYbOG9OJR+7MTQJKs+BP2L6xY3w94CDbTbUEk9lA0g+Jk/TFhL7pXbb3Geyq5hxJJxIKFtMZ0Si27b0Ht6ok6S/FkfsyEaHZ0PafKtndEvgScAZwqO1HathN5m3SmZsE9FpLMJk92l/rIlh/1bBUQrdT5MjWdJ4ckgZIWgX4EDPv8G49qDXNDpKWAg4HXg58lFBceQ2wT41ec5IuBvawfUtTW0nSIsOsk4NFxnls0b6tIpmRyGz7ieEqhB7FzcBzgL8MeiHJUPMT4DgiZ26YwojXAUcDe5U0iXMkrQMcLeku2zs0MW77lTUWmSTtpDM3Obha0q5jaAleO8ZzkvpMkfRg+VnAouW4Vc26xOCWNkc8C7hV0lWMTlyf0DsqyYTjUdtfG/Qi5oJXdYZUbU8HNpK064DWlCTjkmHWSUCvtQSTeQtJm3Ybt31hv9eSDC+S3gmsDpzD6JuC6wa2qCSZpKQzN4notZZgMu9QbhBaLRiusn3vINeTDB+SDiOaBv+OkTCrbW8+uFUlyeQknbkkSUYh6e3AkcCviBDxK4H9bf94kOtKhgtJtwNr235s0GtJkslO5swlSdLJgcD6rd04ScsA5wHpzCVzwg1En8vc1U2SHpPOXJIknczXEVb9B6EqkiRzwrLA7ZKuJgtpkqSnpDOXJEknZ0k6G/h+Od4e+MUA15MMJwcNegFJMq+QOXNJksyEpG2BjYmcuYtsnz7gJSVDSBbSJEl/SGcuSZIkqU4W0iRJ/0hnLkkSACRdYnsTSQ8RgugzHmK4mh4nEwBJNwBbdBbS2J4y2JUlyeQjc+aSJAHA9ibl++KDXksyKchCmiTpE/nBSpJkFJJOnJ2xJJkFZ0k6W9J7Jb0X+Dlw5oDXlCSTkgyzJkkyCknX2V637XgB4Ebbaw5wWckQIumtwCZkIU2S9JR05pIkAUDSAcAngEWBR1rDhObvsbYPGNTakuFB0mrAsrYv7Rh/FfBn278bzMqSZPKSYdYkSQCwfVjJlzvS9hLla3Hbz0xHLpkDvgI81GX8kfJYkiSVyZ25JElmQtIzgNWBRVpjti8a3IqSYUHSzbZfPMZjN9l+Sb/XlCSTnaxmTZJkFJLeD+wDrABMBzYELgc2H+S6kqFhkXEeW7Rvq0iSeYgMsyZJ0sk+RNf+u2xvBrwU+Ptgl5QMEVdL2rVzUNIuwLUDWE+STHpyZy5Jkk4etf2oJCQtbPt2SWsMelHJ0DAVOF3Sjow4b+sBCwHbDGxVSTKJSWcuSZJO/iRpKeAnwLmS/gncM+A1JUOC7b8BG0naDGjlzv3c9i8HuKwkmdRkAUSSJGMiaVNgSeAs248Nej1JkiTJzKQzlyTJKCRtCNxi+6FyvDiwpu0rB7uyJEmSpBvpzCVJMgpJ1wPrupwcJM0HXNOuCpEkSZJMHLKaNUmSTuS2uzzbT5H5tUmSJBOWdOaSJOnkTkl7S1qwfO0D3DnoRSVJkiTdSWcuSZJO9gA2Av5cvl4O7DbQFSVJkiRjkjlzSZIkSZIkQ0zuzCVJMgpJK0g6XdK9kv4m6VRJKwx6XUmSJEl30plLkqSTacAZwHLA8sBPy1iSJEkyAckwa5Iko5A03fY6sxpLkiRJJga5M5ckSSf3SdpJ0vzlayfgH4NeVJIkSdKd3JlLkmQUklYCvg68AjBwGbC37bsHurAkSZKkK+nMJUkySyRNtf2VQa8jSZIkmZl05pIkmSWS7ra90qDXkSRJksxM5swlSTI7aNALSJIkSbqTzlySJLNDbuEnSZJMUFI8O0kSACQ9RHenTcCifV5OkiRJMptkzlySJEmSJMkQk2HWJEmSJEmSISaduSRJkiRJkiEmnbkkSZIkSZIhJp25JEmSJEmSISaduSRJkiRJkiEmnbkkSZIkSZIh5v8D6gaH14g9sAcAAAAASUVORK5CYII=
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Drop some of the columns we know we will not use for this prediction</span>
<span class="n">chicago_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Unnamed: 0&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;ID&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Case Number&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;IUCR&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;X Coordinate&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Y Coordinate&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Updated On&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Year&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;FBI Code&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Beat&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Ward&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Community Area&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Location&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;District&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Latitude&#39;</span><span class="p">,</span>
                 <span class="s1">&#39;Longitude&#39;</span> <span class="p">],</span> <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Block</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04/02/2006 01:00:00 PM</td>
      <td>055XX N MANGO AVE</td>
      <td>OTHER OFFENSE</td>
      <td>HARASSMENT BY TELEPHONE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02/26/2006 01:40:48 PM</td>
      <td>065XX S RHODES AVE</td>
      <td>NARCOTICS</td>
      <td>MANU/DELIVER:CRACK</td>
      <td>SIDEWALK</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01/08/2006 11:16:00 PM</td>
      <td>013XX E 69TH ST</td>
      <td>ASSAULT</td>
      <td>AGGRAVATED: HANDGUN</td>
      <td>OTHER</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04/05/2006 06:45:00 PM</td>
      <td>061XX W NEWPORT AVE</td>
      <td>BATTERY</td>
      <td>SIMPLE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>02/17/2006 09:03:14 PM</td>
      <td>037XX W 60TH ST</td>
      <td>NARCOTICS</td>
      <td>POSS: CANNABIS 30GMS OR LESS</td>
      <td>ALLEY</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>03/30/2006 10:30:00 PM</td>
      <td>014XX W 73RD PL</td>
      <td>ASSAULT</td>
      <td>SIMPLE</td>
      <td>APARTMENT</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>04/05/2006 12:10:00 PM</td>
      <td>050XX N LARAMIE AVE</td>
      <td>BATTERY</td>
      <td>SIMPLE</td>
      <td>SCHOOL, PUBLIC, BUILDING</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>04/05/2006 03:00:00 PM</td>
      <td>067XX S ROCKWELL ST</td>
      <td>THEFT</td>
      <td>$500 AND UNDER</td>
      <td>STREET</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>04/05/2006 09:30:00 PM</td>
      <td>019XX W CHICAGO AVE</td>
      <td>ASSAULT</td>
      <td>SIMPLE</td>
      <td>PARKING LOT/GARAGE(NON.RESID.)</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>04/03/2006 03:00:00 AM</td>
      <td>063XX S EBERHART AVE</td>
      <td>BATTERY</td>
      <td>DOMESTIC BATTERY SIMPLE</td>
      <td>SIDEWALK</td>
      <td>False</td>
      <td>True</td>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># change date time format</span>
<span class="n">chicago_df</span><span class="o">.</span><span class="n">Date</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">Date</span><span class="p">,</span> <span class="nb">format</span> <span class="o">=</span> <span class="s1">&#39;%m/</span><span class="si">%d</span><span class="s1">/%Y %I:%M:%S %p&#39;</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">Date</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[21]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0         2006-04-02 13:00:00
1         2006-02-26 13:40:48
2         2006-01-08 23:16:00
3         2006-04-05 18:45:00
4         2006-02-17 21:03:14
5         2006-03-30 22:30:00
6         2006-04-05 12:10:00
7         2006-04-05 15:00:00
8         2006-04-05 21:30:00
9         2006-04-03 03:00:00
10        2006-04-06 11:15:00
11        2006-04-04 20:15:00
12        2006-04-06 11:30:00
13        2006-02-26 14:47:21
14        2006-04-03 20:09:00
15        2006-02-17 21:26:33
16        2006-04-05 08:00:00
17        2006-03-31 08:20:00
18        2006-04-05 13:30:00
19        2006-03-31 05:00:00
20        2006-03-28 22:00:00
21        2006-02-17 21:49:21
22        2006-04-05 18:18:00
23        2006-04-06 09:45:00
24        2006-03-31 09:13:54
25        2006-04-05 22:30:00
26        2006-04-05 22:10:00
27        2006-03-31 10:00:00
28        2006-02-17 22:07:09
29        2006-04-05 17:00:00
                  ...        
1456684   2016-05-03 22:32:00
1456685   2016-05-03 22:07:00
1456686   2016-05-03 22:31:00
1456687   2016-05-03 22:45:00
1456688   2016-05-03 21:00:00
1456689   2016-05-03 19:13:00
1456690   2016-05-03 19:45:00
1456691   2016-05-03 20:56:00
1456692   2016-05-03 22:10:00
1456693   2016-05-03 22:15:00
1456694   2016-05-03 17:00:00
1456695   2016-05-03 23:58:00
1456696   2016-05-03 15:15:00
1456697   2016-05-03 23:50:00
1456698   2016-05-03 23:38:00
1456699   2016-05-03 20:44:00
1456700   2016-05-03 08:00:00
1456701   2016-05-03 22:10:00
1456702   2016-05-03 23:35:00
1456703   2016-05-03 22:15:00
1456704   2016-05-03 23:30:00
1456705   2016-05-03 23:50:00
1456706   2016-05-03 22:25:00
1456707   2016-05-03 23:00:00
1456708   2016-05-03 23:28:00
1456709   2016-05-03 23:33:00
1456710   2016-05-03 23:30:00
1456711   2016-05-03 00:15:00
1456712   2016-05-03 21:07:00
1456713   2016-05-03 23:38:00
Name: Date, Length: 6017767, dtype: datetime64[ns]</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">index</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DatetimeIndex</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">Date</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[32]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Block</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-04-02 13:00:00</th>
      <td>2006-04-02 13:00:00</td>
      <td>055XX N MANGO AVE</td>
      <td>OTHER OFFENSE</td>
      <td>HARASSMENT BY TELEPHONE</td>
      <td>RESIDENCE</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2006-02-26 13:40:48</th>
      <td>2006-02-26 13:40:48</td>
      <td>065XX S RHODES AVE</td>
      <td>NARCOTICS</td>
      <td>MANU/DELIVER:CRACK</td>
      <td>SIDEWALK</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2006-01-08 23:16:00</th>
      <td>2006-01-08 23:16:00</td>
      <td>013XX E 69TH ST</td>
      <td>ASSAULT</td>
      <td>AGGRAVATED: HANDGUN</td>
      <td>OTHER</td>
      <td>False</td>
      <td>False</td>
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
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_df</span><span class="p">[</span><span class="s1">&#39;Primary Type&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[34]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>THEFT                                1245111
BATTERY                              1079178
CRIMINAL DAMAGE                       702702
NARCOTICS                             674831
BURGLARY                              369056
OTHER OFFENSE                         368169
ASSAULT                               360244
MOTOR VEHICLE THEFT                   271624
ROBBERY                               229467
DECEPTIVE PRACTICE                    225180
CRIMINAL TRESPASS                     171596
PROSTITUTION                           60735
WEAPONS VIOLATION                      60335
PUBLIC PEACE VIOLATION                 48403
OFFENSE INVOLVING CHILDREN             40260
CRIM SEXUAL ASSAULT                    22789
SEX OFFENSE                            20172
GAMBLING                               14755
INTERFERENCE WITH PUBLIC OFFICER       14009
LIQUOR LAW VIOLATION                   12129
ARSON                                   9269
HOMICIDE                                5879
KIDNAPPING                              4734
INTIMIDATION                            3324
STALKING                                2866
OBSCENITY                                422
PUBLIC INDECENCY                         134
OTHER NARCOTIC VIOLATION                 122
NON-CRIMINAL                              96
CONCEALED CARRY LICENSE VIOLATION         90
NON - CRIMINAL                            38
HUMAN TRAFFICKING                         28
RITUALISM                                 16
NON-CRIMINAL (SUBJECT SPECIFIED)           4
Name: Primary Type, dtype: int64</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot top 15 </span>
<span class="n">chicago_df</span><span class="p">[</span><span class="s1">&#39;Primary Type&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:</span><span class="mi">15</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[35]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>THEFT                         1245111
BATTERY                       1079178
CRIMINAL DAMAGE                702702
NARCOTICS                      674831
BURGLARY                       369056
OTHER OFFENSE                  368169
ASSAULT                        360244
MOTOR VEHICLE THEFT            271624
ROBBERY                        229467
DECEPTIVE PRACTICE             225180
CRIMINAL TRESPASS              171596
PROSTITUTION                    60735
WEAPONS VIOLATION               60335
PUBLIC PEACE VIOLATION          48403
OFFENSE INVOLVING CHILDREN      40260
Name: Primary Type, dtype: int64</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">order_data</span> <span class="o">=</span> <span class="n">chicago_df</span><span class="p">[</span><span class="s1">&#39;Primary Type&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:</span><span class="mi">15</span><span class="p">]</span><span class="o">.</span><span class="n">index</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># plot top 15 crime types</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">y</span> <span class="o">=</span> <span class="s1">&#39;Primary Type&#39;</span><span class="p">,</span> <span class="n">data</span> <span class="o">=</span> <span class="n">chicago_df</span><span class="p">,</span> <span class="n">order</span> <span class="o">=</span> <span class="n">order_data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[37]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x27e404d16d8&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot top 15 crime locations</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">y</span> <span class="o">=</span> <span class="s1">&#39;Location Description&#39;</span><span class="p">,</span>
              <span class="n">data</span> <span class="o">=</span> <span class="n">chicago_df</span><span class="p">,</span>
              <span class="n">order</span> <span class="o">=</span> <span class="n">chicago_df</span><span class="p">[</span><span class="s1">&#39;Location Description&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:</span><span class="mi">15</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[38]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x27e40649748&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># How many crimes occured in a year</span>
<span class="n">chicago_df</span><span class="o">.</span><span class="n">resample</span><span class="p">(</span><span class="s1">&#39;Y&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[40]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Date
2005-12-31    455811
2006-12-31    794684
2007-12-31    621848
2008-12-31    852053
2009-12-31    783900
2010-12-31    700691
2011-12-31    352066
2012-12-31    335670
2013-12-31    306703
2014-12-31    274527
2015-12-31    262995
2016-12-31    265462
2017-12-31     11357
Freq: A-DEC, dtype: int64</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot crimes per year</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">resample</span><span class="p">(</span><span class="s1">&#39;Y&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Crime Count Per year&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Years&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of Crimes&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[41]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0, 0.5, &#39;Number of Crimes&#39;)</pre>
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
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot crimes per Month</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">resample</span><span class="p">(</span><span class="s1">&#39;M&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Crime Count Per Month&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Month&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of Crimes&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0, 0.5, &#39;Number of Crimes&#39;)</pre>
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
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Plot crimes per Quarter</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">chicago_df</span><span class="o">.</span><span class="n">resample</span><span class="p">(</span><span class="s1">&#39;Q&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">())</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Crime Count Per Quarter&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Quater&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of Crimes&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[43]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0, 0.5, &#39;Number of Crimes&#39;)</pre>
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
<h3 id="Data-Clean-&amp;-Prep">Data Clean &amp; Prep<a class="anchor-link" href="#Data-Clean-&amp;-Prep">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># reset index</span>
<span class="n">chicago_prophet</span> <span class="o">=</span> <span class="n">chicago_df</span><span class="o">.</span><span class="n">resample</span><span class="p">(</span><span class="s1">&#39;M&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">()</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_prophet</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[46]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-31</td>
      <td>33983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-02-28</td>
      <td>32042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-03-31</td>
      <td>36970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-04-30</td>
      <td>38963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-05-31</td>
      <td>40572</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005-06-30</td>
      <td>40234</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005-07-31</td>
      <td>41976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005-08-31</td>
      <td>41741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005-09-30</td>
      <td>39833</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005-10-31</td>
      <td>40204</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005-11-30</td>
      <td>36244</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005-12-31</td>
      <td>33049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2006-01-31</td>
      <td>37605</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2006-02-28</td>
      <td>34063</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2006-03-31</td>
      <td>43721</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006-04-30</td>
      <td>69128</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2006-05-31</td>
      <td>79013</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2006-06-30</td>
      <td>77348</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006-07-31</td>
      <td>82750</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2006-08-31</td>
      <td>80628</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006-09-30</td>
      <td>75045</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2006-10-31</td>
      <td>76870</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2006-11-30</td>
      <td>70710</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2006-12-31</td>
      <td>67803</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2007-01-31</td>
      <td>67123</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2007-02-28</td>
      <td>53811</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2007-03-31</td>
      <td>71857</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007-04-30</td>
      <td>70389</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2007-05-31</td>
      <td>78170</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2007-06-30</td>
      <td>55802</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2014-08-31</td>
      <td>25802</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2014-09-30</td>
      <td>23811</td>
    </tr>
    <tr>
      <th>117</th>
      <td>2014-10-31</td>
      <td>23911</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2014-11-30</td>
      <td>20680</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2014-12-31</td>
      <td>20891</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2015-01-31</td>
      <td>20656</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2015-02-28</td>
      <td>16287</td>
    </tr>
    <tr>
      <th>122</th>
      <td>2015-03-31</td>
      <td>21560</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2015-04-30</td>
      <td>21610</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2015-05-31</td>
      <td>23570</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2015-06-30</td>
      <td>23059</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2015-07-31</td>
      <td>24101</td>
    </tr>
    <tr>
      <th>127</th>
      <td>2015-08-31</td>
      <td>24685</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2015-09-30</td>
      <td>22996</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2015-10-31</td>
      <td>22979</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2015-11-30</td>
      <td>20486</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2015-12-31</td>
      <td>21006</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2016-01-31</td>
      <td>20375</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2016-02-29</td>
      <td>18590</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2016-03-31</td>
      <td>21878</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2016-04-30</td>
      <td>20962</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2016-05-31</td>
      <td>23332</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2016-06-30</td>
      <td>23791</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2016-07-31</td>
      <td>24646</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-08-31</td>
      <td>24619</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2016-09-30</td>
      <td>23235</td>
    </tr>
    <tr>
      <th>141</th>
      <td>2016-10-31</td>
      <td>23314</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2016-11-30</td>
      <td>21140</td>
    </tr>
    <tr>
      <th>143</th>
      <td>2016-12-31</td>
      <td>19580</td>
    </tr>
    <tr>
      <th>144</th>
      <td>2017-01-31</td>
      <td>11357</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 2 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[48]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_prophet</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Date&#39;</span><span class="p">,</span> <span class="s1">&#39;Crime Count&#39;</span><span class="p">]</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_prophet</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[49]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crime Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-31</td>
      <td>33983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-02-28</td>
      <td>32042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-03-31</td>
      <td>36970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-04-30</td>
      <td>38963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-05-31</td>
      <td>40572</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005-06-30</td>
      <td>40234</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005-07-31</td>
      <td>41976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005-08-31</td>
      <td>41741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005-09-30</td>
      <td>39833</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005-10-31</td>
      <td>40204</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005-11-30</td>
      <td>36244</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005-12-31</td>
      <td>33049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2006-01-31</td>
      <td>37605</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2006-02-28</td>
      <td>34063</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2006-03-31</td>
      <td>43721</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006-04-30</td>
      <td>69128</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2006-05-31</td>
      <td>79013</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2006-06-30</td>
      <td>77348</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006-07-31</td>
      <td>82750</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2006-08-31</td>
      <td>80628</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006-09-30</td>
      <td>75045</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2006-10-31</td>
      <td>76870</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2006-11-30</td>
      <td>70710</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2006-12-31</td>
      <td>67803</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2007-01-31</td>
      <td>67123</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2007-02-28</td>
      <td>53811</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2007-03-31</td>
      <td>71857</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007-04-30</td>
      <td>70389</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2007-05-31</td>
      <td>78170</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2007-06-30</td>
      <td>55802</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2014-08-31</td>
      <td>25802</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2014-09-30</td>
      <td>23811</td>
    </tr>
    <tr>
      <th>117</th>
      <td>2014-10-31</td>
      <td>23911</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2014-11-30</td>
      <td>20680</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2014-12-31</td>
      <td>20891</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2015-01-31</td>
      <td>20656</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2015-02-28</td>
      <td>16287</td>
    </tr>
    <tr>
      <th>122</th>
      <td>2015-03-31</td>
      <td>21560</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2015-04-30</td>
      <td>21610</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2015-05-31</td>
      <td>23570</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2015-06-30</td>
      <td>23059</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2015-07-31</td>
      <td>24101</td>
    </tr>
    <tr>
      <th>127</th>
      <td>2015-08-31</td>
      <td>24685</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2015-09-30</td>
      <td>22996</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2015-10-31</td>
      <td>22979</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2015-11-30</td>
      <td>20486</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2015-12-31</td>
      <td>21006</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2016-01-31</td>
      <td>20375</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2016-02-29</td>
      <td>18590</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2016-03-31</td>
      <td>21878</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2016-04-30</td>
      <td>20962</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2016-05-31</td>
      <td>23332</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2016-06-30</td>
      <td>23791</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2016-07-31</td>
      <td>24646</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-08-31</td>
      <td>24619</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2016-09-30</td>
      <td>23235</td>
    </tr>
    <tr>
      <th>141</th>
      <td>2016-10-31</td>
      <td>23314</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2016-11-30</td>
      <td>21140</td>
    </tr>
    <tr>
      <th>143</th>
      <td>2016-12-31</td>
      <td>19580</td>
    </tr>
    <tr>
      <th>144</th>
      <td>2017-01-31</td>
      <td>11357</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 2 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># rename Date time to DS and prediction to y for final Prophet training</span>

<span class="n">chicago_prophet_df_final</span> <span class="o">=</span> <span class="n">chicago_prophet</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;Date&#39;</span><span class="p">:</span><span class="s1">&#39;ds&#39;</span><span class="p">,</span><span class="s1">&#39;Crime Count&#39;</span><span class="p">:</span><span class="s1">&#39;y&#39;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[55]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">chicago_prophet_df_final</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[55]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-31</td>
      <td>33983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-02-28</td>
      <td>32042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-03-31</td>
      <td>36970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-04-30</td>
      <td>38963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-05-31</td>
      <td>40572</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005-06-30</td>
      <td>40234</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005-07-31</td>
      <td>41976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005-08-31</td>
      <td>41741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005-09-30</td>
      <td>39833</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005-10-31</td>
      <td>40204</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005-11-30</td>
      <td>36244</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005-12-31</td>
      <td>33049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2006-01-31</td>
      <td>37605</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2006-02-28</td>
      <td>34063</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2006-03-31</td>
      <td>43721</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006-04-30</td>
      <td>69128</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2006-05-31</td>
      <td>79013</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2006-06-30</td>
      <td>77348</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006-07-31</td>
      <td>82750</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2006-08-31</td>
      <td>80628</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006-09-30</td>
      <td>75045</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2006-10-31</td>
      <td>76870</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2006-11-30</td>
      <td>70710</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2006-12-31</td>
      <td>67803</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2007-01-31</td>
      <td>67123</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2007-02-28</td>
      <td>53811</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2007-03-31</td>
      <td>71857</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007-04-30</td>
      <td>70389</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2007-05-31</td>
      <td>78170</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2007-06-30</td>
      <td>55802</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2014-08-31</td>
      <td>25802</td>
    </tr>
    <tr>
      <th>116</th>
      <td>2014-09-30</td>
      <td>23811</td>
    </tr>
    <tr>
      <th>117</th>
      <td>2014-10-31</td>
      <td>23911</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2014-11-30</td>
      <td>20680</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2014-12-31</td>
      <td>20891</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2015-01-31</td>
      <td>20656</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2015-02-28</td>
      <td>16287</td>
    </tr>
    <tr>
      <th>122</th>
      <td>2015-03-31</td>
      <td>21560</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2015-04-30</td>
      <td>21610</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2015-05-31</td>
      <td>23570</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2015-06-30</td>
      <td>23059</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2015-07-31</td>
      <td>24101</td>
    </tr>
    <tr>
      <th>127</th>
      <td>2015-08-31</td>
      <td>24685</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2015-09-30</td>
      <td>22996</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2015-10-31</td>
      <td>22979</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2015-11-30</td>
      <td>20486</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2015-12-31</td>
      <td>21006</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2016-01-31</td>
      <td>20375</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2016-02-29</td>
      <td>18590</td>
    </tr>
    <tr>
      <th>134</th>
      <td>2016-03-31</td>
      <td>21878</td>
    </tr>
    <tr>
      <th>135</th>
      <td>2016-04-30</td>
      <td>20962</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2016-05-31</td>
      <td>23332</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2016-06-30</td>
      <td>23791</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2016-07-31</td>
      <td>24646</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2016-08-31</td>
      <td>24619</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2016-09-30</td>
      <td>23235</td>
    </tr>
    <tr>
      <th>141</th>
      <td>2016-10-31</td>
      <td>23314</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2016-11-30</td>
      <td>21140</td>
    </tr>
    <tr>
      <th>143</th>
      <td>2016-12-31</td>
      <td>19580</td>
    </tr>
    <tr>
      <th>144</th>
      <td>2017-01-31</td>
      <td>11357</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 2 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">m</span> <span class="o">=</span> <span class="n">Prophet</span><span class="p">()</span>
<span class="n">m</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">chicago_prophet_df_final</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
C:\Users\Sage\Anaconda3\envs\cv_tf_gpu\lib\site-packages\pystan\misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  elif np.issubdtype(np.asarray(v).dtype, float):
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[56]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;fbprophet.forecaster.Prophet at 0x27e40dc95f8&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">future</span> <span class="o">=</span><span class="n">m</span><span class="o">.</span><span class="n">make_future_dataframe</span><span class="p">(</span><span class="n">periods</span> <span class="o">=</span> <span class="mi">720</span><span class="p">)</span>
<span class="n">forecast</span> <span class="o">=</span> <span class="n">m</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">future</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">forecast</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[66]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-31</td>
      <td>60454.773642</td>
      <td>39871.037769</td>
      <td>73168.474672</td>
      <td>60454.773642</td>
      <td>60454.773642</td>
      <td>-4762.404217</td>
      <td>-4762.404217</td>
      <td>-4762.404217</td>
      <td>-4762.404217</td>
      <td>-4762.404217</td>
      <td>-4762.404217</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55692.369426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-02-28</td>
      <td>60322.370911</td>
      <td>33432.974784</td>
      <td>67348.309861</td>
      <td>60322.370911</td>
      <td>60322.370911</td>
      <td>-9500.516358</td>
      <td>-9500.516358</td>
      <td>-9500.516358</td>
      <td>-9500.516358</td>
      <td>-9500.516358</td>
      <td>-9500.516358</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50821.854553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-03-31</td>
      <td>60175.782173</td>
      <td>42925.248348</td>
      <td>75456.352142</td>
      <td>60175.782173</td>
      <td>60175.782173</td>
      <td>-1224.151952</td>
      <td>-1224.151952</td>
      <td>-1224.151952</td>
      <td>-1224.151952</td>
      <td>-1224.151952</td>
      <td>-1224.151952</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>58951.630221</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-04-30</td>
      <td>60033.922104</td>
      <td>43878.152251</td>
      <td>78470.506727</td>
      <td>60033.922104</td>
      <td>60033.922104</td>
      <td>1182.829000</td>
      <td>1182.829000</td>
      <td>1182.829000</td>
      <td>1182.829000</td>
      <td>1182.829000</td>
      <td>1182.829000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61216.751104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-05-31</td>
      <td>59887.333366</td>
      <td>48818.368558</td>
      <td>82276.165116</td>
      <td>59887.333366</td>
      <td>59887.333366</td>
      <td>5498.247964</td>
      <td>5498.247964</td>
      <td>5498.247964</td>
      <td>5498.247964</td>
      <td>5498.247964</td>
      <td>5498.247964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>65385.581330</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005-06-30</td>
      <td>59745.473296</td>
      <td>47217.517953</td>
      <td>80507.203201</td>
      <td>59745.473296</td>
      <td>59745.473296</td>
      <td>3576.966082</td>
      <td>3576.966082</td>
      <td>3576.966082</td>
      <td>3576.966082</td>
      <td>3576.966082</td>
      <td>3576.966082</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>63322.439378</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005-07-31</td>
      <td>59598.884555</td>
      <td>48384.488293</td>
      <td>80312.801852</td>
      <td>59598.884555</td>
      <td>59598.884555</td>
      <td>4582.849351</td>
      <td>4582.849351</td>
      <td>4582.849351</td>
      <td>4582.849351</td>
      <td>4582.849351</td>
      <td>4582.849351</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64181.733907</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005-08-31</td>
      <td>59452.295814</td>
      <td>47312.131726</td>
      <td>80439.962623</td>
      <td>59452.295814</td>
      <td>59452.295814</td>
      <td>4498.965423</td>
      <td>4498.965423</td>
      <td>4498.965423</td>
      <td>4498.965423</td>
      <td>4498.965423</td>
      <td>4498.965423</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>63951.261237</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005-09-30</td>
      <td>59310.435742</td>
      <td>43437.122285</td>
      <td>78648.909936</td>
      <td>59310.435742</td>
      <td>59310.435742</td>
      <td>1749.360219</td>
      <td>1749.360219</td>
      <td>1749.360219</td>
      <td>1749.360219</td>
      <td>1749.360219</td>
      <td>1749.360219</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61059.795961</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005-10-31</td>
      <td>59163.847001</td>
      <td>44990.963312</td>
      <td>78964.317338</td>
      <td>59163.847001</td>
      <td>59163.847001</td>
      <td>2397.444549</td>
      <td>2397.444549</td>
      <td>2397.444549</td>
      <td>2397.444549</td>
      <td>2397.444549</td>
      <td>2397.444549</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61561.291550</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005-11-30</td>
      <td>59021.986929</td>
      <td>41043.550632</td>
      <td>74137.483958</td>
      <td>59021.986929</td>
      <td>59021.986929</td>
      <td>-2064.694573</td>
      <td>-2064.694573</td>
      <td>-2064.694573</td>
      <td>-2064.694573</td>
      <td>-2064.694573</td>
      <td>-2064.694573</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>56957.292356</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005-12-31</td>
      <td>58875.398188</td>
      <td>36618.109763</td>
      <td>69612.967208</td>
      <td>58875.398188</td>
      <td>58875.398188</td>
      <td>-5991.605511</td>
      <td>-5991.605511</td>
      <td>-5991.605511</td>
      <td>-5991.605511</td>
      <td>-5991.605511</td>
      <td>-5991.605511</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>52883.792677</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2006-01-31</td>
      <td>58728.809447</td>
      <td>37558.565748</td>
      <td>70098.306521</td>
      <td>58728.809447</td>
      <td>58728.809447</td>
      <td>-4772.140541</td>
      <td>-4772.140541</td>
      <td>-4772.140541</td>
      <td>-4772.140541</td>
      <td>-4772.140541</td>
      <td>-4772.140541</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>53956.668907</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2006-02-28</td>
      <td>58596.406714</td>
      <td>32375.683588</td>
      <td>66092.222725</td>
      <td>58596.406714</td>
      <td>58596.406714</td>
      <td>-9502.632319</td>
      <td>-9502.632319</td>
      <td>-9502.632319</td>
      <td>-9502.632319</td>
      <td>-9502.632319</td>
      <td>-9502.632319</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49093.774395</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2006-03-31</td>
      <td>58449.817973</td>
      <td>41885.370746</td>
      <td>74957.489775</td>
      <td>58449.817973</td>
      <td>58449.817973</td>
      <td>-1224.293758</td>
      <td>-1224.293758</td>
      <td>-1224.293758</td>
      <td>-1224.293758</td>
      <td>-1224.293758</td>
      <td>-1224.293758</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57225.524214</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006-04-30</td>
      <td>58307.957895</td>
      <td>44483.727641</td>
      <td>76003.042115</td>
      <td>58307.957895</td>
      <td>58307.957895</td>
      <td>1186.957830</td>
      <td>1186.957830</td>
      <td>1186.957830</td>
      <td>1186.957830</td>
      <td>1186.957830</td>
      <td>1186.957830</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59494.915725</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2006-05-31</td>
      <td>58161.369149</td>
      <td>47149.128630</td>
      <td>80992.830563</td>
      <td>58161.369149</td>
      <td>58161.369149</td>
      <td>5451.047069</td>
      <td>5451.047069</td>
      <td>5451.047069</td>
      <td>5451.047069</td>
      <td>5451.047069</td>
      <td>5451.047069</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>63612.416218</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2006-06-30</td>
      <td>58019.509071</td>
      <td>46008.628080</td>
      <td>77488.141247</td>
      <td>58019.509071</td>
      <td>58019.509071</td>
      <td>3563.602666</td>
      <td>3563.602666</td>
      <td>3563.602666</td>
      <td>3563.602666</td>
      <td>3563.602666</td>
      <td>3563.602666</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61583.111737</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006-07-31</td>
      <td>57872.920325</td>
      <td>46608.693732</td>
      <td>79222.367366</td>
      <td>57872.920325</td>
      <td>57872.920325</td>
      <td>4562.735058</td>
      <td>4562.735058</td>
      <td>4562.735058</td>
      <td>4562.735058</td>
      <td>4562.735058</td>
      <td>4562.735058</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>62435.655383</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2006-08-31</td>
      <td>57726.331578</td>
      <td>45019.959704</td>
      <td>78589.791249</td>
      <td>57726.331578</td>
      <td>57726.331578</td>
      <td>4479.578436</td>
      <td>4479.578436</td>
      <td>4479.578436</td>
      <td>4479.578436</td>
      <td>4479.578436</td>
      <td>4479.578436</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>62205.910014</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006-09-30</td>
      <td>57584.471501</td>
      <td>43179.019023</td>
      <td>76450.527257</td>
      <td>57584.471501</td>
      <td>57584.471501</td>
      <td>1829.654501</td>
      <td>1829.654501</td>
      <td>1829.654501</td>
      <td>1829.654501</td>
      <td>1829.654501</td>
      <td>1829.654501</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59414.126002</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2006-10-31</td>
      <td>57437.882755</td>
      <td>44081.666981</td>
      <td>76434.581556</td>
      <td>57437.882755</td>
      <td>57437.882755</td>
      <td>2439.928848</td>
      <td>2439.928848</td>
      <td>2439.928848</td>
      <td>2439.928848</td>
      <td>2439.928848</td>
      <td>2439.928848</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59877.811603</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2006-11-30</td>
      <td>57296.022677</td>
      <td>38458.984639</td>
      <td>71738.537293</td>
      <td>57296.022677</td>
      <td>57296.022677</td>
      <td>-2045.027660</td>
      <td>-2045.027660</td>
      <td>-2045.027660</td>
      <td>-2045.027660</td>
      <td>-2045.027660</td>
      <td>-2045.027660</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55250.995017</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2006-12-31</td>
      <td>57149.433931</td>
      <td>34875.671158</td>
      <td>69101.588819</td>
      <td>57149.433931</td>
      <td>57149.433931</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51136.523970</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2007-01-31</td>
      <td>56994.736733</td>
      <td>36179.642210</td>
      <td>69663.507586</td>
      <td>56994.736733</td>
      <td>56994.736733</td>
      <td>-4782.491825</td>
      <td>-4782.491825</td>
      <td>-4782.491825</td>
      <td>-4782.491825</td>
      <td>-4782.491825</td>
      <td>-4782.491825</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>52212.244908</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2007-02-28</td>
      <td>56855.010232</td>
      <td>31346.784742</td>
      <td>64065.254522</td>
      <td>56855.010232</td>
      <td>56855.010232</td>
      <td>-9501.516526</td>
      <td>-9501.516526</td>
      <td>-9501.516526</td>
      <td>-9501.516526</td>
      <td>-9501.516526</td>
      <td>-9501.516526</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>47353.493707</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2007-03-31</td>
      <td>56700.313035</td>
      <td>39749.027009</td>
      <td>71869.017210</td>
      <td>56700.313035</td>
      <td>56700.313035</td>
      <td>-1225.130705</td>
      <td>-1225.130705</td>
      <td>-1225.130705</td>
      <td>-1225.130705</td>
      <td>-1225.130705</td>
      <td>-1225.130705</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55475.182330</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007-04-30</td>
      <td>56550.606070</td>
      <td>39542.377373</td>
      <td>73068.382613</td>
      <td>56550.606070</td>
      <td>56550.606070</td>
      <td>1190.085128</td>
      <td>1190.085128</td>
      <td>1190.085128</td>
      <td>1190.085128</td>
      <td>1190.085128</td>
      <td>1190.085128</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57740.691197</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2007-05-31</td>
      <td>56395.908872</td>
      <td>44368.641412</td>
      <td>78465.519686</td>
      <td>56395.908872</td>
      <td>56395.908872</td>
      <td>5401.847116</td>
      <td>5401.847116</td>
      <td>5401.847116</td>
      <td>5401.847116</td>
      <td>5401.847116</td>
      <td>5401.847116</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>61797.755988</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2007-06-30</td>
      <td>56230.874395</td>
      <td>44173.760558</td>
      <td>77348.685514</td>
      <td>56230.874395</td>
      <td>56230.874395</td>
      <td>3550.921888</td>
      <td>3550.921888</td>
      <td>3550.921888</td>
      <td>3550.921888</td>
      <td>3550.921888</td>
      <td>3550.921888</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59781.796283</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>835</th>
      <td>2018-12-23</td>
      <td>5792.144402</td>
      <td>-17303.863375</td>
      <td>16264.162698</td>
      <td>5388.264834</td>
      <td>6187.726208</td>
      <td>-6250.192153</td>
      <td>-6250.192153</td>
      <td>-6250.192153</td>
      <td>-6250.192153</td>
      <td>-6250.192153</td>
      <td>-6250.192153</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-458.047751</td>
    </tr>
    <tr>
      <th>836</th>
      <td>2018-12-24</td>
      <td>5779.077729</td>
      <td>-17372.197730</td>
      <td>15602.303531</td>
      <td>5373.836280</td>
      <td>6175.842870</td>
      <td>-6290.109081</td>
      <td>-6290.109081</td>
      <td>-6290.109081</td>
      <td>-6290.109081</td>
      <td>-6290.109081</td>
      <td>-6290.109081</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-511.031352</td>
    </tr>
    <tr>
      <th>837</th>
      <td>2018-12-25</td>
      <td>5766.011055</td>
      <td>-17093.126515</td>
      <td>16387.761538</td>
      <td>5359.407727</td>
      <td>6163.959533</td>
      <td>-6305.807634</td>
      <td>-6305.807634</td>
      <td>-6305.807634</td>
      <td>-6305.807634</td>
      <td>-6305.807634</td>
      <td>-6305.807634</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-539.796579</td>
    </tr>
    <tr>
      <th>838</th>
      <td>2018-12-26</td>
      <td>5752.944381</td>
      <td>-16237.502702</td>
      <td>15730.051883</td>
      <td>5344.979174</td>
      <td>6152.076195</td>
      <td>-6298.957364</td>
      <td>-6298.957364</td>
      <td>-6298.957364</td>
      <td>-6298.957364</td>
      <td>-6298.957364</td>
      <td>-6298.957364</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-546.012982</td>
    </tr>
    <tr>
      <th>839</th>
      <td>2018-12-27</td>
      <td>5739.877708</td>
      <td>-17445.613748</td>
      <td>16254.224774</td>
      <td>5330.550621</td>
      <td>6140.192857</td>
      <td>-6271.692465</td>
      <td>-6271.692465</td>
      <td>-6271.692465</td>
      <td>-6271.692465</td>
      <td>-6271.692465</td>
      <td>-6271.692465</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-531.814757</td>
    </tr>
    <tr>
      <th>840</th>
      <td>2018-12-28</td>
      <td>5726.811034</td>
      <td>-17558.238469</td>
      <td>15816.503972</td>
      <td>5316.122067</td>
      <td>6128.309519</td>
      <td>-6226.535740</td>
      <td>-6226.535740</td>
      <td>-6226.535740</td>
      <td>-6226.535740</td>
      <td>-6226.535740</td>
      <td>-6226.535740</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-499.724706</td>
    </tr>
    <tr>
      <th>841</th>
      <td>2018-12-29</td>
      <td>5713.744361</td>
      <td>-17312.586246</td>
      <td>16316.338094</td>
      <td>5301.693514</td>
      <td>6116.426182</td>
      <td>-6166.312273</td>
      <td>-6166.312273</td>
      <td>-6166.312273</td>
      <td>-6166.312273</td>
      <td>-6166.312273</td>
      <td>-6166.312273</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-452.567912</td>
    </tr>
    <tr>
      <th>842</th>
      <td>2018-12-30</td>
      <td>5700.677687</td>
      <td>-17385.668014</td>
      <td>17418.411359</td>
      <td>5287.264961</td>
      <td>6104.542844</td>
      <td>-6094.055674</td>
      <td>-6094.055674</td>
      <td>-6094.055674</td>
      <td>-6094.055674</td>
      <td>-6094.055674</td>
      <td>-6094.055674</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-393.377987</td>
    </tr>
    <tr>
      <th>843</th>
      <td>2018-12-31</td>
      <td>5687.611014</td>
      <td>-16358.171656</td>
      <td>16214.789681</td>
      <td>5272.844633</td>
      <td>6092.597135</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>-6012.909961</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-325.298947</td>
    </tr>
    <tr>
      <th>844</th>
      <td>2019-01-01</td>
      <td>5674.544340</td>
      <td>-17063.004192</td>
      <td>15968.201193</td>
      <td>5258.431485</td>
      <td>6080.725101</td>
      <td>-5926.030254</td>
      <td>-5926.030254</td>
      <td>-5926.030254</td>
      <td>-5926.030254</td>
      <td>-5926.030254</td>
      <td>-5926.030254</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-251.485914</td>
    </tr>
    <tr>
      <th>845</th>
      <td>2019-01-02</td>
      <td>5661.477666</td>
      <td>-15975.646019</td>
      <td>16621.566665</td>
      <td>5244.018337</td>
      <td>6068.879860</td>
      <td>-5836.485485</td>
      <td>-5836.485485</td>
      <td>-5836.485485</td>
      <td>-5836.485485</td>
      <td>-5836.485485</td>
      <td>-5836.485485</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-175.007818</td>
    </tr>
    <tr>
      <th>846</th>
      <td>2019-01-03</td>
      <td>5648.410993</td>
      <td>-16432.857422</td>
      <td>17578.453280</td>
      <td>5229.725140</td>
      <td>6057.241455</td>
      <td>-5747.166204</td>
      <td>-5747.166204</td>
      <td>-5747.166204</td>
      <td>-5747.166204</td>
      <td>-5747.166204</td>
      <td>-5747.166204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-98.755212</td>
    </tr>
    <tr>
      <th>847</th>
      <td>2019-01-04</td>
      <td>5635.344319</td>
      <td>-17299.232405</td>
      <td>16661.965163</td>
      <td>5215.605329</td>
      <td>6045.125933</td>
      <td>-5660.700423</td>
      <td>-5660.700423</td>
      <td>-5660.700423</td>
      <td>-5660.700423</td>
      <td>-5660.700423</td>
      <td>-5660.700423</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-25.356104</td>
    </tr>
    <tr>
      <th>848</th>
      <td>2019-01-05</td>
      <td>5622.277646</td>
      <td>-16398.239359</td>
      <td>15549.706351</td>
      <td>5201.674499</td>
      <td>6033.063234</td>
      <td>-5579.380115</td>
      <td>-5579.380115</td>
      <td>-5579.380115</td>
      <td>-5579.380115</td>
      <td>-5579.380115</td>
      <td>-5579.380115</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42.897530</td>
    </tr>
    <tr>
      <th>849</th>
      <td>2019-01-06</td>
      <td>5609.210972</td>
      <td>-17289.434998</td>
      <td>16441.363976</td>
      <td>5187.740855</td>
      <td>6020.894531</td>
      <td>-5505.100675</td>
      <td>-5505.100675</td>
      <td>-5505.100675</td>
      <td>-5505.100675</td>
      <td>-5505.100675</td>
      <td>-5505.100675</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>104.110297</td>
    </tr>
    <tr>
      <th>850</th>
      <td>2019-01-07</td>
      <td>5596.144299</td>
      <td>-15884.687110</td>
      <td>16785.291645</td>
      <td>5173.819232</td>
      <td>6008.687435</td>
      <td>-5439.315182</td>
      <td>-5439.315182</td>
      <td>-5439.315182</td>
      <td>-5439.315182</td>
      <td>-5439.315182</td>
      <td>-5439.315182</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>156.829116</td>
    </tr>
    <tr>
      <th>851</th>
      <td>2019-01-08</td>
      <td>5583.077625</td>
      <td>-15539.457809</td>
      <td>17315.432554</td>
      <td>5159.909516</td>
      <td>5996.521446</td>
      <td>-5383.004841</td>
      <td>-5383.004841</td>
      <td>-5383.004841</td>
      <td>-5383.004841</td>
      <td>-5383.004841</td>
      <td>-5383.004841</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.072784</td>
    </tr>
    <tr>
      <th>852</th>
      <td>2019-01-09</td>
      <td>5570.010951</td>
      <td>-16908.654346</td>
      <td>15989.430712</td>
      <td>5145.999800</td>
      <td>5984.386157</td>
      <td>-5336.666448</td>
      <td>-5336.666448</td>
      <td>-5336.666448</td>
      <td>-5336.666448</td>
      <td>-5336.666448</td>
      <td>-5336.666448</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>233.344504</td>
    </tr>
    <tr>
      <th>853</th>
      <td>2019-01-10</td>
      <td>5556.944278</td>
      <td>-17032.803491</td>
      <td>17387.138959</td>
      <td>5132.104015</td>
      <td>5972.216699</td>
      <td>-5300.317162</td>
      <td>-5300.317162</td>
      <td>-5300.317162</td>
      <td>-5300.317162</td>
      <td>-5300.317162</td>
      <td>-5300.317162</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>256.627116</td>
    </tr>
    <tr>
      <th>854</th>
      <td>2019-01-11</td>
      <td>5543.877604</td>
      <td>-15941.897191</td>
      <td>16690.545634</td>
      <td>5118.225536</td>
      <td>5960.047241</td>
      <td>-5273.516318</td>
      <td>-5273.516318</td>
      <td>-5273.516318</td>
      <td>-5273.516318</td>
      <td>-5273.516318</td>
      <td>-5273.516318</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>270.361287</td>
    </tr>
    <tr>
      <th>855</th>
      <td>2019-01-12</td>
      <td>5530.810931</td>
      <td>-16129.246061</td>
      <td>16127.193538</td>
      <td>5104.347058</td>
      <td>5947.899870</td>
      <td>-5255.403446</td>
      <td>-5255.403446</td>
      <td>-5255.403446</td>
      <td>-5255.403446</td>
      <td>-5255.403446</td>
      <td>-5255.403446</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>275.407485</td>
    </tr>
    <tr>
      <th>856</th>
      <td>2019-01-13</td>
      <td>5517.744257</td>
      <td>-16844.840368</td>
      <td>18350.931741</td>
      <td>5090.468579</td>
      <td>5936.465582</td>
      <td>-5244.751151</td>
      <td>-5244.751151</td>
      <td>-5244.751151</td>
      <td>-5244.751151</td>
      <td>-5244.751151</td>
      <td>-5244.751151</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>272.993106</td>
    </tr>
    <tr>
      <th>857</th>
      <td>2019-01-14</td>
      <td>5504.677583</td>
      <td>-17052.724654</td>
      <td>17184.432672</td>
      <td>5076.590101</td>
      <td>5925.031294</td>
      <td>-5240.031007</td>
      <td>-5240.031007</td>
      <td>-5240.031007</td>
      <td>-5240.031007</td>
      <td>-5240.031007</td>
      <td>-5240.031007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>264.646577</td>
    </tr>
    <tr>
      <th>858</th>
      <td>2019-01-15</td>
      <td>5491.610910</td>
      <td>-16728.281653</td>
      <td>17345.038757</td>
      <td>5062.740775</td>
      <td>5913.414172</td>
      <td>-5239.490200</td>
      <td>-5239.490200</td>
      <td>-5239.490200</td>
      <td>-5239.490200</td>
      <td>-5239.490200</td>
      <td>-5239.490200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>252.120710</td>
    </tr>
    <tr>
      <th>859</th>
      <td>2019-01-16</td>
      <td>5478.544236</td>
      <td>-17625.845855</td>
      <td>16597.840195</td>
      <td>5049.175913</td>
      <td>5901.489625</td>
      <td>-5241.236301</td>
      <td>-5241.236301</td>
      <td>-5241.236301</td>
      <td>-5241.236301</td>
      <td>-5241.236301</td>
      <td>-5241.236301</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>237.307935</td>
    </tr>
    <tr>
      <th>860</th>
      <td>2019-01-17</td>
      <td>5465.477563</td>
      <td>-16691.866686</td>
      <td>18407.723800</td>
      <td>5035.931350</td>
      <td>5889.555559</td>
      <td>-5243.327260</td>
      <td>-5243.327260</td>
      <td>-5243.327260</td>
      <td>-5243.327260</td>
      <td>-5243.327260</td>
      <td>-5243.327260</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>222.150303</td>
    </tr>
    <tr>
      <th>861</th>
      <td>2019-01-18</td>
      <td>5452.410889</td>
      <td>-17165.028670</td>
      <td>16859.792348</td>
      <td>5022.686787</td>
      <td>5877.581118</td>
      <td>-5243.863550</td>
      <td>-5243.863550</td>
      <td>-5243.863550</td>
      <td>-5243.863550</td>
      <td>-5243.863550</td>
      <td>-5243.863550</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>208.547339</td>
    </tr>
    <tr>
      <th>862</th>
      <td>2019-01-19</td>
      <td>5439.344216</td>
      <td>-15907.331378</td>
      <td>15961.632388</td>
      <td>5009.442224</td>
      <td>5865.601582</td>
      <td>-5241.079287</td>
      <td>-5241.079287</td>
      <td>-5241.079287</td>
      <td>-5241.079287</td>
      <td>-5241.079287</td>
      <td>-5241.079287</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>198.264929</td>
    </tr>
    <tr>
      <th>863</th>
      <td>2019-01-20</td>
      <td>5426.277542</td>
      <td>-17285.050014</td>
      <td>16129.531154</td>
      <td>4995.277181</td>
      <td>5853.606555</td>
      <td>-5233.429161</td>
      <td>-5233.429161</td>
      <td>-5233.429161</td>
      <td>-5233.429161</td>
      <td>-5233.429161</td>
      <td>-5233.429161</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>192.848381</td>
    </tr>
    <tr>
      <th>864</th>
      <td>2019-01-21</td>
      <td>5413.210868</td>
      <td>-16648.349053</td>
      <td>17594.094160</td>
      <td>4980.871201</td>
      <td>5841.611528</td>
      <td>-5219.668159</td>
      <td>-5219.668159</td>
      <td>-5219.668159</td>
      <td>-5219.668159</td>
      <td>-5219.668159</td>
      <td>-5219.668159</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>193.542710</td>
    </tr>
  </tbody>
</table>
<p>865 rows × 16 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[67]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">figure</span> <span class="o">=</span> <span class="n">m</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">forecast</span><span class="p">,</span> <span class="n">xlabel</span> <span class="o">=</span> <span class="s1">&#39;Date&#39;</span><span class="p">,</span> <span class="n">ylabel</span> <span class="o">=</span> <span class="s1">&#39;Crime Rate&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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CCgB2EpGNtGup9Wf6jdeKOHMRAGWkpgseg3bG5kBMhEREdEiVilv0BLIJqIqsRCwlZ/JVuVMthRAg8bHDJCJiIiIFjMpAEcr2FpFug4/SJewlICtZOhT/WaDATIRERHRIta3NIOYFV39cYUQgFYCjpKwtUAs4oD9SnTUCyAiIiKi+tFSwJISa9uTka5DCgFHCZS0QNrRKLimYbPIDJCJiIiIFjEpBKT0R05HvY6UrZHTCkoK2ACaYlaka7ocBshEREREi5gQfh2yiHiSnRSApSRWtKewdEka5yaLkA2aQmaATERERLSICSHwzmXNUS+jOrDE0RJxSyHe1Lg1yDykR0RERLTIqQbI1GopEXESe8YYIBMRERFR3VVKPRaCSALkBx98ENdddx02bNiAO++8E8PDw9WP7dy5Ez09Pbj22mvxve99r/r43r17ce2116Knpwdf+tKXqo8fPHgQW7Zswdq1a/Gxj30MhUIBAJDP5/Gxj30MPT092LJlCw4dOhTa/RERERHRhVY0x7GyJRH1MmYkkgB569atePnll/HSSy/hHe94B3bu3AkA2L9/P3bv3o1XXnkFe/fuxf333w/XdeG6Lj796U/jmWeewf79+/H4449j//79AICHHnoIn/3sZ3HgwAG0tLTg61//OgDg61//OlpaWvDmm2/is5/9LB566KEobpWIiIiIALQkbLSnnKiXMSORBMi33347tPbPB9500004evQoAGDPnj3YsWMHHMfB6tWr0dPTg3379mHfvn3o6enBmjVrYNs2duzYgT179sAYgx/+8If4yEc+AgC4++678dRTT1WvdffddwMAPvKRj+C5556DMQ06z5CIiIiIGkbkNch///d/jw9+8IMAgGPHjmH58uXVj3V3d+PYsWOXfXxoaAjNzc3VYLvy+MXX0lqjqakJQ0NDYd0WERERES1QdWvzdtttt+HkyZOXPP6nf/qn+NCHPlT9Z601PvGJTwDAtBleIQQ8z5v28cs9/0rXms6uXbuwa9cuAMDJkydx/Pjxy91WIAYHB+t6/asJ9zIY3MfgcC+DwX0MDvcyONzLYCyEfaxbgPzss89e8eOPPfYYnn76aTz33HPVwLW7uxtHjhypPufo0aPo6uoCgGkfb2trw/DwMEqlErTWFzy/cq3u7m6USiWMjIwgm81Ou5b77rsP9913HwBg06ZN1WvU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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[68]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">figure</span> <span class="o">=</span> <span class="n">m</span><span class="o">.</span><span class="n">plot_components</span><span class="p">(</span><span class="n">forecast</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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eGDnuK2MGkARERGRq+Bt9sAW4octxI8kawip8Rebwd2ldXx05BQbDlex8Z39+Jk9+N6YPu4uF1ADKCIiIuIyQb5mgnwD6dMjgGR7KBPjwyk46kderQcTHBHuLq+NGkARERERF/P0MNEjwJseAd6EtdZzfaKFED8vd5fVRg2giIiISDvyNntgs/i7u4xLeLi7ABERERHpWGoARURERLoZNYAiIiIi3YwaQBEREZFuRg2giIiISDdjMgzDcHcRnUl4eDixsbEun7eqqoqIiM5z/5/OSjk5p4ycU0bOKSPnlJFzyujKdGROx44do7q62unr1AB2kJSUFLZt2+buMjo95eScMnJOGTmnjJxTRs4poyvTGXPSKWARERGRbkYNoIiIiEg34/mLX/ziF+4uortITk52dwnXBOXknDJyThk5p4ycU0bOKaMr09ly0jWAIiIiIt2MTgGLiIiIdDNqAEVERES6GTWAV6GoqIjx48eTkJBAYmIizzzzDACnTp0iNTWV+Ph4UlNTqampAcAwDObPn4/D4SApKYnt27e3zeXp6cmQIUMYMmQIaWlpbllPe3BVRuvXr2/LZ8iQIfj6+rJy5Uq3rcuVXPk5euyxxxg4cCADBw7klVdecct62sNXzejAgQOMHj0aHx8ffv/7318y13333UdkZCQDBw7s8HW0J1dldP78eUaMGMHgwYNJTEzkySefdMt62oMrP0exsbEMGjSIIUOGkJKS0uFraU+uyqmgoOCSv7eDg4P54x//6JY1uZorP0vPPPMMAwcOJDExsWPzMeRrKy0tNfLy8gzDMIy6ujojPj7e2Ldvn/Hoo48aTz31lGEYhvHUU08ZP/7xjw3DMIxVq1YZt956q9Ha2mps2rTJGDFiRNtcAQEBHb+ADuDKjD518uRJIywszDh79mzHLaQduSqjd955x7j55puNpqYmo76+3khOTjZOnz7tnkW52FfNqKKiwtiyZYvx05/+1Pjd7353yVwffPCBkZeXZyQmJnbsItqZqzJqbW01zpw5YxiGYTQ2NhojRowwNm3a1MGraR+u/Bz17t3bqKqq6tgFdBBX5vSp5uZmIyoqyjh27FjHLKKduSqjPXv2GImJicbZs2eNpqYmY+LEicbBgwc7ZA06AngVrFYrw4YNAyAoKIiEhARKSkrIyspizpw5AMyZM6ftSFVWVhbp6emYTCZG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
"
>
</div>

</div>

</div>
</div>

</div>