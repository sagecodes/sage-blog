---
layout: project
title:  "Titanic Survival Exploration"
permalink: /titanic-survival-exploration/
date:   2018-09-03
categories: project
tags: machine-learning data-analysis
author: Sage Elliott
published: true
img: img\projects\titanic_analysis.png
demo_url: https://github.com/sagecodes/titanic_survival_exploration/blob/master/titanic_survival_exploration.ipynb
github_url: https://github.com/sagecodes/titanic_survival_exploration
---

Explored the classic Titanic Survival dataset and created a decision tree model from scratch to predict with an accuracy over 80%.


<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Introduction-and-Foundations">Introduction and Foundations<a class="anchor-link" href="#Introduction-and-Foundations"></a></h2><h2 id="Project:-Titanic-Survival-Exploration">Project: Titanic Survival Exploration<a class="anchor-link" href="#Project:-Titanic-Survival-Exploration"></a></h2><p>In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. In this introductory project, we will explore a subset of the RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. To complete this project, you will need to implement several conditional predictions and answer the questions below. Your project submission will be evaluated based on the completion of the code and your responses to the questions.</p>
<blockquote><p><strong>Tip:</strong> Quoted sections like this will provide helpful instructions on how to navigate and use an iPython notebook.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Getting-Started">Getting Started<a class="anchor-link" href="#Getting-Started"></a></h1><p>To begin working with the RMS Titanic passenger data, we'll first need to <code>import</code> the functionality we need, and load our data into a <code>pandas</code> DataFrame.<br>
Run the code cell below to load our data and display the first few entries (passengers) for examination using the <code>.head()</code> function.</p>
<blockquote><p><strong>Tip:</strong> You can run a code cell by clicking on the cell and using the keyboard shortcut <strong>Shift + Enter</strong> or <strong>Shift + Return</strong>. Alternatively, a code cell can be executed using the <strong>Play</strong> button in the hotbar after selecting it. Markdown cells (text cells like this one) can be edited by double-clicking, and saved using these same shortcuts. <a href="http://daringfireball.net/projects/markdown/syntax">Markdown</a> allows you to write easy-to-read plain text that can be converted to HTML.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import libraries necessary for this project</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">IPython.display</span> <span class="k">import</span> <span class="n">display</span> <span class="c1"># Allows the use of display() for DataFrames</span>

<span class="c1"># Import supplementary visualizations code visuals.py</span>
<span class="kn">import</span> <span class="nn">visuals</span> <span class="k">as</span> <span class="nn">vs</span>

<span class="c1"># Pretty display for notebooks</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="c1"># Load the dataset</span>
<span class="n">in_file</span> <span class="o">=</span> <span class="s1">&#39;titanic_data.csv&#39;</span>
<span class="n">full_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">in_file</span><span class="p">)</span>

<span class="c1"># Print the first few entries of the RMS Titanic data</span>
<span class="n">display</span><span class="p">(</span><span class="n">full_data</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
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
<p>From a sample of the RMS Titanic data, we can see the various features present for each passenger on the ship:</p>
<ul>
<li><strong>Survived</strong>: Outcome of survival (0 = No; 1 = Yes)</li>
<li><strong>Pclass</strong>: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)</li>
<li><strong>Name</strong>: Name of passenger</li>
<li><strong>Sex</strong>: Sex of the passenger</li>
<li><strong>Age</strong>: Age of the passenger (Some entries contain <code>NaN</code>)</li>
<li><strong>SibSp</strong>: Number of siblings and spouses of the passenger aboard</li>
<li><strong>Parch</strong>: Number of parents and children of the passenger aboard</li>
<li><strong>Ticket</strong>: Ticket number of the passenger</li>
<li><strong>Fare</strong>: Fare paid by the passenger</li>
<li><strong>Cabin</strong> Cabin number of the passenger (Some entries contain <code>NaN</code>)</li>
<li><strong>Embarked</strong>: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)</li>
</ul>
<p>Since we're interested in the outcome of survival for each passenger or crew member, we can remove the <strong>Survived</strong> feature from this dataset and store it as its own separate variable <code>outcomes</code>. We will use these outcomes as our prediction targets.<br>
Run the code cell below to remove <strong>Survived</strong> as a feature of the dataset and store it in <code>outcomes</code>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Store the &#39;Survived&#39; feature in a new variable and remove it from the dataset</span>
<span class="n">outcomes</span> <span class="o">=</span> <span class="n">full_data</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">full_data</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># Show the new dataset with &#39;Survived&#39; removed</span>
<span class="n">display</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
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
      <th>PassengerId</th>
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
<p>The very same sample of the RMS Titanic data now shows the <strong>Survived</strong> feature removed from the DataFrame. Note that <code>data</code> (the passenger data) and <code>outcomes</code> (the outcomes of survival) are now <em>paired</em>. That means for any passenger <code>data.loc[i]</code>, they have the survival outcome <code>outcomes[i]</code>.</p>
<p>To measure the performance of our predictions, we need a metric to score our predictions against the true outcomes of survival. Since we are interested in how <em>accurate</em> our predictions are, we will calculate the proportion of passengers where our prediction of their survival is correct. Run the code cell below to create our <code>accuracy_score</code> function and test a prediction on the first five passengers.</p>
<p><strong>Think:</strong> <em>Out of the first five passengers, if we predict that all of them survived, what would you expect the accuracy of our predictions to be?</em></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">accuracy_score</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span> <span class="n">pred</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Returns accuracy score for input truth and predictions. &quot;&quot;&quot;</span>
    
    <span class="c1"># Ensure that the number of predictions matches number of outcomes</span>
    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">truth</span><span class="p">)</span> <span class="o">==</span> <span class="nb">len</span><span class="p">(</span><span class="n">pred</span><span class="p">):</span> 
        
        <span class="c1"># Calculate and return the accuracy as a percent</span>
        <span class="k">return</span> <span class="s2">&quot;Predictions have an accuracy of </span><span class="si">{:.2f}</span><span class="s2">%.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">((</span><span class="n">truth</span> <span class="o">==</span> <span class="n">pred</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="o">*</span><span class="mi">100</span><span class="p">)</span>
    
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">&quot;Number of predictions does not match number of outcomes!&quot;</span>
    
<span class="c1"># Test the &#39;accuracy_score&#39; function</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="n">dtype</span> <span class="o">=</span> <span class="nb">int</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">outcomes</span><span class="p">[:</span><span class="mi">5</span><span class="p">],</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Predictions have an accuracy of 60.00%.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<blockquote><p><strong>Tip:</strong> If you save an iPython Notebook, the output from running code blocks will also be saved. However, the state of your workspace will be reset once a new session is started. Make sure that you run all of the code blocks from your previous session to reestablish variables and functions before picking up where you last left off.</p>
</blockquote>
<h1 id="Making-Predictions">Making Predictions<a class="anchor-link" href="#Making-Predictions"></a></h1><p>If we were asked to make a prediction about any passenger aboard the RMS Titanic whom we knew nothing about, then the best prediction we could make would be that they did not survive. This is because we can assume that a majority of the passengers (more than 50%) did not survive the ship sinking.<br>
The <code>predictions_0</code> function below will always predict that a passenger did not survive.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">predictions_0</span><span class="p">(</span><span class="n">data</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Model with no features. Always predicts a passenger did not survive. &quot;&quot;&quot;</span>

    <span class="n">predictions</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">passenger</span> <span class="ow">in</span> <span class="n">data</span><span class="o">.</span><span class="n">iterrows</span><span class="p">():</span>
        
        <span class="c1"># Predict the survival of &#39;passenger&#39;</span>
        <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    
    <span class="c1"># Return our predictions</span>
    <span class="k">return</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">predictions</span><span class="p">)</span>

<span class="c1"># Make the predictions</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">predictions_0</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-1">Question 1<a class="anchor-link" href="#Question-1"></a></h3><ul>
<li>Using the RMS Titanic data, how accurate would a prediction be that none of the passengers survived?</li>
</ul>
<p><strong>Hint:</strong> Run the code cell below to see the accuracy of this prediction.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">outcomes</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Predictions have an accuracy of 61.62%.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer:</strong> 61.62%</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<p>Let's take a look at whether the feature <strong>Sex</strong> has any indication of survival rates among passengers using the <code>survival_stats</code> function. This function is defined in the <code>visuals.py</code> Python script included with this project. The first two parameters passed to the function are the RMS Titanic data and passenger survival outcomes, respectively. The third parameter indicates which feature we want to plot survival statistics across.<br>
Run the code cell below to plot the survival outcomes of passengers based on their sex.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vs</span><span class="o">.</span><span class="n">survival_stats</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">outcomes</span><span class="p">,</span> <span class="s1">&#39;Sex&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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B/bdd1+ee+457rvvPiZOnMgXv/hFhg4dytNPP71UWTfccAPbbbcdQ4YMYcSIEQBcffXVnHrqqUvWGTVqFJMmTVqy769//evsuuuufOtb3+LII49cst6kSZM45JBDgHd7H7785S9z2WXvXtn63HPP5eKLLwbgoosuYvjw4eywww6cc845nXlY22TAS5K6zfvf/37eeecdZs+evdT88847jz333JNHH32U0aNH89xzz7W6/aJFi9htt92YOnUqI0aM4MorrwTg1FNP5dhjj+Wxxx7j6KOP5rTTTuPDH/4wo0eP5qKLLmLKlClsueWWS5V1/vnnc8cddzB16lQmTpzYYd0XLVrEdtttx4MPPshZZ53FAw88wKJFiwC47rrrGDNmzFLrjx07luuuu27J9PXXX88RRxzBnXfeyfTp03nooYeYMmUKkydP5p577un4wVtOBrwkqVu11jq/5557+OQnPwnAwQcfzLrrrtvqtquvvvqSVv/OO+/MX/7yFwDuv/9+PvGJTwBwzDHHcO+993ZYjz322IPjjz+eK6+8krfffrvD9Zuamvj4xz8OQJ8+fTjwwAO5+eabWbx4MbfeeiuHHnroUuvvuOOOzJ49m+eff56pU6ey7rrrstlmm3HnnXdy5513suOOO7LTTjvx5JNPMn369A73v7w8Bi9J6jbPPPMMTU1NbLjhhkybNm2pZfWMHO/bt++S9Zqamli8eHGr69VT1uWXX86DDz7IrbfeytChQ5kyZQp9+vRZMggQWOqnav369VvquPuYMWP40Y9+xHrrrcfw4cPp37//Mvs4/PDDmTBhAn//+98ZO3YsUHzBOeusszjppJM6rGNn2IKvFeFfd/1JWuXMmTOHk08+mVNPPXWZAB4xYgTXXnstALfffjsvvfTScpX94Q9/mPHjxwNw7bXXsueeewLQv39/Fi5c2Oo2Tz/9NLvuuivnn38+G2ywATNmzGDw4MFMmTKFd955hxkzZvDQQw+1uc+RI0fyyCOPcOWVVy7TPd9s7NixjB8/ngkTJnD44YcDcMABB/DTn/6UV155BYC//e1vyxyy6Aq24CVJDfPaa68xdOhQ3nrrLfr06cMxxxzDGWecscx655xzDkcddRQ77bQTe++9N5ttttly7efSSy/l05/+NBdddBEDBw7kP//zP4EiYD/zmc9w6aWXMmHChKWOw3/xi19k+vTpZCb77rsvQ4YMAWCLLbZg++23Z7vttmOnnXZqc59NTU2MGjWKq6++mnHjxrW6zrbbbsvChQvZZJNN2GijjQDYf//9mTZtGrvvvjtQDN675ppr2HDDDZfrPnck2hqpuDIYNmxYdun14G1Zdp+V+HUnrSymTZvG1ltv3dPV0Apq7fmLiMmZWddvC+2ilySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlyQ11De/+U223XZbdthhB4YOHcqDDz7Y6TInTpzIhRde2AW1K36HXkWe6EaSVhFxXtee6yPP6fh8Fvfffz+33HILjzzyCGussQZz587lzTffrKv8xYsX06dP6zE1evRoRo8evVz1XdXYgpckNcysWbPYYIMNWGONNQDYYIMN2HjjjZdcUhXg4YcfZuTIkUBxSdUTTzyR/fffn2OPPZZdd92Vxx9/fEl5I0eOZPLkyUsu67pgwQIGDx685Pzxr776KptuuilvvfUWTz/9NAceeCA777wze+21F08++SQAzz77LLvvvjvDhw/n7LPP7sZHo3sZ8JKkhtl///2ZMWMGH/zgBznllFO4++67O9xm8uTJ3HTTTfz85z9n7NixXH/99UDxZeH5559n5513XrLuOuusw5AhQ5aUe/PNN3PAAQfQt29fTjzxRH7wgx8wefJkvve973HKKacAcPrpp/PZz36WP/zhD/zDP/xDA+5172DAS5IaZu2112by5MlcccUVDBw4kDFjxnD11Ve3u83o0aNZc801ATjyyCO54YYbgHevp97SmDFjllx3ffz48YwZM4ZXXnmF++67jyOOOIKhQ4dy0kknMWvWLAB+//vfc9RRRwHFpWWrymPwkqSGampqYuTIkYwcOZLtt9+ecePGLXVZ1tpLsgKstdZaS25vsskmrL/++jz22GNcd911/OQnP1mm/NGjR3PWWWfx4osvMnnyZPbZZx8WLVrEgAEDmDJlSqt1qudysis7W/CSpIb585//zPTp05dMT5kyhc0335zBgwczefJkAH75y1+2W8bYsWP57ne/y4IFC9h+++2XWb722muzyy67cPrppzNq1Ciampp473vfyxZbbLGk9Z+ZTJ06FYA99thjqUvLVpUBL0lqmFdeeYXjjjuObbbZhh122IEnnniCc889l3POOYfTTz+dvfbai6ampnbLOPzwwxk/fjxH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
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
<p>Examining the survival statistics, a large majority of males did not survive the ship sinking. However, a majority of females <em>did</em> survive the ship sinking. Let's build on our previous prediction: If a passenger was female, then we will predict that they survived. Otherwise, we will predict the passenger did not survive.<br>
Fill in the missing code below so that the function will make this prediction.<br>
<strong>Hint:</strong> You can access the values of each feature for a passenger like a dictionary. For example, <code>passenger['Sex']</code> is the sex of the passenger.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">predictions_1</span><span class="p">(</span><span class="n">data</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Model with one feature: </span>
<span class="sd">            - Predict a passenger survived if they are female. &quot;&quot;&quot;</span>
    
    <span class="n">predictions</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">passenger</span> <span class="ow">in</span> <span class="n">data</span><span class="o">.</span><span class="n">iterrows</span><span class="p">():</span>
        
        <span class="c1"># If Passenger is female predict they will survive</span>
        <span class="c1"># Else predict they will not survive</span>
        <span class="k">if</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;female&#39;</span><span class="p">:</span>
            <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    
    <span class="c1"># Return our predictions</span>
    <span class="k">return</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">predictions</span><span class="p">)</span>

<span class="c1"># Make the predictions</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">predictions_1</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-2">Question 2<a class="anchor-link" href="#Question-2"></a></h3><ul>
<li>How accurate would a prediction be that all female passengers survived and the remaining passengers did not survive?</li>
</ul>
<p><strong>Hint:</strong> Run the code cell below to see the accuracy of this prediction.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">outcomes</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Predictions have an accuracy of 78.68%.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer</strong>: 78.68%</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<p>Using just the <strong>Sex</strong> feature for each passenger, we are able to increase the accuracy of our predictions by a significant margin. Now, let's consider using an additional feature to see if we can further improve our predictions. For example, consider all of the male passengers aboard the RMS Titanic: Can we find a subset of those passengers that had a higher rate of survival? Let's start by looking at the <strong>Age</strong> of each male, by again using the <code>survival_stats</code> function. This time, we'll use a fourth parameter to filter out the data so that only passengers with the <strong>Sex</strong> 'male' will be included.<br>
Run the code cell below to plot the survival outcomes of male passengers based on their age.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vs</span><span class="o">.</span><span class="n">survival_stats</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">outcomes</span><span class="p">,</span> <span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;Sex == &#39;male&#39;&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfsAAAGDCAYAAAAs+rl+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzt3Xu8VWW56PHfI6B4K7xgqahg28wrqHjLG0fbaopopYKZmrmTLm5p16m0NLVO7cpq76xMNEvOjsRLpXhLO25vlWKQYCq68ZbgDURBRUvR5/wxxoLJYrHWhDXnugx+389nfdYc92fMOcZ85vuOd4w3MhNJklRda3R3AJIkqblM9pIkVZzJXpKkijPZS5JUcSZ7SZIqzmQvSVLFmeylLhQRx0fELQ1Yz8cj4g+NiGkVt39RRJy9CsttGRGvRkSfZsTViO1HREbEP3VlXFKzmexXYxHxZES8Xn75PR8Rv4iI9bo7rq4WEYMi4tcR8UJELIyIv0bEx5uxrcycmJkHN2PdtSLilIh4OCJeKT/bGyJi/XLaZRHxf1ZiXcv9sMjMT2XmN+pY9smI+EDNck9l5nqZ+dbK7E876x8fERfWDPeLiEUrGLdX6+1HxO0R8S+d2P65EXFuq3FDIuLt2hgaISJW+FCUVudyy99mndzeiIiY05l1qOcw2euIzFwP2BXYHTirm+Npqojo28bo/wJmA1sBGwEnAs83cP1dKiIOAL4FHJeZ6wPbAVd2b1RNcydwQM3wcOApYP9W4wCmdVFMJwIvAWMiYq0u2iaU53LN3zNduO3l9IRzQUuZ7AVAZj4N3ATsCBARJ0fEzLJk+HhEjG2ZNyI2jojrI2JBRLwYEXdFxBrltC9HxNPlco9ExEHl+DUi4oyIeCwi5kfElRGxYTltcFl1elJEPFWWsL9as721I2JCRLxUxvSl2hJHRGxWlsznRcQTEXF6zbRzI+LqiPhlRLwMfLyN3d8duCwzF2Xm4sy8LzNvKpdfrnRTW1ptY/1fKUtYG9bMv0u5T/1qS8llVfj3Wq372oj4fPm65f16JSIeiogP1flx7g7cnZn3AWTmi5k5ITNfiYhTgeOBL5Wlv+va21ZEbAdcBOxdzr+gHL+kdmBFx0NE/BewJXBdueyXaj7rvuWyG0ZRo/RM+fle094629jXO4DtImLjcng/YBKwbqtxd2fmm7Xbj4hvltN+XMb345r1fiAiZpUx/SQios73HopkfxbwJnBE7YSIOLg8LxZGxIURcUfU1CxExCfKY/yliLg5IrZaie22KSL2iog/le/ljIgYUTPt5GjjPI+IdSm+DzaLmpqCaFUr1Pr8KM+NL0fE/cCi8n1e4fmpLpSZ/q2mf8CTwAfK11sADwLfKIcPB94DBEXJ6TVg13Lav1MkgH7l337lfNtSlJA3K+cbDLynfP054B5gELAWMB64vGa+BC4B1gaGAv8Atiunf5viS32Dcvn7gTnltDUoSmxfA9YEtgYeBw4pp59L8aV7VDnv2m28D/8P+CMwBtiy1bQRLdtawfu23PqB/wY+WTP/+cBF5euPA38oX+9fvl9RDm8AvF7z/h0DbFaudzSwCNi09Xra2J/9yvWcB+wDrNVq+mXA/2k1bqW2VbuOFR0Prd+rVp9133L4BuCKct/7AQd0tM429vcJ4EPl6+uBA4GJrcZ9bQXbvx34l1bry3KZARQ/VuYBh9Z5Tu1HcexuAPwImFwzbWPgZeDDQF9gXHns/Es5/SjgUYqamL4UPxj+tLLncqvxmwPzgcPKz/afy+GBdZznI1j+2F/m2Gk9TxnHdIrvk7Xp4Pz0r+v+LNnrmrK09geKhPotgMy8ITMfy8IdwC0UX2RQfEFtCmyVmW9m5l1ZnOlvUSTy7SOiX2Y+mZmPlcuMBb6amXMy8x8USfLoWLaq77zMfD0zZwAzKJI+wLHAtzLzpcycA1xQs8zuFF9cX8/MNzLzcYofDWNq5rk7M6/JzLcz8/U23oNjgLuAs4EnImJ6ROy+Eu9h6/X/CjgOoCwRjinHtXYXRWJpeV+PLtf1DEBmXpWZz5TrvQKYBezRUTCZeRdFQtmVIpnOj4gfRDuN0lZ1W6UVHQ/tiohNgQ8Cnyo/2zfLY21l13kHsH9Z8t+D4kflXTXj9innWRnfzswFmfkUcBswrM7lTgJuysyXKD7zD0bEJuW0w4AHM/M3mbmY4jh+rmbZscC/Z+bMcvq3gGErUbq/piy9L2ipIQE+BtyYmTeWn+3vgallLB2d56vqgsycXZ4L9Zyf6gImex2VmQMyc6vM/ExLMoyID0bEPWUV6gKKL4eWatHzKUogt5RVf2cAZOajFCX4c4G5ETEpljYS2gr4bcuXETCT4sfBu2piqf3iew1oaSy4GUUJuEXt660oqhoX1Kz7K63WWzv/cspEc0Zm7lAuN53ii7PeqtvW67+aotp7M4rSe1Ikn9bbTYoq5+PKUR+lKJECEBEnlj88WvZrR5Z+Bu3KzJsy8whgQ+BIitL5ChuidWZbrOB4qMMWwItlYuzMOu+keJ93Ah7PzNcofry2jFsbmFJnTC1WdCyuUESsTfHDcSJAZt5N0X7go+UsyxzH5edfe4loK+CHNZ/BixQl7s3rjLnlXB6QmUfVrPOYVufHvhQ/pDo6z1fVyp6f6gImey0nikZFvwa+B7wrMwcAN1J88ZCZr2TmFzJza4prkp+P8tp8Zv4qM/elOMkT+E652tnAB2u+jAZkZv8s2gp05FmK6vsWW9S8ng080Wq962fmYTXz1N21Y2a+UO73ZhSJchGwTsv0snQ8sPVirdaxgKKEdCzFF/3l7ZRKL6eo4dgK2JPifaccvgQ4Ddio/AweoPwMVmJ/3s7MWykuLezYVrx1bKvd96+946GDZWcDG0bEgJVcZ2t3UtQCHc7SH1UPUhwnhwN/zsy/ryj89vZtJX0IeAdwYUQ8FxHPUSTqE8vpyxzH5Y/J2uN6NjC21bG8dmb+qRMxzQb+q9U6183Mb3d0ntP2e7PM+QC8u415aper5/xUFzDZqy1rUlTHzwMWR8QHgSW3i0XEyIj4p/LL6mWKEvpbEbFtRBxYfon8neK6ccstVhcB32ypkoyIgRFxZJ3xXAmcGREbRMTmFEmpxb3Ay2WjoLUjok9E7Lgy1fAR8Z1ymb5R3J72aeDRzJwP/A/QPyIOj4h+FNdR62lh/SuKL/mP0HYVPgBZNKKbB/wMuLn8oQCwLsWX5rwyxpNZmqw72p8jI2JM+X5FROxBcT32nnKW5ymunbboaFvPA4MiYs0VbK/N42EF26rd92cpGoFdWMbaLyL2r2OdrdfzaLmdcZTJvvxxNaUcd2dby3UU3yo4Cfg5RW3CsPJvH4qq+J0oLqnsFBFHlZevPsuyyfIiiuN8B4CIeGdEHNPJmH4JHBERh5TnRv8oGtUNooPznOK92Sgi3lkzbjpwWBQNK99NUZPXnk6fn2oMk72Wk5mvAKdTJNmXKEqnk2tm2YaiUdurwN3AhZl5O8UXx7eBFyiqQTehqLID+GG5jlsi4hWKxLNnnSF9naK684lyu1dTNIIii/ulj6D4Yn2i3PbPgHe2uaa2rQP8FlhA0XhoK2BUuf6FwGfKdT5NUbKp597jyRTv0/NZtEFoz+XAB6j5UZCZDwHfp3h/n6dIIH+sc39eAj5Jcd39ZYov/PMzs+USwaUU7SoWRMQ1dWzrvylKys9FxAttbG9FxwMUDe3OKrf1v9tY9gSK6/MPA3NZmjzaW2db7qSocamN+y6KY7C9ZP9DipqVlyLignbma1f5I/Qg4D8z87mav2nA74CTylqjY4DvUjSS257i+nnLsfxbipqwSVHc2fEARZuGVZaZsyku43yFIqnPBr4IrNHReZ6ZD1Mcm4+Xn99mFLepzqBoiHcLRePK9rbfiPNTDdDSYlbqNSLi08CYzDygw5mlHiqKxoNzgOMz87bujkfVZslePV5EbBoR+0Rx7/a2wBcoSuJSr1JWpw8oL3V9heL6+D0dLCZ1mk84Um+wJsV9+UMoqtonAQ19FKnURfamuFyzJvAQRQv6tm4HlRrKanxJkirOanxJkirOZC9JUsX16mv2G2+8cQ4ePLi7w5AkqctMmzbthcxs/XCvdvXqZD948GCmTp3a3WFIktRlIuJvK7uM1fiSJFWcyV6SpIoz2UuSVHG9+pq9JKl9b775JnPmzOHvf19Rx3/qqfr378+gQYPo169fp9dlspekCpszZw7rr78+gwcPpuhEUL1BZjJ//nzmzJnDkCFDOr0+q/ElqcL+/ve/s9FGG5noe5mIYKONNmpYjYzJXpIqzkTfOzXyczPZS5Kaqk+fPgwbNowddtiBoUOH8oMf/IC3334bgKlTp3L66ae3udzgwYN54YUXOr39a665hoceeqjT61kZhx12GAsWLOjSbbbHa/aStDoZO7ax6xs/vsNZ1l57baZPnw7A3Llz+ehHP8rChQs577zzGD58OMOHD29sTK1cc801jBw5ku23376h633rrbfo06dPm9NuvPHGhm6rsyzZS5K6zCabbMLFF1/Mj3/8YzKT22+/nZEjRwIwf/58Dj74YHbZZRfGjh3LinplXW+99fjqV7/K0KFD2WuvvXj++ecB+Nvf/sZBBx3EzjvvzEEHHcRTTz3Fn/70JyZPnswXv/hFhg0bxmOPPbbMuq666ip23HFHhg4dyv777w/AZZddxmmnnbZknpEjR3L77bcv2fbXvvY19txzT771rW9x7LHHLpnv9ttv54gjjgCW1kp8+ctf5sILl/bIfe655/L9738fgPPPP5/dd9+dnXfemXPOOaczb2uHTPaSpC619dZb8/bbbzN37txlxp933nnsu+++3HfffYwaNYqnnnqqzeUXLVrEXnvtxYwZM9h///255JJLADjttNM48cQTuf/++zn++OM5/fTTef/738+oUaM4//zzmT59Ou95z3uWWdfXv/51br75ZmbMmMHkyZM7jH3RokXsuOOOTJkyhTPPPJN77rmHRYsWAXDFFVcwevToZeYfM2YMV1xxxZLhK6+8kmOOOYZbbrmFWbNmce+99zJ9+nSmTZvGnXfe2fGbt4pM9pKkLtdWqf3OO+/kYx/7GACHH344G2ywQZvLrrnmmktqA3bbbTeefPJJAO6++24++tGPAnDCCSfwhz/8ocM49tlnHz7+8Y9zySWX8NZbb3U4f58+ffjIRz4CQN++fTn00EO57rrrWLx4MTfccANHHnnkMvPvsssuzJ07l2eeeYYZM2awwQYbsOWWW3LLLbdwyy23sMsuu7Drrrvy8MMPM2vWrA63v6q8Zi9J6lKPP/44ffr0YZNNNmHmzJnLTKunBXq/fv2WzNenTx8WL17c5nz1rOuiiy5iypQp3HDDDQwbNozp06fTt2/fJQ0IgWVuf+vfv/8y1+lHjx7NT37yEzbccEN233131l9//eW2cfTRR3P11Vfz3HPPMWbMGKD4sXPmmWcyttFtKFbAZK/u00UHebepo+GStLqZN28en/rUpzjttNOWS8b7778/EydO5KyzzuKmm27ipZdeWql1v//972fSpEmccMIJTJw4kX333ReA9ddfn1deeaXNZR577DH23HNP9txzT6677jpmz57N4MGDufDCC3n77bd5+umnuffee1e4zREjRnDKKadwySWXLFeF32LMmDF88pOf5IUXXuCOO+4A4JBDDuHss8/m+OOPZ7311uPpp5+mX79+bLLJJiu1z/Uy2UuSmur1119n2LBhvPnmm/Tt25cTTjiBz3/+88vNd84553Dcccex6667csABB7Dllluu1HYuuOACPvGJT3D++eczcOBAfvGLXwBLk+0FF1zA1Vdfvcx1+y9+8YvMmjWLzOSggw5i6NChAAwZMoSddtqJHXfckV133XWF2+zTpw8jR47ksssuY8KECW3Os8MOO/DKK6+w+eabs+mmmwJw8MEHM3PmTPbee2+gaPj3y1/+smnJPlbU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
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
<p>Examining the survival statistics, the majority of males younger than 10 survived the ship sinking, whereas most males age 10 or older <em>did not survive</em> the ship sinking. Let's continue to build on our previous prediction: If a passenger was female, then we will predict they survive. If a passenger was male and younger than 10, then we will also predict they survive. Otherwise, we will predict they do not survive.<br>
Fill in the missing code below so that the function will make this prediction.<br>
<strong>Hint:</strong> You can start your implementation of this function using the prediction code you wrote earlier from <code>predictions_1</code>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">predictions_2</span><span class="p">(</span><span class="n">data</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Model with two features: </span>
<span class="sd">            - Predict a passenger survived if they are female.</span>
<span class="sd">            - Predict a passenger survived if they are male and younger than 10. &quot;&quot;&quot;</span>
    
    <span class="n">predictions</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">passenger</span> <span class="ow">in</span> <span class="n">data</span><span class="o">.</span><span class="n">iterrows</span><span class="p">():</span>
        
         <span class="c1"># If Passenger is female predict they will survive</span>
        <span class="c1"># Else predict they will not survive</span>
        <span class="k">if</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;female&#39;</span><span class="p">:</span>
            <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;male&#39;</span> <span class="ow">and</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span> <span class="o">&lt;</span> <span class="mi">10</span><span class="p">:</span>
            <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    
    <span class="c1"># Return our predictions</span>
    <span class="k">return</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">predictions</span><span class="p">)</span>

<span class="c1"># Make the predictions</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">predictions_2</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-3">Question 3<a class="anchor-link" href="#Question-3"></a></h3><ul>
<li>How accurate would a prediction be that all female passengers and all male passengers younger than 10 survived? </li>
</ul>
<p><strong>Hint:</strong> Run the code cell below to see the accuracy of this prediction.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">outcomes</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Predictions have an accuracy of 79.35%.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer</strong>: 79.24%</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<p>Adding the feature <strong>Age</strong> as a condition in conjunction with <strong>Sex</strong> improves the accuracy by a small margin more than with simply using the feature <strong>Sex</strong> alone. Now it's your turn: Find a series of features and conditions to split the data on to obtain an outcome prediction accuracy of at least 80%. This may require multiple features and multiple levels of conditional statements to succeed. You can use the same feature multiple times with different conditions.<br>
<strong>Pclass</strong>, <strong>Sex</strong>, <strong>Age</strong>, <strong>SibSp</strong>, and <strong>Parch</strong> are some suggested features to try.</p>
<p>Use the <code>survival_stats</code> function below to to examine various survival statistics.<br>
<strong>Hint:</strong> To use mulitple filter conditions, put each condition in the list passed as the last argument. Example: <code>["Sex == 'male'", "Age &lt; 18"]</code></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[423]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># vs.survival_stats(data, outcomes, &#39;Age&#39;, [&quot;Pclass == 3&quot;, &quot;Sex == &#39;female&#39;&quot;])</span>
<span class="c1"># vs.survival_stats(data, outcomes, &#39;Pclass&#39;, [&quot;Age &lt; 20&quot;, &quot;Sex == &#39;male&#39;&quot;])</span>
<span class="c1"># vs.survival_stats(data, outcomes, &#39;Pclass&#39;, [&quot;Age &lt; 30&quot;, &quot;Sex == &#39;female&#39;&quot;])</span>
<span class="c1"># vs.survival_stats(data, outcomes, &#39;Age&#39;, [&quot;Sex == &#39;female&#39;&quot;,&quot;Pclass == 1&quot;])</span>
<span class="c1"># vs.survival_stats(data, outcomes, &#39;Pclass&#39;, [&quot;Sex == &#39;female&#39;&quot;])</span>
<span class="c1"># vs.survival_stats(data, outcomes, &#39;Parch&#39;, [&quot;Sex == &#39;female&#39;&quot;])</span>
<span class="n">vs</span><span class="o">.</span><span class="n">survival_stats</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">outcomes</span><span class="p">,</span> <span class="s1">&#39;SibSp&#39;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;Sex == &#39;female&#39;&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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1J03utmJYoSwWeWvkoHAPRGxL0WNfftm6w4E5ndjbJK0dujq2nwNA5atv/76zJw5E4CFCxfy/ve/nyVLlnDxxRczfPhwhg8f3rUxtXD99dczcuRIdt111y4td8WKFfTp06fVZbfcckuX7qsjuq0Gn5l/zsytMnNQZg6iSOrDMvMp4Ebg1CjsDyzJzAXdFZskqXtstdVWjB8/nu9+97tkJlOnTmXkyJEAPPPMMxx++OHstddenHnmmbQ12ulGG23EZz/7WYYMGcL+++/P008/DcDf/vY3DjvsMPbcc08OO+wwnnjiCf74xz9y44038slPfpKhQ4fyyCOPrFLWddddx+67786QIUM4+OCDAbjyyis5++yzV64zcuRIpk6dunLfn//859lvv/348pe/zIknnrhyvalTp3LMMccAb7Q+fPrTn+ayy94Y8fqiiy7iG9/4BgCXXHIJ++yzD3vuuSfjxo3rzNvaqrol+Ii4GrgT2CUi5kXEGe2sfgvwKPAwcDnw0XrFJUlqrLe+9a28/vrrLFy4cJX5F198Me985zu59957GTVqFE888USr2y9dupT999+fWbNmcfDBB3P55ZcDcPbZZ3Pqqady3333cfLJJ3POOefwjne8g1GjRnHJJZcwc+ZMdtppp1XK+sIXvsCvfvUrZs2axY033rja2JcuXcruu+/OtGnTuOCCC7jrrrtYunQpANdccw2jR49eZf0xY8ZwzTXXrJy+9tprOeGEE5gyZQpz5szh7rvvZubMmcyYMYM77rhj9W/eGqhnL/qTMnObzOyXmQMz84oWywdl5uLyeWbmxzJzp8zcIzO9wbwkVVhrtfM77riDD3zgAwAcffTRbLbZZq1uu+66666s9e+99948/vjjANx55528//3vB+CUU07h97///WrjOPDAAzn99NO5/PLLWbFixWrX79OnD//yL/8CQN++fTnyyCO56aabWL58OTfffDPHHnvsKuvvtddeLFy4kPnz5zNr1iw222wzdthhB6ZMmcKUKVPYa6+9GDZsGA8++CBz5sxZ7f7XRK8ebEaS1Ps8+uij9OnTh6222orZs2evsqyWnuP9+vVbuV6fPn1Yvnx5q+vVUtYPfvADpk2bxs0338zQoUOZOXMmffv2XdkJEFjlUrX+/fuvct599OjRfO9732PzzTdnn332YeONN37TPo4//ngmT57MU089xZgxY4DiB84FF1zAmWeeudoYO8oEXydxcdd0ZMlxq+/AIkm9xaJFizjrrLM4++yz35SADz74YCZOnMjnPvc5br31Vp577rk1Kvsd73gHkyZN4pRTTmHixIm8853vBGDjjTfmxRdfbHWbRx55hP3224/99tuPm266iblz5zJo0CAuu+wyXn/9dZ588knuvvvuNvc5YsQIzjjjDC6//PI3Nc83GTNmDB/+8IdZvHgxt99+OwBHHHEEF154ISeffDIbbbQRTz75JP369WOrrbZao9fcHhO8JKmuXnnlFYYOHcqyZcvo27cvp5xyCuedd96b1hs3bhwnnXQSw4YN45BDDmGHHXZYo/1ceumlfOhDH+KSSy5hwIAB/OQnPwHeSLCXXnopkydPXuU8/Cc/+UnmzJlDZnLYYYcxZMgQAAYPHswee+zB7rvvzrBhw9rcZ58+fRg5ciRXXnklEyZMaHWd3XbbjRdffJHtttuObbYp7uF2+OGHM3v2bA444ACg6Lx31VVXdWmCj7Z6KfYGw4cPz546Hrw1eEk9wezZs3n729/e6DDUQa39/yJiRmau9tpCR5OTJKmCTPCSJFWQCV6SpAoywUuSVEEmeEmSKsgEL0lSBZngJUl19aUvfYnddtuNPffck6FDhzJt2rROl3njjTfyla98pQuiK65BryJvdCNJa5GuukdHk9Xdq+POO+/kl7/8Jffccw/rrbceixcv5rXXXqup7OXLl9O3b+tpatSoUYwaNWqN412bWIOXJNXNggUL2HLLLVlvvfUA2HLLLdl2221XDqcKMH36dEaMGAEUw6mOHTuWww8/nFNPPZX99tuP+++/f2V5I0aMYMaMGSuHdF2yZAmDBg1aee/4l19+me23355ly5bxyCOPcOSRR7L33ntz0EEH8eCDDwLw2GOPccABB7DPPvtw4YUXduO70b1M8JKkujn88MOZO3cub3vb2/joRz+68l7s7ZkxYwY33HADP/3pTxkzZgzXXnstUPxYmD9/PnvvvffKdd/ylrcwZMiQleXedNNNHHHEEfTr14+xY8fyne98hxkzZvD1r3+dj360GIn83HPP5SMf+Qh/+tOf+Id/+Ic6vOqewQQvSaqbjTbaiBkzZjB+/HgGDBjA6NGjufLKK9vdZtSoUay//voAnHjiiVx33XXAG2OptzR69OiVY65PmjSJ0aNH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
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
<p>After exploring the survival statistics visualization, fill in the missing code below so that the function will make your prediction.<br>
Make sure to keep track of the various features and conditions you tried before arriving at your final prediction model.<br>
<strong>Hint:</strong> You can start your implementation of this function using the prediction code you wrote earlier from <code>predictions_2</code>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[424]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">predictions_3</span><span class="p">(</span><span class="n">data</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Model with multiple features. Makes a prediction with an accuracy of at least 80%. &quot;&quot;&quot;</span>
    
    <span class="n">predictions</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">passenger</span> <span class="ow">in</span> <span class="n">data</span><span class="o">.</span><span class="n">iterrows</span><span class="p">():</span>
        
        <span class="k">if</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;female&#39;</span><span class="p">:</span>
            <span class="k">if</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;SibSp&#39;</span><span class="p">]</span> <span class="o">&lt;</span> <span class="mi">3</span><span class="p">:</span>
                <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
            <span class="k">else</span><span class="p">:</span>
                <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">if</span> <span class="n">passenger</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span> <span class="o">&lt;</span> <span class="mi">10</span><span class="p">:</span>
                <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>   
            <span class="k">else</span><span class="p">:</span>
                <span class="n">predictions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>

    <span class="c1"># Return our predictions</span>
    <span class="k">return</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">predictions</span><span class="p">)</span>

<span class="c1"># Make the predictions</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">predictions_3</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-4">Question 4<a class="anchor-link" href="#Question-4"></a></h3><ul>
<li>Describe the steps you took to implement the final prediction model so that it got <strong>an accuracy of at least 80%</strong>. What features did you look at? Were certain features more informative than others? Which conditions did you use to split the survival outcomes in the data? How accurate are your predictions?</li>
</ul>
<p><strong>Hint:</strong> Run the code cell below to see the accuracy of your predictions.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[425]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">outcomes</span><span class="p">,</span> <span class="n">predictions</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Predictions have an accuracy of 80.36%.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer</strong>: I looked at the data through a lot of variations of filtering:</p>

<pre><code>vs.survival_stats(data, outcomes, 'Age', ["Pclass == 3", "Sex == 'female'"])
vs.survival_stats(data, outcomes, 'Pclass', ["Age &lt; 20", "Sex == 'male'"])
vs.survival_stats(data, outcomes, 'Pclass', ["Age &lt; 30", "Sex == 'female'"])
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'female'","Pclass == 1"])
vs.survival_stats(data, outcomes, 'Pclass', ["Sex == 'female'"])
vs.survival_stats(data, outcomes, 'Parch', ["Sex == 'female'"])
vs.survival_stats(data, outcomes, 'SibSp', ["Sex == 'female'"])</code></pre>
<p>Ultimately I implemented the amount of siblings aboard(SibSp) for all female passengers. Onces that less than 3 sibling were more likely to have surived according to the data. This of course could be correlation and should be investigated further to make sure its not actually overfitting to the dataset.  Prediction is now at 80.36%</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion"></a></h1><p>After several iterations of exploring and conditioning on the data, you have built a useful algorithm for predicting the survival of each passenger aboard the RMS Titanic. The technique applied in this project is a manual implementation of a simple machine learning model, the <em>decision tree</em>. A decision tree splits a set of data into smaller and smaller groups (called <em>nodes</em>), by one feature at a time. Each time a subset of the data is split, our predictions become more accurate if each of the resulting subgroups are more homogeneous (contain similar labels) than before. The advantage of having a computer do things for us is that it will be more exhaustive and more precise than our manual exploration above. <a href="http://www.r2d3.us/visual-intro-to-machine-learning-part-1/">This link</a> provides another introduction into machine learning using a decision tree.</p>
<p>A decision tree is just one of many models that come from <em>supervised learning</em>. In supervised learning, we attempt to use features of the data to predict or model things with objective outcome labels. That is to say, each of our data points has a known outcome value, such as a categorical, discrete label like <code>'Survived'</code>, or a numerical, continuous value like predicting the price of a house.</p>
<h3 id="Question-5">Question 5<a class="anchor-link" href="#Question-5"></a></h3><p><em>Think of a real-world scenario where supervised learning could be applied. What would be the outcome variable that you are trying to predict? Name two features about the data used in this scenario that might be helpful for making the predictions.</em></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer</strong>: I think a neat example of supervised Machine Learning is automating quality control in manufacturing.</p>
<p>In this example the outcome variable would be if a part is good or not.
two importan features of the data would be: Proper tagging of good / bad parts and a good amount of data to make accurate predictions.</p>
<p>The exact features about the data would depend on the part being monitored, but we can use a cat / dog clasisfier as an example. When training a classifier with supervised learning you need the data to be properly tagged. Once the training begins the algorithm will extract features, such as the shape of the dogs head, or the shape of cats ears. The features extracted would likely be many.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

</div>
</div>
</div>

