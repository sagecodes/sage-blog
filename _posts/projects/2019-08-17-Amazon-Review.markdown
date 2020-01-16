---
layout: project
title:  "Review Classifier"
permalink: /amazon-review-prediction/
date: 2019-08-17
categories: project
tags: machine-learning case-study
author: Sage Elliott
published: true
github_url: https://github.com/sagecodes/Amazon-Review-Classification-Random-Forest
---

Use Random Forest model, sklearn, python and the Alexa Amazon Review dataset to predict positive or negative reviews based on text


## About:

This project / case study is for phase 1 of my [100 days of machine learning code](https://sageelliott.com/100daysofmlcode/) challenge.

This is a homework solution to a section in [Machine Learning Classification Bootcamp in Python](https://www.udemy.com/machine-learning-classification). 

#### Problem Statement:

Predict if feedback is Positive or Negative on Amazon Alexa reviews based on the text of the review. 

## Technology used:

#### Model(s): 
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)

#### Dataset(s):

- [Amazon Alexa Reviews](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)

#### Libraries:

- [Scikit Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
<!--- [numpy](https://www.numpy.org/)-->
- [seaborn](https://seaborn.pydata.org/)

#### Resources:

- [Scikit Learn Random forest classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Scikit Feature Extraction | CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

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
<h3 id="Import-Data-&amp;-Libraries">Import Data &amp; Libraries<a class="anchor-link" href="#Import-Data-&amp;-Libraries">&#182;</a></h3>
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
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/amazon/amazon_alexa.tsv&#39;</span><span class="p">,</span> <span class="n">sep</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\t</span><span class="s1">&#39;</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>31-Jul-18</td>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Music</td>
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
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(3150, 5)</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">keys</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Index([&#39;rating&#39;, &#39;date&#39;, &#39;variation&#39;, &#39;verified_reviews&#39;, &#39;feedback&#39;], dtype=&#39;object&#39;)</pre>
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
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;verified_reviews&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0                                           Love my Echo!
1                                               Loved it!
2       Sometimes while playing a game, you can answer...
3       I have had a lot of fun with this thing. My 4 ...
4                                                   Music
5       I received the echo as a gift. I needed anothe...
6       Without having a cellphone, I cannot use many ...
7       I think this is the 5th one I&#39;ve purchased. I&#39;...
8                                             looks great
9       Love it! I’ve listened to songs I haven’t hear...
10      I sent it to my 85 year old Dad, and he talks ...
11      I love it! Learning knew things with it eveyda...
12      I purchased this for my mother who is having k...
13                                     Love, Love, Love!!
14                               Just what I expected....
15                              I love it, wife hates it.
16      Really happy with this purchase.  Great speake...
17      We have only been using Alexa for a couple of ...
18      We love the size of the 2nd generation echo. S...
19      I liked the original Echo. This is the same bu...
20      Love the Echo and how good the music sounds pl...
21      We love Alexa! We use her to play music, play ...
22      Have only had it set up for a few days. Still ...
23      I love it. It plays my sleep sounds immediatel...
24      I got a second unit for the bedroom, I was exp...
25                                        Amazing product
26      I love my Echo. It&#39;s easy to operate, loads of...
27                              Sounds great!! Love them!
28      Fun item to play with and get used to using.  ...
29                                Just like the other one
                              ...                        
3120                                                     
3121    I like the hands free operation vs the Tap. We...
3122    I dislike that it confuses my requests all the...
3123                                                     
3124    Love my Alexa! Actually have 3 throughout the ...
3125    This product is easy to use and very entertain...
3126                                                     
3127    works great but speaker is not the good for mu...
3128      Outstanding product - easy to use.  works great
3129    We have six of these throughout our home and t...
3130            Use the product for music and it’s great!
3131                           Easy to set-up and to use.
3132                                     It works great!!
3133    I like having more Alexa devices in my house a...
3134                                           PHENOMENAL
3135                 I loved it does exactly what it says
3136    I used it to control my smart home devices. Wo...
3137                                      Very convenient
3138    Este producto llegó y a la semana se quedó sin...
3139              Easy to set up Ready to use in minutes.
3140                                                Barry
3141                                                     
3142    My three year old loves it.  Good for doing ba...
3143           Awesome device wish I bought one ages ago.
3144                                              love it
3145    Perfect for kids, adults and everyone in betwe...
3146    Listening to music, searching locations, check...
3147    I do love these things, i have them running my...
3148    Only complaint I have is that the sound qualit...
3149                                                 Good
Name: verified_reviews, Length: 3150, dtype: object</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;variation&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0           Charcoal Fabric 
1           Charcoal Fabric 
2             Walnut Finish 
3           Charcoal Fabric 
4           Charcoal Fabric 
5       Heather Gray Fabric 
6          Sandstone Fabric 
7           Charcoal Fabric 
8       Heather Gray Fabric 
9       Heather Gray Fabric 
10          Charcoal Fabric 
11          Charcoal Fabric 
12               Oak Finish 
13          Charcoal Fabric 
14               Oak Finish 
15      Heather Gray Fabric 
16      Heather Gray Fabric 
17      Heather Gray Fabric 
18          Charcoal Fabric 
19         Sandstone Fabric 
20          Charcoal Fabric 
21          Charcoal Fabric 
22      Heather Gray Fabric 
23          Charcoal Fabric 
24         Sandstone Fabric 
25         Sandstone Fabric 
26          Charcoal Fabric 
27          Charcoal Fabric 
28          Charcoal Fabric 
29          Charcoal Fabric 
                ...         
3120              Black  Dot
3121              Black  Dot
3122              Black  Dot
3123              Black  Dot
3124              Black  Dot
3125              Black  Dot
3126              Black  Dot
3127              Black  Dot
3128              White  Dot
3129              White  Dot
3130              Black  Dot
3131              Black  Dot
3132              Black  Dot
3133              White  Dot
3134              Black  Dot
3135              White  Dot
3136              Black  Dot
3137              Black  Dot
3138              White  Dot
3139              White  Dot
3140              White  Dot
3141              Black  Dot
3142              White  Dot
3143              Black  Dot
3144              Black  Dot
3145              Black  Dot
3146              Black  Dot
3147              Black  Dot
3148              White  Dot
3149              Black  Dot
Name: variation, Length: 3150, dtype: object</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">positive</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="p">[</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;feedback&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">]</span>
<span class="n">negative</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="p">[</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;feedback&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">0</span><span class="p">]</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">positive</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>



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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>31-Jul-18</td>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Music</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I received the echo as a gift. I needed anothe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>31-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Without having a cellphone, I cannot use many ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I think this is the 5th one I've purchased. I'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>looks great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Love it! I’ve listened to songs I haven’t hear...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I sent it to my 85 year old Dad, and he talks ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it! Learning knew things with it eveyda...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>I purchased this for my mother who is having k...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love, Love, Love!!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>Just what I expected....</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I love it, wife hates it.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Really happy with this purchase.  Great speake...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>We have only been using Alexa for a couple of ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love the size of the 2nd generation echo. S...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I liked the original Echo. This is the same bu...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love the Echo and how good the music sounds pl...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love Alexa! We use her to play music, play ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Have only had it set up for a few days. Still ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it. It plays my sleep sounds immediatel...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I got a second unit for the bedroom, I was exp...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Amazing product</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love my Echo. It's easy to operate, loads of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Sounds great!! Love them!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Fun item to play with and get used to using.  ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Just like the other one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I like the hands free operation vs the Tap. We...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I dislike that it confuses my requests all the...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3123</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Love my Alexa! Actually have 3 throughout the ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>This product is easy to use and very entertain...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>works great but speaker is not the good for mu...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Outstanding product - easy to use.  works great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>We have six of these throughout our home and t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Use the product for music and it’s great!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Easy to set-up and to use.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>It works great!!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I like having more Alexa devices in my house a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>PHENOMENAL</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I loved it does exactly what it says</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I used it to control my smart home devices. Wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Very convenient</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Este producto llegó y a la semana se quedó sin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Easy to set up Ready to use in minutes.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Barry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>My three year old loves it.  Good for doing ba...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Awesome device wish I bought one ages ago.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>love it</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Perfect for kids, adults and everyone in betwe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Listening to music, searching locations, check...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I do love these things, i have them running my...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Only complaint I have is that the sound qualit...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>4</td>
      <td>29-Jul-18</td>
      <td>Black  Dot</td>
      <td>Good</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2893 rows × 5 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">negative</span>
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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>It's like Siri, in fact, Siri answers more acc...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>111</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Sound is terrible if u want good music too get...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Not much features.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Stopped working after 2 weeks ,didn't follow c...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Sad joke. Worthless.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>187</th>
      <td>2</td>
      <td>29-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Really disappointed Alexa has to be plug-in to...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2</td>
      <td>29-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>It's got great sound and bass but it doesn't w...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2</td>
      <td>29-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I am not super impressed with Alexa. When my P...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>2</td>
      <td>29-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Too difficult to set up.  It keeps timing out ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>341</th>
      <td>1</td>
      <td>28-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Alexa hardly came on..</td>
      <td>0</td>
    </tr>
    <tr>
      <th>350</th>
      <td>1</td>
      <td>31-Jul-18</td>
      <td>Black</td>
      <td>Item no longer works after just 5 months of us...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>1</td>
      <td>29-Jul-18</td>
      <td>Black</td>
      <td>This thing barely works. You have to select 3r...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>368</th>
      <td>1</td>
      <td>28-Jul-18</td>
      <td>Black</td>
      <td>I returned 2 Echo Dots &amp; am only getting refun...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>369</th>
      <td>1</td>
      <td>28-Jul-18</td>
      <td>Black</td>
      <td>not working</td>
      <td>0</td>
    </tr>
    <tr>
      <th>373</th>
      <td>1</td>
      <td>27-Jul-18</td>
      <td>Black</td>
      <td>I'm an Echo fan but this one did not work</td>
      <td>0</td>
    </tr>
    <tr>
      <th>374</th>
      <td>1</td>
      <td>26-Jul-18</td>
      <td>Black</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>376</th>
      <td>2</td>
      <td>26-Jul-18</td>
      <td>Black</td>
      <td>Doesn't always respond when spoken to with pro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1</td>
      <td>25-Jul-18</td>
      <td>White</td>
      <td>It worked for a month or so then it stopped. I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>382</th>
      <td>2</td>
      <td>25-Jul-18</td>
      <td>Black</td>
      <td>Poor quality. Gave it away.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>1</td>
      <td>24-Jul-18</td>
      <td>Black</td>
      <td>Never could get it to work. A techie friend lo...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>394</th>
      <td>2</td>
      <td>22-Jul-18</td>
      <td>White</td>
      <td>Initially, this echo dot worked very well. Ove...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>396</th>
      <td>1</td>
      <td>20-Jul-18</td>
      <td>Black</td>
      <td>I bought an echo dot that had been refurbished...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>398</th>
      <td>1</td>
      <td>19-Jul-18</td>
      <td>Black</td>
      <td>Dont trust this....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>406</th>
      <td>1</td>
      <td>16-Jul-18</td>
      <td>White</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>418</th>
      <td>1</td>
      <td>13-Jul-18</td>
      <td>Black</td>
      <td>I wanted to use these as a radio and intercom ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>1</td>
      <td>12-Jul-18</td>
      <td>Black</td>
      <td>Item has never worked. Out of box it is broken...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>1</td>
      <td>11-Jul-18</td>
      <td>Black</td>
      <td>Great product but returning for new Alexa Dot....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>434</th>
      <td>1</td>
      <td>9-Jul-18</td>
      <td>Black</td>
      <td>&amp;#34;NEVER BUY CERTIFIED AND REFURBISHED ECHO ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>470</th>
      <td>1</td>
      <td>1-Jul-18</td>
      <td>White</td>
      <td>This item did not work. Certified refurbished ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>473</th>
      <td>2</td>
      <td>29-Jun-18</td>
      <td>White</td>
      <td>None</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2688</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Weak sound. Compared to the Google Home Mini t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2696</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Echo Dot responds to us when we aren't even ta...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2697</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>NOT CONNECTED TO MY PHONE PLAYLIST :(</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2716</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>The only negative we have on this product is t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2740</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I didn’t order it</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2745</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>The product sounded the same as the emoji spea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2812</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I am quite disappointed by this product.There ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Nope. Still a lot to be improved. For most of ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2842</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I reached out to Amazon, because the device wa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I didn't like that almost everytime i asked Al...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2866</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>The volume is very low</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2876</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>2892</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Cheap and cheap sound.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>For the price, the product is nice quality and...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Used twice not working!!!!!!!</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2932</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>This device does not interact with my home fil...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2940</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Not all that happy. The speaker isn’t great an...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2945</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>When you think about it this really doesn’t do...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2962</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>This worked well for about 6 months but then s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2964</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Ask it to play Motown radio on Pandora and it ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2979</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>3010</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Sound is terrible. Cannot pair with echo to pl...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I am having real difficulty working with the E...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3024</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I was really happy with my original echo so i ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3039</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Weak sound. Compared to the Google Home Mini t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3047</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Echo Dot responds to us when we aren't even ta...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3048</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>NOT CONNECTED TO MY PHONE PLAYLIST :(</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3067</th>
      <td>2</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>The only negative we have on this product is t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3091</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I didn’t order it</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3096</th>
      <td>1</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>The product sounded the same as the emoji spea...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>257 rows × 5 columns</p>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;feedback&#39;</span><span class="p">],</span> <span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Count&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x10f3db470&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;rating&#39;</span><span class="p">],</span> <span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Count&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[12]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x113b41e48&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
SVKT/wf8zBpSDpfZoAAAAABJRU5ErkJggg==
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;rating&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">bins</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x113c2cd30&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">30</span><span class="p">,</span><span class="mi">15</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;variation&#39;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s1">&#39;rating&#39;</span><span class="p">,</span>
            <span class="n">data</span><span class="o">=</span><span class="n">alexa_df</span><span class="p">,</span> <span class="n">palette</span><span class="o">=</span><span class="s1">&#39;deep&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x113d6ce80&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
PzT0BAAAAGATHksIAAAAAABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBni
FgAAAAAAAGWIWwAAAAAAAJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4B
AAAAAABQhrgFAAAAAABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAA
AAAAAGWIWwAAAAAAAJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAA
AABQhrgFAAAAAABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAA
AGWIWwAAAAAAAJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQ
hrgFAAAAAABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAAAGWI
WwAAAAAAAJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQhrgF
AAAAAABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAAAGWIWwAA
AAAAAJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQhrgFAAAA
AABAGeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAAAGWIWwAAAAAA
AJQhbgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQhrgFAAAAAABA
GeIWAAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAAAGWIWwAAAAAAAJQh
bgEAAAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQhrgFAAAAAABAGeIW
AAAAAAAAZYhbAAAAAAAAlCFuAQAAAAAAUIa4BQAAAAAAQBniFgAAAAAAAGWIWwAAAAAAAJQhbgEA
AAAAAFCGuAUAAAAAAEAZ4hYAAAAAAABliFsAAAAAAACUIW4BAAAAAABQhrgFAAAAAABAGeIWAAAA
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
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMItAAAAAAAA
hiHcAgAAAAAAYBjCLQAAAAAAAIYh3AIAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGNsablXVnavq
HVX1rqp6+HaOCwAAAAAAgJ1v28KtqjomyV8k+eEkJye5d1WdvF3jAwAAAAAAYOfbzjO3bp3kXa21
97TWvpjk7CR32cbxAQAAAAAAsMNtZ7h1nSTvm3v+/v4aAAAAAAAAbEq11rZnwFV3T/JDrbX79+c/
k+TWrbUHrvjc6UlO709vkuQd2zJBR8YJST56pCeCNSmj5aZ8lp8yWm7KZ/kpo+WnjJab8ll+ymj5
KaPlpnyWnzJabspn+Smj5aeMlttOLJ/rt9ZOXO9Dx27jBLw/yXXnnn9zkg+s/FBr7WlJnraN03HE
VNX5rbVTj/R0sJgyWm7KZ/kpo+WmfJafMlp+ymi5KZ/lp4yWnzJabspn+Smj5aZ8lp8yWn7KaLkd
zeWznZclPC/JjarqBlX1NUnuleQftnF8AAAAAAAA7HDbduZWa+3LVfWrSV6W5Jgkz2itXbZd4wMA
AAAAAGDn287LEqa19pIkL9nOcSy5HXm5xR1GGS035bP8lNFyUz7LTxktP2W03JTP8lNGy08ZLTfl
s/yU0XJTPstPGS0/ZbTcjtryqdbakZ4GAAAAAAAA2JDtvOcWAAAAAAAAbKmhw62q+oaqOruq3l1V
b62ql1TVjavqtKr6pyM9fUlSVXuq6oQFr19SVbv73+3WGMZJVXXpBsf3e1V1p0OZ5u1UVU+qql+f
e/6yqnr63PMnVNWD1xnG5Ycw/tMWzeuqul9VfWSuTJ7VX193nlbVT1TVw9d4/35V9eebne6DsXL+
HMq4q+qUqvqRueePrKqHHOo0rhjHg6vq7b0+XFRVT6yqK27lOPp4zqyq986V7xnrfH7VurvK59Ys
+01M529X1WVVdXGfzu/eouHuqqpT13j/flX1TVsxrjXGcVJVfW6uDHZX1desM00bWnZ7+3/NrZva
7VVV31xVL66qf+3rsD9Za17076y7buuf+dTc/H1lf/0BVXXfdb57alX96aGM/2hXVV/p8/2iqrpg
tr45mPX4KsNcs+6yMRvd/li0jFfV06vq5P74t7Z/ise2DHWhqm5TVW/u0/G2qnrkJsd7UlX9j818
dxkdRF1Y2OZvZX1QTvtb5rqzHfsBIxml7hzt5bTSktSp+f3QC6rqtnOv320z0zC6JaxPR6SMakGf
5iaH9X019SPsrqrrVNXzt3p6Vxnnyv6iLe0bmRvurqp6R+3bx71bf/0NhzDMF/Zhvav2339+eVX9
wYrPnlJVb1tlGD9WVRf29uWtVfWL/fWfnC2b/fmafYq1Bf2FS9bWXVRV76yqZ1XVdTbwvR29b7XE
7d2OKadhw62qqiQvTLKrtXbD1trJSX4rybW3YNjbei+yOXdorZ3S/zbdKM9U1TGttUe01l65FRO3
Td6QZNbIXiHJCUluPvf+7ZK8fhvHf9ps/As8d65M7pskG5mnrbV/aK09dgunc1mckuRH1v3UBlXV
MSuePyDJDya5TWvt25PcKsmHk1xlve9u0kPnyndhJ/5GVdWxW1n2fSP6x5J8V2vtO5LcKcn7tmLY
G3C/JNsabnXvniuDU1prXzyUgdXkCq21H2mtfXKrJnI79fXX3yd5UWvtRklunOS4JL+/RaN47dz8
vVOStNb+srX2rLW+1Fo7v7W2ZujLuj7X5/t3JvnNJH+w3hc4bDay/bHwwIrW2v1ba2/tT5d2w36J
LENdeGaS01trpyT5tiR/t8nhnJRk+NBkziFvi29xfVBO+9tJdWen2al1Z6dbhjqV9P3QJA9P8tQj
NA3LZNnqU3KYy2gb+jTvk+SP+vL+H621LQnl1ukf3a+/aJv7xe4zt4/7/D6+A/r2Ntpv1Fq7ay/v
+2du/znJA5Pcc8XH75Xkb1eM54qZ7nH04719uUWSXf3tn0yyN9w6TP20y9TWfWeSmyS5MMk5tc5B
vNn5+1bL2t7tmHIaNtxKcockX2qt/eXshdba7tbaa/vT46rq+TWdEXJWX3Gkqh5RVedV1aVV9bS5
13dV1WOq6l+S/FpVXbsn+Rf1v9mC+OD+3UtXJK8vqqq39CMlTt/MD6qq46rqVT1lv6Sq7jL39rFV
9cyazuZ4flVdtX9nT/9Nr0ty95o7sqSqblVVb+jTf25VXX0z07XFXp994dLNk1ya5L+q6mur6kpJ
bpbkwnXmRZK9ZxHsWlDOe6qfdVPT2Qi7quqkJA9I8qB+RMP3bWSCV8zTPVX1qLnpuml/fe+RFlV1
9758XFRVr5kb1DdV1UtrOlPjDw96zm2Bqjqxql7Q68B5VfU9/fVb92Xlwv7/Jr1h+70k9+zza7aC
P7nPz/fU3NlPVfXTfTnbXVVPnW1UVNXlNR2p8uYkt10xSb+d5JdmoURr7Yuttce21j692ndXq79V
dcOqumBuOm5UVW85iHnylKo6v9fdR614+6H9N51bVd/aP39mTWeXnZPkcSvKftV24yB8Y5KPtta+
0OfHR1trH+jDXqvtelyfxnfOluuqukpNR4FdXFXPTQ8Mq+qY/hsu7cvwg/ryfWqSs3r5XaWqfqAv
D5dU1TN6/VyrDlytf+68/r0D6uwaZXDA8jf39nV7vXlHVf1u//xJNR2h+uQkF/TPzNf5+/bffVFV
Pfsgy+BwuGOSz7fW/jpJWmtfSfKgJD9fVVftv++1fR7vPepqXk3t+4VV9S0bGWHNHcW7xjKz90ih
qrp97Tt67cLat/5Ydd3Kqq6R5BMrX1yrfKvqYbXvLNbHrvjeFWraDvg/h2Had6INbX9k8fbjrpq2
Jx6b5Cq9bpzV31t1/cdeR6oufH2SDyZTOzvbIezt4bOr6tU1bZP9z/56VdXj59aPs+2exyb5vl6+
D9r0XFgeG60LyeGpD8ppsaWqO92i/YAD9pH7tJzRHz+pql7dH/9AVf3Nwc2KpTBE3emO5nJayzJs
m70mybeuMg0H9F/0x4u2yUe3bPVp3uEqo4V9movWdbWgD6yq7p/kHkke0V87qfrZOjXtX/5d9X6B
ms74PLW/t/eqP1V1t6o6sz9e2eexof6i2r9v5Po19eld3P9fb27Yf9qH8546hDPjZtPf58s5VfW3
SS7pr21qGWitvSPJJ2v/K+jcI8nZKz569STHJvlY/94XWmvv6O3HTyR5fB/3Desg+mmr6ker6o21
gasIreGIt3Vt8qQk/5nkh/tw7t3HcWlVPa6/dkD93YGWtr3bMeXUWhvyL8kZSZ604L3TknwqyTdn
CvDemOR7+3vXmvvcszOl7MmUsD957r3nJvn1/viYJMcnuWWmhvJqmY6yvyzJLeaHm6kD+dIkX9ef
70lywirTuKcPa3eSN/fXjk1yjf74hCTvSlKZjkRsSb6nv/eMJA+ZG87D5oZ7ZpK7JfmaJO9Jcqv+
+jWSHHuky21umq+X5BczhU2PznS0x/ckec1a86I/v3wD5bx3vmfqtN/VHz9yNu9Wma77JflIL5Pd
SX5ufp7ODfeB/fEvJ3n63Hf/vD++JMl1+uNrzr3/nr4cXTnJvyW57jbN36/M/YbdSf59btr+dm4e
XS/J21YuH5nOFnrByt81N//ekORKvVw+lulI95sl+cckV+yfe3KS+/bHLck9VpnOqyf5xDq/Zb/v
ZnH9PSfJKf3xY2ZltGJYZyZ579x8+fYVdfeYTO3Ad8yV9W/3x/dN8k9zw/mnJMesUvYHtBsHWXbH
9Wl7Z5+Ht9/Ab9+V5An98Y8keWV//OAkz+iPvyPJlzPVhVsmecXcsK45N5xT++MrZzpj7Mb9+bPm
fteerF4HHpPkp2fD7L/hait+30lJPjdXBn+xgeXvg0m+Lvva1lP7cL6a6Yy/+XZldgTMO7Kv/l9r
tXl9JP+yYP2VaYPmO5JcNcmV+2s3SnJ+f3xaX/Zul+QtSa63yjBOy9QuzubxbBl+ZPatNxYtM6dl
33L+j9m3zjkuU5s8G/YBba6/vfN/1v6+vc+rW84t+5f2x4vK94czta9XnV92e3ndJslzZuXpb9Pl
sydrbH+stYxn/zby8rlhLlz/Hc1/y1AXkjwi0479C3uZz8b1yCQXZVqvnJBpffdNSX4qySsyrb+v
nWn76Rvn28ad8rdeXeifOSz1QTkNV3dW2w9YdR+5j/N5/buvTXJu//zvJvnFIz2vd3jdOarLaQnr
1JnZ159w9+zr/5l/fU9W7784YJv8SM/TLSybZapPh72Msnaf5lrrukXzY35a55fvhyR5an/8ben9
AqvMt7slOXNuWPN9HhvtL9r7vM+Xn+2Pfz7TVUtmw35en/6Tk7xr7vu7F8yPXZn28Wf7uF83P/19
vnwmyQ0OdhnIKtsPSR46K5tMdf28Bd99eqYrDz0n05lzV1hZFvPPs6Cfdjbfktw1Uzv8tZuoT0vV
1s299sdJfiPTNty/Jzmx/+ZXJ/nJlcvhTv3LkrZ3O6WcRj5zaz3nttbe31r7aqYKflJ//Q79SIVL
Mh09P38q4HPnHt8xyVOSvUdCfSrJ9yZ5YWvtM621yzNdVmp29s8ZVXVRkjcluW6mhmI9s8sSzo4I
qCSPqaqLk7wyyXWy75Tk97XWZqcp/k2fltWme+YmST7YWjuv/4ZPt9a+vIFpOhxmqfXtMlXWN849
n12eca15MW9ROW/W/GUJ/3rBZ/6+/3/LgvG9PsmZNR1hOp+Yv6q19qnW2ueTvDXJ9Q9xWhf53Nxv
OCXTjs/MnZL8eVXtTvIPSa7RjxQ5Psnz+tE9T8r+9WKlf27TUSkfzbQiv3aSH8i0w3ReH/YPJJmd
UfKVJC9YZTiVKbyanlT9UD8KYM/cESQrv7uo/j49yc/1IxTumRWnjM+ZvyzhJf21e9R05teFfXgn
z33+OXP/5886e16bzrZZabV2Y8N6u3LLJKdnClqfW1X362+v1Xattkx+f6a2Iq21i5Nc3F9/T5Jv
qao/q6o7J/n0KpNykyTvba29sz9/Zh/eWuP7wSQP7+W/K1NAdr1Vhj1/WcJf6a+ttfy9orX2sdba
5/p4Z23fv7XW3rTK8O+Y5Pl9+Uxr7eOrfOZI22/ZX+X1Kyb5q17Wz8v+y+TNsu/yB/++YPjzlyVc
dKnDjbRjT6zpSN5rzq0/trrN3Wlm7e9Nk9w5ybNmR1vNWVS+d0ry1621zyYHLLtPzbRTslWXrjxa
bWT742CX8bXWf0ezI14XWmu/l6nz6eWZLlf30rm3X9xa+1xfV5yT5NaZ1i/P6evvDyX5l0yXS96J
NlIXksNQH5TTAZa97qy2H7BoH/ktSW7Z9zW+kGk5O7W/99qMaZS6c7SX07wjXqe6x/eyPT3JLxzE
9C/aJt8JlqY+dctURmut6w52fnxv+llHrbVLs69fYD3zfR4H0180c9vs65d5dvbvw3xRa+2rbTrj
dG8/X++/WmT+soQfW+X9c1tr7+2PD3X7/Owkd6vpEnL3yr5+of201u7fh31uphDxGesMd61+2jtk
Chd+tLV2wFlXG7Asbd1Ks2m4VaZA+CP9N5+V/fuYdrpla+9WGrqcDte9pbbDZZmS70W+MPf4K5ku
63flTCnmqa2199V009Urz33uM+uMc9XLL1XVaZkag9u21j5b0+nJV17ts+u4T6Z09JattS9V1Z65
4azsCJ1/vtp0L+o8XQaz641+e6YzMd6X5H9l6mSfrQzWmhfzDijn/vjL2XfZzc2UxVpm45wf316t
tQf0U5h/NMnuqjplxfcWfvcwuEKm5fRz8y9W1Z8lOae1dteaLt+4a41hrPY7KskzW2u/ucrnP79a
ENRa+3RVfaaqbtBae29r7WVJXlbTZdG+ZuV316m/L8h0dOGrk7xlwcbOAarqBpk2Qm7VWvtETafh
zy8vbcHj9dqKTeu/d1eSXX2j42er6uys3XYtWiYPaAP67/zOJD+U5FcynWL/8ys+tt6l5lYbXyX5
qTadxn+wHp3Fy9+itm9RGSxz2zdzWaaj8faqqmtkOjDi3UkeluRDSb4zU539/NxHP5ip7G+R5AOH
MA3rtWOPrap/znQ00Ztq3w1wl6EdG0JrbXY5iRNXvPWgrF6+ay27b8gUcD+hTQdIsDkb2f442GV8
rfUfObJ1obX27iRPqaq/SvKRqvq62VsrP5r11307yUbqQnKY6oNyWt2S1p1F+wGrDWO2D/dzffwX
Z+q8u2GSt603DUtqlLpztJfTqo7wttlDW79P0AKr9l+stk3eWnv7BsY3gqWqTzn8ZbRWn+Za67rN
zI9F5pfvlf1m8/vba+2vb9T8uOZ/w1at1+en95C2z3ufy54kt8+0377y9hrzn70kySU13Q7hvZnO
wlpkrTblPZlCiRsnOf/gp3q/aVqm/dBbJHlVxr4t0lZYtvZupaHLaciJ7l6d5Er97Jgke69devs1
vjNrrD9aVcdl7XDsVUl+qQ/3mN7x+JokP1nTNWuvln2njB6f6fJqn63p/jO32eRvOj7Jh/sG5h2y
/5k916uqWYN67ySvW2dYb890j6db9d9w9Vr7RpCH0+uT/FiSj/cjUT6e6TJmt82UXidrz4uN2JMp
vU7270T+r0yXw9s2VXXD1tqbW2uPSPLRTB3Wy+LlSX519mQueDs+yX/0x/eb+/xG59erMh3Z8vV9
uNeqqo2U2R9k2iG7Zv9eZXEYubD+9pXsyzKdNbXojLvVXCPTRtCnqura6deYnXPPuf9vzPpWazc2
rKZrV8+f9XlKpktYHkzbNfOaTCFxqurbMl3uLn0j5wqttRck+Z0k39U/P1/Wb09yUvX7jCX5mUxH
i63lZUkeODs6qKpusYFpnFm0/CXJf+vL01Uy3Zh1zRttZiqDe8x27qvqWgcxHYfLq5Jctarum0zL
SpInZLoMxGczzY8P9qN1fib7nwH6yUzB+WP6gRXbordjl7TWHpdp4/qm2zWunapvDxyTfh32OYvK
9+Xp913r359fdv9vkpdkOmJyWdblI9rI9sdGfKmmm0gnm1//HTWOVF2o6Z4Fsw6TG2XaQfxkf36X
qrpyX1ecluS8TOvNe/b194mZjlI8N4dh2/EI2Kq6kBxifVBOiy1p3VnNon3k2XsP6f9fm+lSPLtb
a8t+INIio9Sd1RxN5bSqJd8225NV+i92+Db50tSnDdqTrS2jtfo0F63rNuN1mQ5oTVWdnKlzfeZD
VXWzms5Quusaw9hMf9EbMp31lEz9Euv1YW6lrVgGnpPpLLV3t9bev/LNqjpuxf74rO8mWTxf1uqn
/bck/z3TGVcbOTNuoWVo62pyRqbLab40yZuT3L6qTuj9H/fOvj6m+fq7Uy1le7dTymnYcKtvaN01
U8fnu6vqskzXl154JHtr7ZNJ/irTtaZflGkHaZFfy5ROX5LpVP2bt9YuyHRtynMzFfjTW2sXZloA
jq3pEnqPznRpws04K8mpVXV+psZ//miPt2U6g+PiJNdKv/TZIq21L2bqkP+zmi6X+Ips/RlMm3VJ
put/v2nFa59q/VJiWXtebMSjkvxJVb0204b+zD8muWtNl7/7vtW/esgeX/3me5k2Si7apvFsxhmZ
5uvFVfXWTDsuSfKHSf6gql6f/TvSz8l0Q+Ldte+G3Qfop5P/7yQv78voKzI1jut5SqbLTr65f+/1
mS4PeOHKD26g/p6V6WiTl29gvLNhXtTHdVmmoyVWBidXqqo3Z2oPNnJz8gPajY1OS3dckmdW1Vv7
/Dg5ySMPsu2aeUqmm1FenOlMoNnG8HUynRW2O1N7NjvC48wkf9lfr0xHbz6v/5avJvnLrO3RmU5z
v7gv+4/ewDTOLFr+kmkj+NmZTsl+QWttzaOYWmuXJfn9JP/S274nHsR0HBZz66+7V9W/Zro/2eeT
/Fb/yJMztfdvynTk1mdWfP9DSX48yV/U/je63Uq/XtMNRC/KdJ+0/7dN49lpZjdZ3Z3pksE/u8qZ
q6uWb2vtpZkuF3t+//5D5r/UWntikguSPLvvgHLwNrL9sRFPy9TWnXUI67+dbhnqws8keUcfxrMz
Xc5mNg3nJvnnTMvCo1trH8h035qLM223vTrTPW3/s7/25ZpusL2RbYERbFVdSA69Piin/S173TnA
GvvIyRSUfGOSN/btl89n7EvdjVJ3DnCUldO8ZahTG7Go/2Inb5MvU33aiC0to3X6NBet6zbjyUlO
7PPiN/pwZ7dPeHime2u9OtMVQhbZTH/RGZluHXFxpvbq19ab0F7PDtkWLQPPy9Sfc/aC9yvJw6pq
1g4/KvuCv7OTPLSqLqyqG85N15r9tG26Cs59MvXD7P3eBi1LW/f4/tvemekSd3dorX2xtfbBTH1P
52Rari9orb24f2dv/T3I3zySZWvvdlQ51Q47GAc4SlXVQ5Ic31r7nSM9LQDAcqrp0r6Xt9b+6EhP
C6/3sqgAAAMUSURBVIspJwA4dP3siyu21j7fA5NXJblxD1oAhufSNsDwquqFma4Lf8cjPS0AAAAA
S+CqSc7plxOrJL8k2AJ2EmduAQAAAAAAMAz3bAAAAAAAAGAYwi0AAAAAAACGIdwCAAAAAABgGMIt
AACAw6yqXlJV11znM7+14vkbtneqAAAAxlCttSM9DQAAAEeFqqpM+2Ff3cBnL2+tHXcYJgsAAGAo
ztwCAAA4SFX1uKr65bnnj6yq362qV1XVBVV1SVXdpb93UlW9raqenOSCJNetqj1VdUJ//0VV9Zaq
uqyqTu+vPTbJVapqd1Wd1V+7vP+vqnp8VV3ax3PP/vppVbWrqp5fVW+vqrN6mAYAALCjOHMLAADg
IFXVLZL8cWvt9v35W5PcOcknW2uf7sHVm5LcKMn1k7wnye1aa2/qn9+T5NTW2ker6lqttY9X1VWS
nJfk9q21j608c2v2vKp+KskD+vhO6N/57iQ3SfLiJDdP8oEkr0/y0Nba67Z9hgAAABxGxx7pCQAA
ABhNa+3Cqvr6qvqmJCcm+USSDyZ5UlV9f5KvJrlOkmv3r/zbLNhaxRlVddf++LqZArGPrTH6703y
nNbaV5J8qKr+Jcmtknw6ybmttfcnSVXtTnJSEuEWAACwowi3AAAANuf5Se6W5BuSnJ3kPpmCrlu2
1r7Uz866cv/sZ1YbQFWdluROSW7bWvtsVe2a+84ia11q8Atzj78S+3wAAMAO5J5bAAAAm3N2kntl
Crien+T4JB/uwdYdMl2OcD3HJ/lED7ZumuQ2c+99qaquuMp3XpPknlV1TFWdmOT7k5x7KD8EAABg
JMItAACATWitXZbk6kn+o7X2wSRnJTm1qs7PdBbX2zcwmJcmObaqLk7y6Ez36Zp5WpKLq+qsFd95
YZKLk1yU5NVJHtZa+89D+jEAAAADqdbakZ4GAAAAAAAA2BBnbgEAAAAAADAM4RYAAAAAAADDEG4B
AAAAAAAwDOEWAAAAAAAAwxBuAQAAAAAAMAzhFgAAAAAAAMMQbgEAAAAAADAM4RYAAAAAAADD+P+v
5wshyDOhCwAAAABJRU5ErkJggg==
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
<h3 id="Data-Clean-/-Prep">Data Clean / Prep<a class="anchor-link" href="#Data-Clean-/-Prep">&#182;</a></h3><p>For this example we will be only looking at the text to see if the review is positive or negative. Binary Classification.</p>
<p>Drop: Date, Rating</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>31-Jul-18</td>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Music</td>
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
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;date&#39;</span><span class="p">,</span> <span class="s1">&#39;rating&#39;</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>



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
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Charcoal Fabric</td>
      <td>Music</td>
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
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">variation_dummies</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;variation&#39;</span><span class="p">],</span> <span class="n">drop_first</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">variation_dummies</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[20]:</div>



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
      <th>Black  Dot</th>
      <th>Black  Plus</th>
      <th>Black  Show</th>
      <th>Black  Spot</th>
      <th>Charcoal Fabric</th>
      <th>Configuration: Fire TV Stick</th>
      <th>Heather Gray Fabric</th>
      <th>Oak Finish</th>
      <th>Sandstone Fabric</th>
      <th>Walnut Finish</th>
      <th>White</th>
      <th>White  Dot</th>
      <th>White  Plus</th>
      <th>White  Show</th>
      <th>White  Spot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>3120</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3123</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 15 columns</p>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;variation&#39;</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[23]:</div>



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
      <th>verified_reviews</th>
      <th>feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love my Echo!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Loved it!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Music</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I received the echo as a gift. I needed anothe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Without having a cellphone, I cannot use many ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I think this is the 5th one I've purchased. I'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>looks great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Love it! I’ve listened to songs I haven’t hear...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I sent it to my 85 year old Dad, and he talks ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>I love it! Learning knew things with it eveyda...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I purchased this for my mother who is having k...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Love, Love, Love!!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Just what I expected....</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>I love it, wife hates it.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Really happy with this purchase.  Great speake...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>We have only been using Alexa for a couple of ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>We love the size of the 2nd generation echo. S...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>I liked the original Echo. This is the same bu...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Love the Echo and how good the music sounds pl...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>We love Alexa! We use her to play music, play ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Have only had it set up for a few days. Still ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>I love it. It plays my sleep sounds immediatel...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>I got a second unit for the bedroom, I was exp...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Amazing product</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>I love my Echo. It's easy to operate, loads of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sounds great!! Love them!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fun item to play with and get used to using.  ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Just like the other one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3120</th>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>I like the hands free operation vs the Tap. We...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>I dislike that it confuses my requests all the...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3123</th>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>Love my Alexa! Actually have 3 throughout the ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>This product is easy to use and very entertain...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3126</th>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>works great but speaker is not the good for mu...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>Outstanding product - easy to use.  works great</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>We have six of these throughout our home and t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>Use the product for music and it’s great!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>Easy to set-up and to use.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>It works great!!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>I like having more Alexa devices in my house a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>PHENOMENAL</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>I loved it does exactly what it says</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>I used it to control my smart home devices. Wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>Very convenient</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>Este producto llegó y a la semana se quedó sin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>Easy to set up Ready to use in minutes.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>Barry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3141</th>
      <td></td>
      <td>1</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>My three year old loves it.  Good for doing ba...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>Awesome device wish I bought one ages ago.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>love it</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>Perfect for kids, adults and everyone in betwe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>Listening to music, searching locations, check...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>I do love these things, i have them running my...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>Only complaint I have is that the sound qualit...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>Good</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 2 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">alexa_df</span><span class="p">,</span> <span class="n">variation_dummies</span><span class="p">],</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>



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
      <th>verified_reviews</th>
      <th>feedback</th>
      <th>Black  Dot</th>
      <th>Black  Plus</th>
      <th>Black  Show</th>
      <th>Black  Spot</th>
      <th>Charcoal Fabric</th>
      <th>Configuration: Fire TV Stick</th>
      <th>Heather Gray Fabric</th>
      <th>Oak Finish</th>
      <th>Sandstone Fabric</th>
      <th>Walnut Finish</th>
      <th>White</th>
      <th>White  Dot</th>
      <th>White  Plus</th>
      <th>White  Show</th>
      <th>White  Spot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love my Echo!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Loved it!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Music</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I received the echo as a gift. I needed anothe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Without having a cellphone, I cannot use many ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I think this is the 5th one I've purchased. I'...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>looks great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Love it! I’ve listened to songs I haven’t hear...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I sent it to my 85 year old Dad, and he talks ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>I love it! Learning knew things with it eveyda...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I purchased this for my mother who is having k...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Love, Love, Love!!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Just what I expected....</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>I love it, wife hates it.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Really happy with this purchase.  Great speake...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>We have only been using Alexa for a couple of ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>We love the size of the 2nd generation echo. S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>I liked the original Echo. This is the same bu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Love the Echo and how good the music sounds pl...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>We love Alexa! We use her to play music, play ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Have only had it set up for a few days. Still ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>I love it. It plays my sleep sounds immediatel...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>I got a second unit for the bedroom, I was exp...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Amazing product</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>I love my Echo. It's easy to operate, loads of...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Sounds great!! Love them!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fun item to play with and get used to using.  ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Just like the other one</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>3120</th>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>I like the hands free operation vs the Tap. We...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>I dislike that it confuses my requests all the...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3123</th>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>Love my Alexa! Actually have 3 throughout the ...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>This product is easy to use and very entertain...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3126</th>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>works great but speaker is not the good for mu...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>Outstanding product - easy to use.  works great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>We have six of these throughout our home and t...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>Use the product for music and it’s great!</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>Easy to set-up and to use.</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>It works great!!</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>I like having more Alexa devices in my house a...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>PHENOMENAL</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>I loved it does exactly what it says</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>I used it to control my smart home devices. Wo...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>Very convenient</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>Este producto llegó y a la semana se quedó sin...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>Easy to set up Ready to use in minutes.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>Barry</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3141</th>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>My three year old loves it.  Good for doing ba...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>Awesome device wish I bought one ages ago.</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>love it</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>Perfect for kids, adults and everyone in betwe...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>Listening to music, searching locations, check...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>I do love these things, i have them running my...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>Only complaint I have is that the sound qualit...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>Good</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 17 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="n">vectorizer</span> <span class="o">=</span> <span class="n">CountVectorizer</span><span class="p">()</span>
<span class="n">alexa_countcetorizer</span> <span class="o">=</span> <span class="n">vectorizer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;verified_reviews&#39;</span><span class="p">])</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_countcetorizer</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(3150, 4044)</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">vectorizer</span><span class="o">.</span><span class="n">get_feature_names</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[&#39;00&#39;, &#39;000&#39;, &#39;07&#39;, &#39;10&#39;, &#39;100&#39;, &#39;100x&#39;, &#39;11&#39;, &#39;1100sf&#39;, &#39;12&#39;, &#39;129&#39;, &#39;12am&#39;, &#39;15&#39;, &#39;150&#39;, &#39;18&#39;, &#39;19&#39;, &#39;1964&#39;, &#39;1990&#39;, &#39;1gb&#39;, &#39;1rst&#39;, &#39;1st&#39;, &#39;20&#39;, &#39;200&#39;, &#39;2000&#39;, &#39;2017&#39;, &#39;229&#39;, &#39;23&#39;, &#39;24&#39;, &#39;25&#39;, &#39;29&#39;, &#39;2nd&#39;, &#39;2package&#39;, &#39;30&#39;, &#39;300&#39;, &#39;30pm&#39;, &#39;34&#39;, &#39;360&#39;, &#39;39&#39;, &#39;3rd&#39;, &#39;3x&#39;, &#39;3xs&#39;, &#39;40&#39;, &#39;45&#39;, &#39;48&#39;, &#39;4am&#39;, &#39;4ghz&#39;, &#39;4k&#39;, &#39;4th&#39;, &#39;50&#39;, &#39;54&#39;, &#39;5am&#39;, &#39;5ghz&#39;, &#39;5th&#39;, &#39;600&#39;, &#39;62&#39;, &#39;672&#39;, &#39;6th&#39;, &#39;70&#39;, &#39;75&#39;, &#39;79&#39;, &#39;80&#39;, &#39;80s&#39;, &#39;81&#39;, &#39;83&#39;, &#39;85&#39;, &#39;88&#39;, &#39;888&#39;, &#39;8gb&#39;, &#39;90&#39;, &#39;91&#39;, &#39;911&#39;, &#39;99&#39;, &#39;_specifically_&#39;, &#39;a1&#39;, &#39;a19&#39;, &#39;abay&#39;, &#39;abc&#39;, &#39;abd&#39;, &#39;abilities&#39;, &#39;ability&#39;, &#39;able&#39;, &#39;abode&#39;, &#39;about&#39;, &#39;above&#39;, &#39;absolutely&#39;, &#39;absolutly&#39;, &#39;ac&#39;, &#39;accent&#39;, &#39;acceptable&#39;, &#39;accepting&#39;, &#39;access&#39;, &#39;accessable&#39;, &#39;accessible&#39;, &#39;accessing&#39;, &#39;accessories&#39;, &#39;accesss&#39;, &#39;accident&#39;, &#39;accidentally&#39;, &#39;accompanying&#39;, &#39;accomplish&#39;, &#39;accomplished&#39;, &#39;according&#39;, &#39;accordingly&#39;, &#39;account&#39;, &#39;accounts&#39;, &#39;accuracy&#39;, &#39;accurate&#39;, &#39;accurately&#39;, &#39;accustom&#39;, &#39;acknowledge&#39;, &#39;acoustical&#39;, &#39;across&#39;, &#39;act&#39;, &#39;acting&#39;, &#39;action&#39;, &#39;actions&#39;, &#39;activate&#39;, &#39;activated&#39;, &#39;activates&#39;, &#39;activating&#39;, &#39;activation&#39;, &#39;actively&#39;, &#39;activities&#39;, &#39;acts&#39;, &#39;actually&#39;, &#39;ad&#39;, &#39;adapted&#39;, &#39;adapter&#39;, &#39;adapting&#39;, &#39;add&#39;, &#39;added&#39;, &#39;addict&#39;, &#39;addicted&#39;, &#39;addicts&#39;, &#39;adding&#39;, &#39;addition&#39;, &#39;additional&#39;, &#39;additionally&#39;, &#39;addons&#39;, &#39;addressed&#39;, &#39;addresses&#39;, &#39;adds&#39;, &#39;adept&#39;, &#39;adequate&#39;, &#39;adjacent&#39;, &#39;adjust&#39;, &#39;adjusting&#39;, &#39;adjustment&#39;, &#39;adjusts&#39;, &#39;admit&#39;, &#39;adopters&#39;, &#39;adorable&#39;, &#39;ads&#39;, &#39;adults&#39;, &#39;advance&#39;, &#39;advanced&#39;, &#39;advantage&#39;, &#39;advantages&#39;, &#39;advertise&#39;, &#39;advertised&#39;, &#39;advertisement&#39;, &#39;advertising&#39;, &#39;advice&#39;, &#39;advise&#39;, &#39;advised&#39;, &#39;aesthetic&#39;, &#39;af&#39;, &#39;affirm&#39;, &#39;affirmations&#39;, &#39;afford&#39;, &#39;affordable&#39;, &#39;afraid&#39;, &#39;after&#39;, &#39;afternoon&#39;, &#39;afterwards&#39;, &#39;again&#39;, &#39;age&#39;, &#39;agent&#39;, &#39;ages&#39;, &#39;ago&#39;, &#39;agree&#39;, &#39;agreement&#39;, &#39;ahead&#39;, &#39;ai&#39;, &#39;aide&#39;, &#39;aint&#39;, &#39;air&#39;, &#39;aka&#39;, &#39;al&#39;, &#39;alabama&#39;, &#39;alarm&#39;, &#39;alarms&#39;, &#39;albeit&#39;, &#39;alcohol&#39;, &#39;alert&#39;, &#39;alerts&#39;, &#39;alex&#39;, &#39;alexa&#39;, &#39;alexas&#39;, &#39;alexi&#39;, &#39;alexia&#39;, &#39;alexis&#39;, &#39;alexus&#39;, &#39;algo&#39;, &#39;alive&#39;, &#39;all&#39;, &#39;alleviate&#39;, &#39;allow&#39;, &#39;allowed&#39;, &#39;allowing&#39;, &#39;allows&#39;, &#39;allrecipes&#39;, &#39;almost&#39;, &#39;alone&#39;, &#39;along&#39;, &#39;alongside&#39;, &#39;alot&#39;, &#39;alots&#39;, &#39;aloud&#39;, &#39;alread&#39;, &#39;already&#39;, &#39;alright&#39;, &#39;also&#39;, &#39;altering&#39;, &#39;alternative&#39;, &#39;alternatives&#39;, &#39;although&#39;, &#39;always&#39;, &#39;am&#39;, &#39;amaonmazing&#39;, &#39;amaxing&#39;, &#39;amaze&#39;, &#39;amazed&#39;, &#39;amazin&#39;, &#39;amazing&#39;, &#39;amazingly&#39;, &#39;amazon&#39;, &#39;amazonia&#39;, &#39;amazons&#39;, &#39;ambient&#39;, &#39;american&#39;, &#39;americans&#39;, &#39;among&#39;, &#39;amount&#39;, &#39;amounts&#39;, &#39;amozon&#39;, &#39;amplifier&#39;, &#39;amused&#39;, &#39;amusing&#39;, &#39;an&#39;, &#39;analog&#39;, &#39;and&#39;, &#39;android&#39;, &#39;ands&#39;, &#39;angle&#39;, &#39;annoying&#39;, &#39;another&#39;, &#39;answer&#39;, &#39;answered&#39;, &#39;answering&#39;, &#39;answers&#39;, &#39;ant&#39;, &#39;anti&#39;, &#39;anticipate&#39;, &#39;anticipated&#39;, &#39;any&#39;, &#39;anybody&#39;, &#39;anyhow&#39;, &#39;anylist&#39;, &#39;anymore&#39;, &#39;anyone&#39;, &#39;anypod&#39;, &#39;anything&#39;, &#39;anytime&#39;, &#39;anyway&#39;, &#39;anyways&#39;, &#39;anywhere&#39;, &#39;apartment&#39;, &#39;app&#39;, &#39;apparent&#39;, &#39;apparently&#39;, &#39;appealing&#39;, &#39;appear&#39;, &#39;appears&#39;, &#39;apple&#39;, &#39;appliance&#39;, &#39;appliances&#39;, &#39;application&#39;, &#39;applications&#39;, &#39;appointments&#39;, &#39;appreciated&#39;, &#39;apprehensive&#39;, &#39;approaching&#39;, &#39;appropriate&#39;, &#39;approximately&#39;, &#39;apps&#39;, &#39;are&#39;, &#39;area&#39;, &#39;areas&#39;, &#39;aren&#39;, &#39;arent&#39;, &#39;argue&#39;, &#39;argument&#39;, &#39;arguments&#39;, &#39;arises&#39;, &#39;arlo&#39;, &#39;arm&#39;, &#39;around&#39;, &#39;array&#39;, &#39;arrive&#39;, &#39;arrived&#39;, &#39;arriving&#39;, &#39;articles&#39;, &#39;artist&#39;, &#39;artists&#39;, &#39;as&#39;, &#39;asap&#39;, &#39;ase&#39;, &#39;ask&#39;, &#39;asked&#39;, &#39;askes&#39;, &#39;asking&#39;, &#39;asleep&#39;, &#39;aspect&#39;, &#39;aspects&#39;, &#39;ass&#39;, &#39;assigned&#39;, &#39;assist&#39;, &#39;assistance&#39;, &#39;assistant&#39;, &#39;assume&#39;, &#39;assumed&#39;, &#39;assuming&#39;, &#39;assumption&#39;, &#39;at&#39;, &#39;atención&#39;, &#39;atmosphere&#39;, &#39;atrás&#39;, &#39;attach&#39;, &#39;attached&#39;, &#39;attachment&#39;, &#39;attempt&#39;, &#39;attempted&#39;, &#39;attempting&#39;, &#39;attention&#39;, &#39;attractive&#39;, &#39;audible&#39;, &#39;audibles&#39;, &#39;audio&#39;, &#39;audioapple&#39;, &#39;audiobook&#39;, &#39;audiobooks&#39;, &#39;audiophile&#39;, &#39;august&#39;, &#39;aunt&#39;, &#39;auto&#39;, &#39;automatic&#39;, &#39;automatically&#39;, &#39;automation&#39;, &#39;aux&#39;, &#39;auxiliary&#39;, &#39;av&#39;, &#39;avail&#39;, &#39;availability&#39;, &#39;available&#39;, &#39;avoid&#39;, &#39;awake&#39;, &#39;aware&#39;, &#39;away&#39;, &#39;awesome&#39;, &#39;awful&#39;, &#39;awhile&#39;, &#39;awkward&#39;, &#39;awsome&#39;, &#39;b073sqyxtw&#39;, &#39;baby&#39;, &#39;back&#39;, &#39;background&#39;, &#39;backgrounds&#39;, &#39;backyard&#39;, &#39;bad&#39;, &#39;baffle&#39;, &#39;baffled&#39;, &#39;ball&#39;, &#39;ban&#39;, &#39;band&#39;, &#39;bandwagon&#39;, &#39;bandwidth&#39;, &#39;bang&#39;, &#39;bar&#39;, &#39;bare&#39;, &#39;barely&#39;, &#39;bargain&#39;, &#39;bark&#39;, &#39;barn&#39;, &#39;barret&#39;, &#39;barry&#39;, &#39;base&#39;, &#39;baseball&#39;, &#39;based&#39;, &#39;basement&#39;, &#39;basic&#39;, &#39;basically&#39;, &#39;bass&#39;, &#39;bathroom&#39;, &#39;bathrooms&#39;, &#39;batman&#39;, &#39;batteries&#39;, &#39;battery&#39;, &#39;bc&#39;, &#39;be&#39;, &#39;beam&#39;, &#39;beat&#39;, &#39;beautiful&#39;, &#39;beautifully&#39;, &#39;beauty&#39;, &#39;became&#39;, &#39;because&#39;, &#39;becausse&#39;, &#39;become&#39;, &#39;becomes&#39;, &#39;becoming&#39;, &#39;bed&#39;, &#39;bedroom&#39;, &#39;bedrooms&#39;, &#39;bedside&#39;, &#39;bedtime&#39;, &#39;beefy&#39;, &#39;been&#39;, &#39;before&#39;, &#39;begin&#39;, &#39;beginners&#39;, &#39;beginning&#39;, &#39;begun&#39;, &#39;behaved&#39;, &#39;behind&#39;, &#39;being&#39;, &#39;believe&#39;, &#39;believer&#39;, &#39;bells&#39;, &#39;belong&#39;, &#39;below&#39;, &#39;benefit&#39;, &#39;benefits&#39;, &#39;beside&#39;, &#39;besides&#39;, &#39;best&#39;, &#39;bet&#39;, &#39;beta&#39;, &#39;better&#39;, &#39;bettter&#39;, &#39;between&#39;, &#39;beyond&#39;, &#39;bezel&#39;, &#39;bezos&#39;, &#39;bf&#39;, &#39;bff&#39;, &#39;bible&#39;, &#39;big&#39;, &#39;bigger&#39;, &#39;biggest&#39;, &#39;bill&#39;, &#39;billboard&#39;, &#39;bills&#39;, &#39;bing&#39;, &#39;birth&#39;, &#39;birthday&#39;, &#39;bit&#39;, &#39;bizarre&#39;, &#39;black&#39;, &#39;blanket&#39;, &#39;blast&#39;, &#39;blasting&#39;, &#39;blessing&#39;, &#39;blind&#39;, &#39;blink&#39;, &#39;blinks&#39;, &#39;blocking&#39;, &#39;bloods&#39;, &#39;bloomberg&#39;, &#39;blown&#39;, &#39;blows&#39;, &#39;blue&#39;, &#39;blueprints&#39;, &#39;bluetooth&#39;, &#39;blurring&#39;, &#39;board&#39;, &#39;boat&#39;, &#39;bob&#39;, &#39;body&#39;, &#39;bolt&#39;, &#39;bonkers&#39;, &#39;bonus&#39;, &#39;book&#39;, &#39;books&#39;, &#39;boom&#39;, &#39;boombox&#39;, &#39;booming&#39;, &#39;boost&#39;, &#39;boring&#39;, &#39;born&#39;, &#39;bose&#39;, &#39;boss&#39;, &#39;bot&#39;, &#39;both&#39;, &#39;bother&#39;, &#39;bothered&#39;, &#39;bothers&#39;, &#39;bothersome&#39;, &#39;bottom&#39;, &#39;bough&#39;, &#39;bought&#39;, &#39;box&#39;, &#39;boyfriend&#39;, &#39;brainer&#39;, &#39;brand&#39;, &#39;brandnew&#39;, &#39;brands&#39;, &#39;bread&#39;, &#39;break&#39;, &#39;breakfast&#39;, &#39;breeze&#39;, &#39;bridge&#39;, &#39;brief&#39;, &#39;briefing&#39;, &#39;briefings&#39;, &#39;briefs&#39;, &#39;bright&#39;, &#39;brightness&#39;, &#39;bring&#39;, &#39;bringing&#39;, &#39;british&#39;, &#39;broadway&#39;, &#39;broke&#39;, &#39;broken&#39;, &#39;brought&#39;, &#39;bt&#39;, &#39;bucks&#39;, &#39;buddies&#39;, &#39;budget&#39;, &#39;buffer&#39;, &#39;buffering&#39;, &#39;buffet&#39;, &#39;bug&#39;, &#39;bugging&#39;, &#39;bugs&#39;, &#39;build&#39;, &#39;building&#39;, &#39;built&#39;, &#39;bulb&#39;, &#39;bulbs&#39;, &#39;buld&#39;, &#39;bulky&#39;, &#39;bummed&#39;, &#39;bunch&#39;, &#39;bundle&#39;, &#39;bundled&#39;, &#39;burns&#39;, &#39;business&#39;, &#39;busy&#39;, &#39;but&#39;, &#39;buts&#39;, &#39;button&#39;, &#39;buttons&#39;, &#39;buy&#39;, &#39;buyer&#39;, &#39;buyers&#39;, &#39;buying&#39;, &#39;buys&#39;, &#39;buzzing&#39;, &#39;by&#39;, &#39;bye&#39;, &#39;cable&#39;, &#39;calendar&#39;, &#39;calendars&#39;, &#39;call&#39;, &#39;called&#39;, &#39;calling&#39;, &#39;calls&#39;, &#39;calm&#39;, &#39;calmer&#39;, &#39;cam&#39;, &#39;cambiar&#39;, &#39;came&#39;, &#39;camelot&#39;, &#39;camera&#39;, &#39;cameras&#39;, &#39;campus&#39;, &#39;cams&#39;, &#39;can&#39;, &#39;canary&#39;, &#39;cancel&#39;, &#39;canceling&#39;, &#39;cancelled&#39;, &#39;cancels&#39;, &#39;cannot&#39;, &#39;cant&#39;, &#39;capabilities&#39;, &#39;capability&#39;, &#39;capable&#39;, &#39;capacity&#39;, &#39;capasity&#39;, &#39;car&#39;, &#39;card&#39;, &#39;cards&#39;, &#39;cardsrotate&#39;, &#39;care&#39;, &#39;carefully&#39;, &#39;careless&#39;, &#39;carful&#39;, &#39;carolina&#39;, &#39;carrier&#39;, &#39;carry&#39;, &#39;cart&#39;, &#39;cartoons&#39;, &#39;case&#39;, &#39;cases&#39;, &#39;cat&#39;, &#39;catch&#39;, &#39;catches&#39;, &#39;categories&#39;, &#39;cause&#39;, &#39;caused&#39;, &#39;cave&#39;, &#39;cbs&#39;, &#39;cd&#39;, &#39;ceases&#39;, &#39;ceiling&#39;, &#39;ceilings&#39;, &#39;celebs&#39;, &#39;cell&#39;, &#39;cellphone&#39;, &#39;cent&#39;, &#39;center&#39;, &#39;certain&#39;, &#39;certainly&#39;, &#39;certified&#39;, &#39;chachki&#39;, &#39;chair&#39;, &#39;chalk&#39;, &#39;challenge&#39;, &#39;challenged&#39;, &#39;champ&#39;, &#39;chance&#39;, &#39;change&#39;, &#39;changed&#39;, &#39;changer&#39;, &#39;changes&#39;, &#39;changing&#39;, &#39;channel&#39;, &#39;channels&#39;, &#39;characteristics&#39;, &#39;charge&#39;, &#39;chargeable&#39;, &#39;charger&#39;, &#39;charging&#39;, &#39;charlotte&#39;, &#39;charm&#39;, &#39;charmed&#39;, &#39;chart&#39;, &#39;chat&#39;, &#39;chatting&#39;, &#39;cheap&#39;, &#39;cheaper&#39;, &#39;cheapest&#39;, &#39;check&#39;, &#39;checked&#39;, &#39;checking&#39;, &#39;child&#39;, &#39;childhood&#39;, &#39;children&#39;, &#39;chocolate&#39;, &#39;choice&#39;, &#39;choices&#39;, &#39;choose&#39;, &#39;choosing&#39;, &#39;choppy&#39;, &#39;chores&#39;, &#39;chose&#39;, &#39;chosen&#39;, &#39;christmas&#39;, &#39;chromebook&#39;, &#39;chromecast&#39;, &#39;circle&#39;, &#39;citizens&#39;, &#39;city&#39;, &#39;clapper&#39;, &#39;clarity&#39;, &#39;classes&#39;, &#39;classic&#39;, &#39;classical&#39;, &#39;classroom&#39;, &#39;clean&#39;, &#39;cleaner&#39;, &#39;cleaning&#39;, &#39;clear&#39;, &#39;clearer&#39;, &#39;clearly&#39;, &#39;click&#39;, &#39;clients&#39;, &#39;clips&#39;, &#39;clock&#39;, &#39;clockhome&#39;, &#39;clocking&#39;, &#39;clocks&#39;, &#39;clone&#39;, &#39;close&#39;, &#39;closed&#39;, &#39;closer&#39;, &#39;clothes&#39;, &#39;cloud&#39;, &#39;clue&#39;, &#39;cm_cr_ryp_prd_ttl_sol_18&#39;, &#39;cnn&#39;, &#39;co&#39;, &#39;coast&#39;, &#39;codes&#39;, &#39;coffee&#39;, &#39;cohesive&#39;, &#39;collection&#39;, &#39;collections&#39;, &#39;collectors&#39;, &#39;college&#39;, &#39;colon&#39;, &#39;color&#39;, &#39;colors&#39;, &#39;com&#39;, &#39;comands&#39;, &#39;combination&#39;, &#39;combine&#39;, &#39;combined&#39;, &#39;come&#39;, &#39;comeletely&#39;, &#39;comes&#39;, &#39;comfort&#39;, &#39;comfortable&#39;, &#39;comforting&#39;, &#39;coming&#39;, &#39;command&#39;, &#39;commanded&#39;, &#39;commands&#39;, &#39;comment&#39;, &#39;comments&#39;, &#39;commercials&#39;, &#39;commodity&#39;, &#39;common&#39;, &#39;communicate&#39;, &#39;communicated&#39;, &#39;communicating&#39;, &#39;communication&#39;, &#39;community&#39;, &#39;commute&#39;, &#39;como&#39;, &#39;compacity&#39;, &#39;compact&#39;, &#39;companion&#39;, &#39;company&#39;, &#39;comparable&#39;, &#39;compare&#39;, &#39;compared&#39;, &#39;compatible&#39;, &#39;competition&#39;, &#39;complacated&#39;, &#39;complain&#39;, &#39;complained&#39;, &#39;complaining&#39;, &#39;complaint&#39;, &#39;complaints&#39;, &#39;complete&#39;, &#39;completed&#39;, &#39;completely&#39;, &#39;complicated&#39;, &#39;compliment&#39;, &#39;compliments&#39;, &#39;components&#39;, &#39;compound&#39;, &#39;computer&#39;, &#39;computers&#39;, &#39;con&#39;, &#39;concept&#39;, &#39;concern&#39;, &#39;concerned&#39;, &#39;concerning&#39;, &#39;concerns&#39;, &#39;concise&#39;, &#39;condition&#39;, &#39;conditioning&#39;, &#39;conditions&#39;, &#39;conectado&#39;, &#39;conferencing&#39;, &#39;confident&#39;, &#39;configure&#39;, &#39;configured&#39;, &#39;conflict&#39;, &#39;confused&#39;, &#39;confuses&#39;, &#39;confusing&#39;, &#39;confusion&#39;, &#39;connect&#39;, &#39;connected&#39;, &#39;connecting&#39;, &#39;connection&#39;, &#39;connectivity&#39;, &#39;connects&#39;, &#39;cons&#39;, &#39;conscious&#39;, &#39;consider&#39;, &#39;considering&#39;, &#39;consistent&#39;, &#39;consistently&#39;, &#39;conspiracy&#39;, &#39;constant&#39;, &#39;constantly&#39;, &#39;constructed&#39;, &#39;consulting&#39;, &#39;consumer&#39;, &#39;contact&#39;, &#39;contacted&#39;, &#39;contacts&#39;, &#39;contains&#39;, &#39;content&#39;, &#39;contents&#39;, &#39;continous&#39;, &#39;continually&#39;, &#39;continue&#39;, &#39;continues&#39;, &#39;continuous&#39;, &#39;continuously&#39;, &#39;control&#39;, &#39;controll&#39;, &#39;controllable&#39;, &#39;controlled&#39;, &#39;controller&#39;, &#39;controlling&#39;, &#39;controls&#39;, &#39;convenience&#39;, &#39;convenient&#39;, &#39;conversation&#39;, &#39;conversations&#39;, &#39;convert&#39;, &#39;convinced&#39;, &#39;cook&#39;, &#39;cooking&#39;, &#39;cool&#39;, &#39;cooler&#39;, &#39;coolest&#39;, &#39;coop&#39;, &#39;coordinator&#39;, &#39;cord&#39;, &#39;cordless&#39;, &#39;cordthank&#39;, &#39;core&#39;, &#39;correct&#39;, &#39;corrected&#39;, &#39;correctly&#39;, &#39;corresponds&#39;, &#39;cortna&#39;, &#39;cost&#39;, &#39;costs&#39;, &#39;cotton&#39;, &#39;couch&#39;, &#39;could&#39;, &#39;couldn&#39;, &#39;counter&#39;, &#39;counters&#39;, &#39;countless&#39;, &#39;countries&#39;, &#39;country&#39;, &#39;county&#39;, &#39;couple&#39;, &#39;course&#39;, &#39;cousin&#39;, &#39;cousins&#39;, &#39;cover&#39;, &#39;covered&#39;, &#39;covers&#39;, &#39;cozi&#39;, &#39;cpr&#39;, &#39;cracked&#39;, &#39;crackle&#39;, &#39;crackling&#39;, &#39;crap&#39;, &#39;crappy&#39;, &#39;crashed&#39;, &#39;crashes&#39;, &#39;crashing&#39;, &#39;crazy&#39;, &#39;creapy&#39;, &#39;create&#39;, &#39;created&#39;, &#39;credited&#39;, &#39;creepy&#39;, &#39;crib&#39;, &#39;crisp&#39;, &#39;critically&#39;, &#39;cropping&#39;, &#39;cross&#39;, &#39;crunchyroll&#39;, &#39;csi&#39;, &#39;cualquier&#39;, &#39;cue&#39;, &#39;cumbersome&#39;, &#39;cups&#39;, &#39;current&#39;, &#39;currently&#39;, &#39;cursed&#39;, &#39;curve&#39;, &#39;custom&#39;, &#39;customer&#39;, &#39;customers&#39;, &#39;customizable&#39;, &#39;customization&#39;, &#39;customize&#39;, &#39;cut&#39;, &#39;cute&#39;, &#39;cutie&#39;, &#39;cutting&#39;, &#39;cycle&#39;, &#39;cycled&#39;, &#39;cycles&#39;, &#39;cylinder&#39;, &#39;cylindercal&#39;, &#39;dad&#39;, &#39;daily&#39;, &#39;damage&#39;, &#39;dance&#39;, &#39;dancing&#39;, &#39;dare&#39;, &#39;dark&#39;, &#39;darn&#39;, &#39;dash&#39;, &#39;data&#39;, &#39;date&#39;, &#39;dated&#39;, &#39;dates&#39;, &#39;daughter&#39;, &#39;day&#39;, &#39;days&#39;, &#39;de&#39;, &#39;deactivate&#39;, &#39;dead&#39;, &#39;deaf&#39;, &#39;deal&#39;, &#39;deals&#39;, &#39;debating&#39;, &#39;dec&#39;, &#39;decent&#39;, &#39;decide&#39;, &#39;decided&#39;, &#39;decides&#39;, &#39;decision&#39;, &#39;deck&#39;, &#39;decor&#39;, &#39;decorated&#39;, &#39;decrease&#39;, &#39;dedicated&#39;, &#39;deep&#39;, &#39;deeper&#39;, &#39;default&#39;, &#39;defeats&#39;, &#39;defective&#39;, &#39;defence&#39;, &#39;defently&#39;, &#39;definately&#39;, &#39;define&#39;, &#39;definitely&#39;, &#39;definition&#39;, &#39;definitively&#39;, &#39;defuser&#39;, &#39;degree&#39;, &#39;degrees&#39;, &#39;del&#39;, &#39;delay&#39;, &#39;delete&#39;, &#39;deliver&#39;, &#39;delivered&#39;, &#39;delivers&#39;, &#39;delivery&#39;, &#39;demand&#39;, &#39;dementia&#39;, &#39;den&#39;, &#39;denon&#39;, &#39;dense&#39;, &#39;dented&#39;, &#39;department&#39;, &#39;dependable&#39;, &#39;dependence&#39;, &#39;depending&#39;, &#39;deployed&#39;, &#39;depreciates&#39;, &#39;depth&#39;, &#39;described&#39;, &#39;description&#39;, &#39;design&#39;, &#39;designed&#39;, &#39;designers&#39;, &#39;desired&#39;, &#39;desk&#39;, &#39;desktop&#39;, &#39;despite&#39;, &#39;detailed&#39;, &#39;details&#39;, &#39;detect&#39;, &#39;determined&#39;, &#39;developed&#39;, &#39;developers&#39;, &#39;development&#39;, &#39;device&#39;, &#39;deviceoverall&#39;, &#39;devices&#39;, &#39;devise&#39;, &#39;devises&#39;, &#39;dhiw&#39;, &#39;diagnostics&#39;, &#39;dial&#39;, &#39;dictionary&#39;, &#39;did&#39;, &#39;didn&#39;, &#39;didnt&#39;, &#39;died&#39;, &#39;dies&#39;, &#39;differ&#39;, &#39;difference&#39;, &#39;differences&#39;, &#39;different&#39;, &#39;differentiate&#39;, &#39;difficult&#39;, &#39;difficulty&#39;, &#39;dig&#39;, &#39;digital&#39;, &#39;digitol&#39;, &#39;digs&#39;, &#39;dim&#39;, &#39;dimat&#39;, &#39;dimension&#39;, &#39;dimmer&#39;, &#39;dimming&#39;, &#39;dims&#39;, &#39;dining&#39;, &#39;dinner&#39;, &#39;dinosaurs&#39;, &#39;direct&#39;, &#39;direction&#39;, &#39;directions&#39;, &#39;directly&#39;, &#39;directtv&#39;, &#39;directv&#39;, &#39;disability&#39;, &#39;disable&#39;, &#39;disabled&#39;, &#39;disagree&#39;, &#39;disappoint&#39;, &#39;disappointed&#39;, &#39;disappointing&#39;, &#39;disappointment&#39;, &#39;disappointments&#39;, &#39;disarm&#39;, &#39;disaster&#39;, &#39;disconcerting&#39;, &#39;disconnect&#39;, &#39;disconnected&#39;, &#39;disconnecting&#39;, &#39;disconnections&#39;, &#39;disconnects&#39;, &#39;discount&#39;, &#39;discounts&#39;, &#39;discourage&#39;, &#39;discover&#39;, &#39;discovered&#39;, &#39;discoveredthat&#39;, &#39;discovering&#39;, &#39;discovery&#39;, &#39;dish&#39;, &#39;dislike&#39;, &#39;dislikes&#39;, &#39;dismiss&#39;, &#39;dismissed&#39;, &#39;display&#39;, &#39;displayed&#39;, &#39;displaying&#39;, &#39;displays&#39;, &#39;disposable&#39;, &#39;dissatisfaction&#39;, &#39;distance&#39;, &#39;distorted&#39;, &#39;distracting&#39;, &#39;distraction&#39;, &#39;disturbing&#39;, &#39;ditch&#39;, &#39;ditched&#39;, &#39;diversity&#39;, &#39;divertido&#39;, &#39;dj&#39;, &#39;do&#39;, &#39;docking&#39;, &#39;doctor&#39;, &#39;documentation&#39;, &#39;dodging&#39;, &#39;does&#39;, &#39;doesn&#39;, &#39;doesnt&#39;, &#39;dog&#39;, &#39;dogs&#39;, &#39;doing&#39;, &#39;dollar&#39;, &#39;dollars&#39;, &#39;domain&#39;, &#39;don&#39;, &#39;done&#39;, &#39;dont&#39;, &#39;door&#39;, &#39;doorbell&#39;, &#39;doors&#39;, &#39;dorm&#39;, &#39;dot&#39;, &#39;dots&#39;, &#39;doubtful&#39;, &#39;down&#39;, &#39;downfall&#39;, &#39;download&#39;, &#39;downloaded&#39;, &#39;downloading&#39;, &#39;downright&#39;, &#39;downside&#39;, &#39;downstairs&#39;, &#39;dp&#39;, &#39;drag&#39;, &#39;draw&#39;, &#39;drawback&#39;, &#39;drawing&#39;, &#39;dressed&#39;, &#39;drive&#39;, &#39;driven&#39;, &#39;drivers&#39;, &#39;drives&#39;, &#39;driving&#39;, &#39;drop&#39;, &#39;dropped&#39;, &#39;dropping&#39;, &#39;drops&#39;, &#39;dryer&#39;, &#39;due&#39;, &#39;dumb&#39;, &#39;dumber&#39;, &#39;dunce&#39;, &#39;dunno&#39;, &#39;during&#39;, &#39;dust&#39;, &#39;duty&#39;, &#39;dying&#39;, &#39;dylan&#39;, &#39;each&#39;, &#39;ear&#39;, &#39;early&#39;, &#39;earn&#39;, &#39;ease&#39;, &#39;easier&#39;, &#39;easily&#39;, &#39;east&#39;, &#39;easy&#39;, &#39;eavesdropping&#39;, &#39;echo&#39;, &#39;echoes&#39;, &#39;echoplus&#39;, &#39;echos&#39;, &#39;eco&#39;, &#39;ecobee3&#39;, &#39;ecoo&#39;, &#39;ecosystem&#39;, &#39;ed&#39;, &#39;edge&#39;, &#39;edit&#39;, &#39;educated&#39;, &#39;educational&#39;, &#39;eeaanh&#39;, &#39;effected&#39;, &#39;effective&#39;, &#39;effects&#39;, &#39;efficiency&#39;, &#39;efficient&#39;, &#39;effort&#39;, &#39;effortless&#39;, &#39;efforts&#39;, &#39;eg&#39;, &#39;eh&#39;, &#39;either&#39;, &#39;el&#39;, &#39;elderly&#39;, &#39;electeonically&#39;, &#39;electrician&#39;, &#39;electricity&#39;, &#39;electronic&#39;, &#39;electronically&#39;, &#39;electronics&#39;, &#39;elegant&#39;, &#39;element&#39;, &#39;eliminate&#39;, &#39;else&#39;, &#39;elsewhere&#39;, &#39;em&#39;, &#39;email&#39;, &#39;embarrassed&#39;, &#39;emergency&#39;, &#39;emoji&#39;, &#39;employees&#39;, &#39;en&#39;, &#39;enable&#39;, &#39;enabled&#39;, &#39;enables&#39;, &#39;encyclopedias&#39;, &#39;end&#39;, &#39;ended&#39;, &#39;endless&#39;, &#39;ends&#39;, &#39;engage&#39;, &#39;engagement&#39;, &#39;engaging&#39;, &#39;engine&#39;, &#39;engineers&#39;, &#39;english&#39;, &#39;enhanced&#39;, &#39;enjoy&#39;, &#39;enjoyable&#39;, &#39;enjoyed&#39;, &#39;enjoying&#39;, &#39;enjoyment&#39;, &#39;enjoys&#39;, &#39;enough&#39;, &#39;enrolment&#39;, &#39;enter&#39;, &#39;entering&#39;, &#39;enters&#39;, &#39;entertained&#39;, &#39;entertaining&#39;, &#39;entertainment&#39;, &#39;entire&#39;, &#39;entirely&#39;, &#39;entry&#39;, &#39;eq&#39;, &#39;equal&#39;, &#39;equalized&#39;, &#39;equalizer&#39;, &#39;equipment&#39;, &#39;equipo&#39;, &#39;error&#39;, &#39;errors&#39;, &#39;es&#39;, &#39;escencia&#39;, &#39;esp&#39;, &#39;espanol&#39;, &#39;español&#39;, &#39;especially&#39;, &#39;essential&#39;, &#39;essentially&#39;, &#39;esta&#39;, &#39;estar&#39;, &#39;este&#39;, &#39;estudio&#39;, &#39;estés&#39;, &#39;etc&#39;, &#39;etekcity&#39;, &#39;ethernet&#39;, &#39;evaluate&#39;, &#39;even&#39;, &#39;evening&#39;, &#39;event&#39;, &#39;events&#39;, &#39;eventually&#39;, &#39;ever&#39;, &#39;every&#39;, &#39;everybody&#39;, &#39;everyday&#39;, &#39;everyone&#39;, &#39;everything&#39;, &#39;everytime&#39;, &#39;everywhere&#39;, &#39;eveyday&#39;, &#39;evolve&#39;, &#39;evrything&#39;, &#39;ex&#39;, &#39;exact&#39;, &#39;exactly&#39;, &#39;example&#39;, &#39;examples&#39;, &#39;exasperation&#39;, &#39;exceeded&#39;, &#39;exceeds&#39;, &#39;excelente&#39;, &#39;excellent&#39;, &#39;excellently&#39;, &#39;except&#39;, &#39;exception&#39;, &#39;exceptionally&#39;, &#39;excessive&#39;, &#39;exchange&#39;, &#39;exchanges&#39;, &#39;exchanging&#39;, &#39;excited&#39;, &#39;excitement&#39;, &#39;excuses&#39;, &#39;exho&#39;, &#39;existence&#39;, &#39;existent&#39;, &#39;existing&#39;, &#39;expanded&#39;, &#39;expanding&#39;, &#39;expect&#39;, &#39;expectation&#39;, &#39;expectations&#39;, &#39;expected&#39;, &#39;expecting&#39;, &#39;expensive&#39;, &#39;experience&#39;, &#39;experienced&#39;, &#39;experiences&#39;, &#39;expert&#39;, &#39;expired&#39;, &#39;expires&#39;, &#39;explanation&#39;, &#39;explicit&#39;, &#39;explore&#39;, &#39;explored&#39;, &#39;exploring&#39;, &#39;extend&#39;, &#39;extended&#39;, &#39;extender&#39;, &#39;extends&#39;, &#39;extension&#39;, &#39;extent&#39;, &#39;external&#39;, &#39;extra&#39;, &#39;extras&#39;, &#39;extremely&#39;, &#39;extrimelly&#39;, &#39;eye&#39;, &#39;eyes&#39;, &#39;fabric&#39;, &#39;fabulous&#39;, &#39;face&#39;, &#39;facebook&#39;, &#39;faces&#39;, &#39;facetime&#39;, &#39;fact&#39;, &#39;factor&#39;, &#39;factory&#39;, &#39;facts&#39;, &#39;fail&#39;, &#39;failed&#39;, &#39;failing&#39;, &#39;fails&#39;, &#39;fair&#39;, &#39;fairly&#39;, &#39;fairness&#39;, &#39;fall&#39;, &#39;falling&#39;, &#39;falls&#39;, &#39;false&#39;, &#39;familiar&#39;, &#39;family&#39;, &#39;fan&#39;, &#39;fanatic&#39;, &#39;fans&#39;, &#39;fantastic&#39;, &#39;far&#39;, &#39;farther&#39;, &#39;fascinating&#39;, &#39;fashioned&#39;, &#39;fast&#39;, &#39;faster&#39;, &#39;fat&#39;, &#39;father&#39;, &#39;fathers&#39;, &#39;fault&#39;, &#39;faulty&#39;, &#39;favorite&#39;, &#39;favorites&#39;, &#39;featues&#39;, &#39;feature&#39;, &#39;featured&#39;, &#39;features&#39;, &#39;fee&#39;, &#39;feed&#39;, &#39;feedback&#39;, &#39;feeds&#39;, &#39;feee&#39;, &#39;feel&#39;, &#39;feeling&#39;, &#39;feels&#39;, &#39;fees&#39;, &#39;feet&#39;, &#39;fell&#39;, &#39;felt&#39;, &#39;fencing&#39;, &#39;few&#39;, &#39;fi&#39;, &#39;fiances&#39;, &#39;fidelity&#39;, &#39;figure&#39;, &#39;figured&#39;, &#39;figuring&#39;, &#39;fill&#39;, &#39;filled&#39;, &#39;filling&#39;, &#39;fills&#39;, &#39;final&#39;, &#39;finally&#39;, &#39;find&#39;, &#39;finding&#39;, &#39;finds&#39;, &#39;fine&#39;, &#39;fingertips&#39;, &#39;finicky&#39;, &#39;finish&#39;, &#39;fios&#39;, &#39;fire&#39;, &#39;firestick&#39;, &#39;firmare&#39;, &#39;firmware&#39;, &#39;first&#39;, &#39;fit&#39;, &#39;fits&#39;, &#39;five&#39;, &#39;fix&#39;, &#39;fixed&#39;, &#39;fixes&#39;, &#39;fixing&#39;, &#39;fixture&#39;, &#39;fixtures&#39;, &#39;flash&#39;, &#39;flashes&#39;, &#39;flat&#39;, &#39;flaw&#39;, &#39;flawless&#39;, &#39;flawlessly&#39;, &#39;flaws&#39;, &#39;fledged&#39;, &#39;flexibility&#39;, &#39;flexible&#39;, &#39;flickering&#39;, &#39;floating&#39;, &#39;floor&#39;, &#39;floored&#39;, &#39;fm&#39;, &#39;folks&#39;, &#39;follow&#39;, &#39;followed&#39;, &#39;font&#39;, &#39;foot&#39;, &#39;football&#39;, &#39;footprint&#39;, &#39;for&#39;, &#39;force&#39;, &#39;forces&#39;, &#39;forecast&#39;, &#39;forecasts&#39;, &#39;forever&#39;, &#39;forget&#39;, &#39;forgot&#39;, &#39;forgotten&#39;, &#39;forjust&#39;, &#39;form&#39;, &#39;forth&#39;, &#39;fortunately&#39;, &#39;forums&#39;, &#39;forward&#39;, &#39;found&#39;, &#39;four&#39;, &#39;fourth&#39;, &#39;free&#39;, &#39;freeze&#39;, &#39;freezes&#39;, &#39;frequently&#39;, &#39;fri&#39;, &#39;friday&#39;, &#39;friend&#39;, &#39;friendly&#39;, &#39;friends&#39;, &#39;from&#39;, &#39;front&#39;, &#39;frustrated&#39;, &#39;frustrating&#39;, &#39;frustration&#39;, &#39;full&#39;, &#39;fuller&#39;, &#39;fully&#39;, &#39;fumble&#39;, &#39;fun&#39;, &#39;funciona&#39;, &#39;funcionamiento&#39;, &#39;funciones&#39;, &#39;function&#39;, &#39;functionalities&#39;, &#39;functionality&#39;, &#39;functions&#39;, &#39;funny&#39;, &#39;further&#39;, &#39;furthermore&#39;, &#39;fussing&#39;, &#39;fussy&#39;, &#39;future&#39;, &#39;fw&#39;, &#39;gadget&#39;, &#39;gadgets&#39;, &#39;gain&#39;, &#39;galaxy&#39;, &#39;game&#39;, &#39;games&#39;, &#39;gameshow&#39;, &#39;gaming&#39;, &#39;gap&#39;, &#39;garage&#39;, &#39;garbage&#39;, &#39;gateway&#39;, &#39;gather&#39;, &#39;gatherings&#39;, &#39;gave&#39;, &#39;gazebo&#39;, &#39;gb&#39;, &#39;ge&#39;, &#39;geared&#39;, &#39;geek&#39;, &#39;geeks&#39;, &#39;gen&#39;, &#39;gen2&#39;, &#39;gender&#39;, &#39;general&#39;, &#39;generally&#39;, &#39;generation&#39;, &#39;genial&#39;, &#39;genre&#39;, &#39;genres&#39;, &#39;geo&#39;, &#39;get&#39;, &#39;gets&#39;, &#39;getting&#39;, &#39;gf&#39;, &#39;ghost&#39;, &#39;gift&#39;, &#39;gifts&#39;, &#39;girlfriend&#39;, &#39;girls&#39;, &#39;give&#39;, &#39;given&#39;, &#39;gives&#39;, &#39;giving&#39;, &#39;gizmo&#39;, &#39;glad&#39;, &#39;glaring&#39;, &#39;glasses&#39;, &#39;glitch&#39;, &#39;glitches&#39;, &#39;glitching&#39;, &#39;glorified&#39;, &#39;glow&#39;, &#39;go&#39;, &#39;god&#39;, &#39;godsend&#39;, &#39;goes&#39;, &#39;going&#39;, &#39;golden&#39;, &#39;gone&#39;, &#39;goo&#39;, &#39;good&#39;, &#39;goodies&#39;, &#39;goodmorning&#39;, &#39;goodness&#39;, &#39;google&#39;, &#39;googled&#39;, &#39;got&#39;, &#39;gotten&#39;, &#39;government&#39;, &#39;grab&#39;, &#39;grace&#39;, &#39;grand&#39;, &#39;grandaughter&#39;, &#39;grandchildren&#39;, &#39;granddaughter&#39;, &#39;grandfather&#39;, &#39;grandkids&#39;, &#39;grandmother&#39;, &#39;grandparent&#39;, &#39;grandparents&#39;, &#39;grands&#39;, &#39;grandson&#39;, &#39;grandsons&#39;, &#39;granite&#39;, &#39;granted&#39;, &#39;graphics&#39;, &#39;gratamente&#39;, &#39;greade&#39;, &#39;great&#39;, &#39;greater&#39;, &#39;greatest&#39;, &#39;greatly&#39;, &#39;green&#39;, &#39;greeting&#39;, &#39;grip&#39;, &#39;gripe&#39;, &#39;grocery&#39;, &#39;groggy&#39;, &#39;ground&#39;, &#39;group&#39;, &#39;groups&#39;, &#39;growing&#39;, &#39;grownups&#39;, &#39;grows&#39;, &#39;guarantee&#39;, &#39;guaranteeing&#39;, &#39;guard&#39;, &#39;guess&#39;, &#39;guest&#39;, &#39;guide&#39;, &#39;guilty&#39;, &#39;guy&#39;, &#39;guys&#39;, &#39;habit&#39;, &#39;habla&#39;, &#39;had&#39;, &#39;hadn&#39;, &#39;haha&#39;, &#39;hahaawesome&#39;, &#39;hahahaha&#39;, &#39;hairs&#39;, &#39;hal&#39;, &#39;half&#39;, &#39;hallway&#39;, &#39;hand&#39;, &#39;handle&#39;, &#39;handled&#39;, &#39;handles&#39;, &#39;hands&#39;, &#39;handy&#39;, &#39;hang&#39;, &#39;happen&#39;, &#39;happened&#39;, &#39;happening&#39;, &#39;happens&#39;, &#39;happier&#39;, &#39;happy&#39;, &#39;hard&#39;, &#39;hardcore&#39;, &#39;harder&#39;, &#39;hardly&#39;, &#39;harmony&#39;, &#39;harvard&#39;, &#39;has&#39;, &#39;hasn&#39;, &#39;hassel&#39;, &#39;hassle&#39;, &#39;hate&#39;, &#39;hated&#39;, &#39;hates&#39;, &#39;hauler&#39;, &#39;have&#39;, &#39;haven&#39;, &#39;havent&#39;, &#39;having&#39;, &#39;haywire&#39;, &#39;hbo&#39;, &#39;hcfe&#39;, &#39;hd&#39;, &#39;hd8&#39;, &#39;hdm1&#39;, &#39;hdmi&#39;, &#39;he&#39;, &#39;headline&#39;, &#39;headphone&#39;, &#39;headphones&#39;, &#39;heads&#39;, &#39;healing&#39;, &#39;hear&#39;, &#39;heard&#39;, &#39;hearing&#39;, &#39;hears&#39;, &#39;heart&#39;, &#39;heaven&#39;, &#39;heavy&#39;, &#39;heck&#39;, &#39;hectic&#39;, &#39;held&#39;, &#39;helful&#39;, &#39;hell&#39;, &#39;help&#39;, &#39;helped&#39;, &#39;helper&#39;, &#39;helpful&#39;, &#39;helping&#39;, &#39;helps&#39;, &#39;hence&#39;, &#39;her&#39;, &#39;here&#39;, &#39;hers&#39;, &#39;herself&#39;, &#39;hes&#39;, &#39;hesitant&#39;, &#39;hesitate&#39;, &#39;hesitated&#39;, &#39;hey&#39;, &#39;hi&#39;, &#39;hiccups&#39;, &#39;hide&#39;, &#39;high&#39;, &#39;higher&#39;, &#39;highest&#39;, &#39;highly&#39;, &#39;him&#39;, &#39;himself&#39;, &#39;hints&#39;, &#39;hire&#39;, &#39;hired&#39;, &#39;hiring&#39;, &#39;his&#39;, &#39;history&#39;, &#39;hit&#39;, &#39;hmm&#39;, &#39;hmmm&#39;, &#39;hmmmm&#39;, &#39;hold&#39;, &#39;holder&#39;, &#39;holding&#39;, &#39;hole&#39;, &#39;holiday&#39;, &#39;holy&#39;, &#39;home&#39;, &#39;homes&#39;, &#39;homescreen&#39;, &#39;homework&#39;, &#39;honest&#39;, &#39;honestly&#39;, &#39;hong&#39;, &#39;hook&#39;, &#39;hooked&#39;, &#39;hope&#39;, &#39;hoped&#39;, &#39;hopefully&#39;, &#39;hoping&#39;, &#39;hora&#39;, &#39;horrible&#39;, &#39;horse&#39;, &#39;hospital&#39;, &#39;hospitals&#39;, &#39;hosting&#39;, &#39;hot&#39;, &#39;hotel&#39;, &#39;hour&#39;, &#39;hours&#39;, &#39;house&#39;, &#39;household&#39;, &#39;houses&#39;, &#39;how&#39;, &#39;however&#39;, &#39;hr&#39;, &#39;https&#39;, &#39;hub&#39;, &#39;hubbed&#39;, &#39;hubby&#39;, &#39;hubs&#39;, &#39;hue&#39;, &#39;huele&#39;, &#39;huge&#39;, &#39;hulu&#39;, &#39;human&#39;, &#39;humour&#39;, &#39;hundred&#39;, &#39;hundreds&#39;, &#39;husband&#39;, &#39;hut&#39;, &#39;hvac&#39;, &#39;hype&#39;, &#39;id&#39;, &#39;idea&#39;, &#39;ideal&#39;, &#39;if&#39;, &#39;ifs&#39;, &#39;ight&#39;, &#39;ignored&#39;, &#39;ignoring&#39;, &#39;iheart&#39;, &#39;iheartradio&#39;, &#39;ihome&#39;, &#39;ii&#39;, &#39;illustrated&#39;, &#39;im&#39;, &#39;image&#39;, &#39;images&#39;, &#39;imagination&#39;, &#39;imagine&#39;, &#39;imagined&#39;, &#39;imhave&#39;, &#39;immediately&#39;, &#39;impaired&#39;, &#39;impede&#39;, &#39;imperfection&#39;, &#39;implementing&#39;, &#39;important&#39;, &#39;importantly&#39;, &#39;impressed&#39;, &#39;impressive&#39;, &#39;improve&#39;, &#39;improved&#39;, &#39;improvement&#39;, &#39;improvements&#39;, &#39;improving&#39;, &#39;impulse&#39;, &#39;imrproved&#39;, &#39;imusic&#39;, &#39;in&#39;, &#39;inability&#39;, &#39;inactivity&#39;, &#39;include&#39;, &#39;included&#39;, &#39;includes&#39;, &#39;including&#39;, &#39;inclusive&#39;, &#39;income&#39;, &#39;inconvenience&#39;, &#39;inconvenient&#39;, &#39;incorporated&#39;, &#39;increase&#39;, &#39;increasing&#39;, &#39;incredible&#39;, &#39;incredibly&#39;, &#39;india&#39;, &#39;indicated&#39;, &#39;indicator&#39;, &#39;indispensable&#39;, &#39;individual&#39;, &#39;individually&#39;, &#39;indoor&#39;, &#39;indundated&#39;, &#39;industry&#39;, &#39;inexpensive&#39;, &#39;inexperience&#39;, &#39;infact&#39;, &#39;inferior&#39;, &#39;info&#39;, &#39;información&#39;, &#39;information&#39;, &#39;informative&#39;, &#39;informed&#39;, &#39;infotainment&#39;, &#39;initial&#39;, &#39;initially&#39;, &#39;initiate&#39;, &#39;inline&#39;, &#39;innovative&#39;, &#39;input&#39;, &#39;insanely&#39;, &#39;insanity&#39;, &#39;insert&#39;, &#39;inside&#39;, &#39;insist&#39;, &#39;inspired&#39;, &#39;install&#39;, &#39;installation&#39;, &#39;installed&#39;, &#39;installing&#39;, &#39;installs&#39;, &#39;instant&#39;, &#39;instantaneous&#39;, &#39;instantly&#39;, &#39;instead&#39;, &#39;instruction&#39;, &#39;instructions&#39;, &#39;integrate&#39;, &#39;integrated&#39;, &#39;integrates&#39;, &#39;integrating&#39;, &#39;integration&#39;, &#39;intelagence&#39;, &#39;inteligente&#39;, &#39;intelligent&#39;, &#39;intend&#39;, &#39;intended&#39;, &#39;intention&#39;, &#39;interact&#39;, &#39;interacting&#39;, &#39;interaction&#39;, &#39;interactions&#39;, &#39;interactive&#39;, &#39;intercom&#39;, &#39;intercoms&#39;, &#39;interest&#39;, &#39;interested&#39;, &#39;interesting&#39;, &#39;interface&#39;, &#39;interfacing&#39;, &#39;interference&#39;, &#39;interferes&#39;, &#39;intermittent&#39;, &#39;intermittently&#39;, &#39;internal&#39;, &#39;international&#39;, &#39;internet&#39;, &#39;interpret&#39;, &#39;interrogated&#39;, &#39;interrupt&#39;, &#39;interruption&#39;, &#39;intimidating&#39;, &#39;into&#39;, &#39;introduce&#39;, &#39;introducing&#39;, &#39;introduction&#39;, &#39;intrusive&#39;, &#39;intuitive&#39;, &#39;invasion&#39;, &#39;invasions&#39;, &#39;invasive&#39;, &#39;invention&#39;, &#39;invest&#39;, &#39;invested&#39;, &#39;investing&#39;, &#39;investment&#39;, &#39;inviting&#39;, &#39;involved&#39;, &#39;involves&#39;, &#39;iot&#39;, &#39;iove&#39;, &#39;ipad&#39;, &#39;ipads&#39;, &#39;ipdates&#39;, &#39;iphone&#39;, &#39;irritated&#39;, &#39;irritating&#39;, &#39;is&#39;, &#39;ise&#39;, &#39;ish&#39;, &#39;island&#39;, &#39;isn&#39;, &#39;isnt&#39;, &#39;isolated&#39;, &#39;issue&#39;, &#39;issues&#39;, &#39;isue&#39;, &#39;it&#39;, &#39;ita&#39;, &#39;italian&#39;, &#39;italy&#39;, &#39;item&#39;, &#39;items&#39;, &#39;its&#39;, &#39;itself&#39;, &#39;itunes&#39;, &#39;iy&#39;, &#39;jack&#39;, &#39;jacuzzi&#39;, &#39;jamming&#39;, &#39;jams&#39;, &#39;jaws&#39;, &#39;jazz&#39;, &#39;jeapordy&#39;, &#39;jeff&#39;, &#39;jeopardy&#39;, &#39;jetsons&#39;, &#39;jimmy&#39;, &#39;job&#39;, &#39;johnny&#39;, &#39;join&#39;, &#39;joke&#39;, &#39;joked&#39;, &#39;jokes&#39;, &#39;journey&#39;, &#39;joy&#39;, &#39;jump&#39;, &#39;jumped&#39;, &#39;jumping&#39;, &#39;june&#39;, &#39;junk&#39;, &#39;just&#39;, &#39;karen&#39;, &#39;kasa&#39;, &#39;keen&#39;, &#39;keep&#39;, &#39;keeper&#39;, &#39;keeping&#39;, &#39;keeps&#39;, &#39;kept&#39;, &#39;key&#39;, &#39;keyboard&#39;, &#39;kick&#39;, &#39;kicking&#39;, &#39;kid&#39;, &#39;kids&#39;, &#39;killer&#39;, &#39;kind&#39;, &#39;kinda&#39;, &#39;kindle&#39;, &#39;kinds&#39;, &#39;king&#39;, &#39;kitchen&#39;, &#39;knee&#39;, &#39;knew&#39;, &#39;knob&#39;, &#39;knock&#39;, &#39;knocked&#39;, &#39;know&#39;, &#39;knowing&#39;, &#39;knowledgable&#39;, &#39;knowledge&#39;, &#39;knowledgeable&#39;, &#39;known&#39;, &#39;knows&#39;, &#39;kodi&#39;, &#39;kong&#39;, &#39;korea&#39;, &#39;kwikset&#39;, &#39;la&#39;, &#39;labeled&#39;, &#39;lack&#39;, &#39;lacking&#39;, &#39;lacks&#39;, &#39;ladies&#39;, &#39;lady&#39;, &#39;lag&#39;, &#39;lagging&#39;, &#39;lags&#39;, &#39;lame&#39;, &#39;lamp&#39;, &#39;lamps&#39;, &#39;land&#39;, &#39;language&#39;, &#39;lapsed&#39;, &#39;laptop&#39;, &#39;large&#39;, &#39;larger&#39;, &#39;las&#39;, &#39;last&#39;, &#39;lastly&#39;, &#39;late&#39;, &#39;lately&#39;, &#39;later&#39;, &#39;lauded&#39;, &#39;laugh&#39;, &#39;laughs&#39;, &#39;laughter&#39;, &#39;laundry&#39;, &#39;law&#39;, &#39;layer&#39;, &#39;laying&#39;, &#39;laziness&#39;, &#39;lazy&#39;, &#39;lcd&#39;, &#39;leaning&#39;, &#39;learn&#39;, &#39;learned&#39;, &#39;learnimg&#39;, &#39;learning&#39;, &#39;learns&#39;, &#39;leary&#39;, &#39;least&#39;, &#39;leave&#39;, &#39;leaves&#39;, &#39;leaving&#39;, &#39;led&#39;, &#39;left&#39;, &#39;leg&#39;, &#39;legally&#39;, &#39;leisure&#39;, &#39;length&#39;, &#39;less&#39;, &#39;lesson&#39;, &#39;let&#39;, &#39;lets&#39;, &#39;level&#39;, &#39;levels&#39;, &#39;lg&#39;, &#39;libraries&#39;, &#39;library&#39;, &#39;life&#39;, &#39;lifetime&#39;, &#39;lifht&#39;, &#39;light&#39;, &#39;lightbulb&#39;, &#39;lightening&#39;, &#39;lighting&#39;, &#39;lightning&#39;, &#39;lights&#39;, &#39;like&#39;, &#39;liked&#39;, &#39;likely&#39;, &#39;likes&#39;, &#39;liking&#39;, &#39;lil&#39;, &#39;lilttle&#39;, &#39;limitations&#39;, &#39;limited&#39;, &#39;line&#39;, &#39;lines&#39;, &#39;link&#39;, &#39;linked&#39;, &#39;linking&#39;, &#39;links&#39;, &#39;list&#39;, &#39;listen&#39;, &#39;listened&#39;, &#39;listening&#39;, &#39;listens&#39;, &#39;lists&#39;, &#39;lit&#39;, &#39;literally&#39;, &#39;literate&#39;, &#39;little&#39;, &#39;live&#39;, &#39;lived&#39;, &#39;lives&#39;, &#39;living&#39;, &#39;livingroom&#39;, &#39;ll&#39;, &#39;llama&#39;, &#39;llegó&#39;, &#39;lm&#39;, &#39;lo&#39;, &#39;load&#39;, &#39;loaded&#39;, &#39;loads&#39;, &#39;local&#39;, &#39;locate&#39;, &#39;located&#39;, &#39;location&#39;, &#39;locations&#39;, &#39;lock&#39;, &#39;locked&#39;, &#39;locks&#39;, &#39;logitech&#39;, &#39;logo&#39;, &#39;logra&#39;, &#39;lol&#39;, &#39;lolol&#39;, &#39;lonely&#39;, &#39;long&#39;, &#39;longer&#39;, &#39;longevity&#39;, &#39;look&#39;, &#39;looked&#39;, &#39;looking&#39;, &#39;looks&#39;, &#39;looooooove&#39;, &#39;loose&#39;, &#39;looses&#39;, &#39;loosing&#39;, &#39;lose&#39;, &#39;loses&#39;, &#39;losing&#39;, &#39;loss&#39;, &#39;lost&#39;, &#39;lot&#39;, &#39;lots&#39;, &#39;loud&#39;, &#39;louder&#39;, &#39;louis&#39;, &#39;lov&#39;, &#39;love&#39;, &#39;loved&#39;, &#39;lovee&#39;, &#39;lover&#39;, &#39;loves&#39;, &#39;loving&#39;, &#39;low&#39;, &#39;lower&#39;, &#39;luck&#39;, &#39;luckily&#39;, &#39;lucky&#39;, &#39;lullaby&#39;, &#39;lurking&#39;, &#39;luv&#39;, &#39;lve&#39;, &#39;lyric&#39;, &#39;lyrical&#39;, &#39;lyrics&#39;, &#39;mac&#39;, &#39;machine&#39;, &#39;machines&#39;, &#39;maddening&#39;, &#39;made&#39;, &#39;madlibs&#39;, &#39;magically&#39;, &#39;mailed&#39;, &#39;main&#39;, &#39;mainly&#39;, &#39;mainstream&#39;, &#39;maintain&#39;, &#39;maintaining&#39;, &#39;majel&#39;, &#39;majes&#39;, &#39;major&#39;, &#39;make&#39;, &#39;makes&#39;, &#39;making&#39;, &#39;makings&#39;, &#39;male&#39;, &#39;malone&#39;, &#39;mama&#39;, &#39;man&#39;, &#39;manage&#39;, &#39;management&#39;, &#39;mandatory&#39;, &#39;maneuver&#39;, &#39;manners&#39;, &#39;manual&#39;, &#39;manually&#39;, &#39;manuals&#39;, &#39;manufacturers&#39;, &#39;many&#39;, &#39;marginal&#39;, &#39;mark&#39;, &#39;marked&#39;, &#39;market&#39;, &#39;marketing&#39;, &#39;marvelous&#39;, &#39;massive&#39;, &#39;match&#39;, &#39;matched&#39;, &#39;material&#39;, &#39;matter&#39;, &#39;maximize&#39;, &#39;may&#39;, &#39;maybe&#39;, &#39;mb&#39;, &#39;me&#39;, &#39;mean&#39;, &#39;meaningful&#39;, &#39;means&#39;, &#39;meant&#39;, &#39;media&#39;, &#39;medical&#39;, &#39;medications&#39;, &#39;mediocre&#39;, &#39;meditation&#39;, &#39;medium&#39;, &#39;meh&#39;, &#39;member&#39;, &#39;members&#39;, &#39;membership&#39;, &#39;memory&#39;, &#39;mention&#39;, &#39;mentioned&#39;, &#39;menu&#39;, &#39;mere&#39;, &#39;message&#39;, &#39;messages&#39;, &#39;messaging&#39;, &#39;messed&#39;, &#39;met&#39;, &#39;metro&#39;, &#39;mexico&#39;, &#39;mi&#39;, &#39;miami&#39;, &#39;mic&#39;, &#39;microphone&#39;, &#39;microphones&#39;, &#39;mics&#39;, &#39;mid&#39;, &#39;middle&#39;, &#39;mids&#39;, &#39;might&#39;, &#39;miles&#39;, &#39;million&#39;, &#39;mimic&#39;, &#39;mind&#39;, &#39;mindset&#39;, &#39;mine&#39;, &#39;mini&#39;, &#39;minimal&#39;, &#39;minimum&#39;, &#39;minor&#39;, &#39;minorly&#39;, &#39;mins&#39;, &#39;mint&#39;, &#39;minus&#39;, &#39;minute&#39;, &#39;minutes&#39;, &#39;mirroring&#39;, &#39;misled&#39;, &#39;misplace&#39;, &#39;miss&#39;, &#39;missed&#39;, &#39;missing&#39;, &#39;mistakes&#39;, &#39;misunderstands&#39;, &#39;mixed&#39;, &#39;moana&#39;, &#39;mobile&#39;, &#39;mobility&#39;, &#39;mode&#39;, &#39;model&#39;, &#39;models&#39;, &#39;modern&#39;, &#39;mom&#39;, &#39;moment&#39;, &#39;moms&#39;, &#39;mon&#39;, &#39;money&#39;, &#39;monitor&#39;, &#39;month&#39;, &#39;monthly&#39;, &#39;months&#39;, &#39;mood&#39;, &#39;more&#39;, &#39;moreover&#39;, &#39;morning&#39;, &#39;most&#39;, &#39;mostly&#39;, &#39;mother&#39;, &#39;motivation&#39;, &#39;motown&#39;, &#39;mount&#39;, &#39;mounted&#39;, &#39;move&#39;, &#39;moved&#39;, &#39;movie&#39;, &#39;movies&#39;, &#39;moving&#39;, &#39;mu&#39;, &#39;much&#39;, &#39;muffled&#39;, &#39;multi&#39;, &#39;multiple&#39;, &#39;music&#39;, &#39;must&#39;, &#39;mute&#39;, &#39;muy&#39;, &#39;my&#39;, &#39;mybedroom&#39;, &#39;myself&#39;, &#39;múltiples&#39;, &#39;na&#39;, &#39;name&#39;, &#39;named&#39;, &#39;names&#39;, &#39;nana&#39;, &#39;nanny&#39;, &#39;native&#39;, &#39;natural&#39;, &#39;nature&#39;, &#39;navigate&#39;, &#39;navigating&#39;, &#39;navigation&#39;, &#39;naw&#39;, &#39;nbc&#39;, &#39;nbsp&#39;, &#39;nc&#39;, &#39;nd&#39;, &#39;ne&#39;, &#39;near&#39;, &#39;nearly&#39;, &#39;neat&#39;, &#39;necessity&#39;, &#39;need&#39;, &#39;needed&#39;, &#39;needing&#39;, &#39;needs&#39;, &#39;negative&#39;, &#39;neighbors&#39;, &#39;neither&#39;, &#39;nephews&#39;, &#39;nervana&#39;, &#39;nervous&#39;, &#39;nest&#39;, &#39;net&#39;, &#39;netflix&#39;, &#39;network&#39;, &#39;never&#39;, &#39;new&#39;, &#39;newer&#39;, &#39;newest&#39;, &#39;news&#39;, &#39;newsflash&#39;, &#39;nexia&#39;, &#39;next&#39;, &#39;nfl&#39;, &#39;ni&#39;, &#39;nice&#39;, &#39;nicely&#39;, &#39;nicer&#39;, &#39;niece&#39;, &#39;nigh&#39;, &#39;night&#39;, &#39;nightmare&#39;, &#39;nights&#39;, &#39;nightstand&#39;, &#39;nil&#39;, &#39;nit&#39;, &#39;nite&#39;, &#39;nj&#39;, &#39;no&#39;, &#39;nobody&#39;, &#39;nois&#39;, &#39;noise&#39;, &#39;non&#39;, &#39;none&#39;, &#39;nonsense&#39;, &#39;nope&#39;, &#39;nor&#39;, &#39;norm&#39;, &#39;normal&#39;, &#39;north&#39;, &#39;nos&#39;, &#39;not&#39;, &#39;note&#39;, &#39;nothing&#39;, &#39;notice&#39;, &#39;noticeable&#39;, &#39;noticed&#39;, &#39;notification&#39;, &#39;notifications&#39;, &#39;notifies&#39;, &#39;novelty&#39;, &#39;now&#39;, &#39;nowhere&#39;, &#39;npr&#39;, &#39;nrw&#39;, &#39;nsa&#39;, &#39;nudged&#39;, &#39;numb&#39;, &#39;number&#39;, &#39;numbers&#39;, &#39;numerous&#39;, &#39;nurses&#39;, &#39;nuts&#39;, &#39;ny&#39;, &#39;obsessed&#39;, &#39;obtrusive&#39;, &#39;obvious&#39;, &#39;occasion&#39;, &#39;occasional&#39;, &#39;occasionally&#39;, &#39;ocean&#39;, &#39;odd&#39;, &#39;odds&#39;, &#39;of&#39;, &#39;off&#39;, &#39;offer&#39;, &#39;offered&#39;, &#39;offers&#39;, &#39;office&#39;, &#39;officially&#39;, &#39;offing&#39;, &#39;often&#39;, &#39;oh&#39;, &#39;ok&#39;, &#39;okay&#39;, &#39;old&#39;, &#39;older&#39;, &#39;oldest&#39;, &#39;olor&#39;, &#39;omg&#39;, &#39;on&#39;, &#39;once&#39;, &#39;onceproblem&#39;, &#39;one&#39;, &#39;ones&#39;, &#39;onetime&#39;, &#39;online&#39;, &#39;only&#39;, &#39;onme&#39;, &#39;onto&#39;, &#39;ontrac&#39;, &#39;oops&#39;, &#39;open&#39;, &#39;opened&#39;, &#39;opening&#39;, &#39;opens&#39;, &#39;opera&#39;, &#39;operate&#39;, &#39;operation&#39;, &#39;operations&#39;, &#39;operator&#39;, &#39;opinion&#39;, &#39;opportunity&#39;, &#39;opt&#39;, &#39;optical&#39;, &#39;optimum&#39;, &#39;option&#39;, &#39;optional&#39;, &#39;options&#39;, &#39;or&#39;, &#39;orange&#39;, &#39;orchestra&#39;, &#39;order&#39;, &#39;ordered&#39;, &#39;ordering&#39;, &#39;orders&#39;, &#39;organization&#39;, &#39;organized&#39;, &#39;orientation&#39;, &#39;oriented&#39;, &#39;original&#39;, &#39;originale&#39;, &#39;originally&#39;, &#39;other&#39;, &#39;others&#39;, &#39;otherwise&#39;, &#39;our&#39;, &#39;ours&#39;, &#39;ourselves&#39;, &#39;out&#39;, &#39;outdoor&#39;, &#39;outdoors&#39;, &#39;outlet&#39;, &#39;outlets&#39;, &#39;output&#39;, &#39;outrageous&#39;, &#39;outside&#39;, &#39;outsmart&#39;, &#39;outstanding&#39;, &#39;oven&#39;, &#39;over&#39;, &#39;overa&#39;, &#39;overall&#39;, &#39;overcoming&#39;, &#39;overheating&#39;, &#39;overpriced&#39;, &#39;override&#39;, &#39;overtime&#39;, &#39;overview&#39;, &#39;overwhelming&#39;, &#39;owe&#39;, &#39;owlhead&#39;, &#39;own&#39;, &#39;owned&#39;, &#39;owner&#39;, &#39;owners&#39;, &#39;ownership&#39;, &#39;owning&#39;, &#39;package&#39;, &#39;packaged&#39;, &#39;packages&#39;, &#39;packaging&#39;, &#39;packing&#39;, &#39;page&#39;, &#39;pages&#39;, &#39;paid&#39;, &#39;pain&#39;, &#39;pair&#39;, &#39;paired&#39;, &#39;pairing&#39;, &#39;pamphlet&#39;, &#39;pandora&#39;, &#39;pants&#39;, &#39;paper&#39;, &#39;par&#39;, &#39;paranoid&#39;, &#39;pare&#39;, &#39;parents&#39;, &#39;park&#39;, &#39;paroduct&#39;, &#39;part&#39;, &#39;participating&#39;, &#39;particular&#39;, &#39;particularly&#39;, &#39;parties&#39;, &#39;partner&#39;, &#39;parts&#39;, &#39;party&#39;, &#39;pass&#39;, &#39;password&#39;, &#39;past&#39;, &#39;patch&#39;, &#39;patience&#39;, &#39;patient&#39;, &#39;patio&#39;, &#39;pattern&#39;, &#39;pause&#39;, &#39;pauses&#39;, &#39;pay&#39;, &#39;payed&#39;, &#39;paying&#39;, &#39;pc&#39;, &#39;películas&#39;, &#39;pen&#39;, &#39;pencil&#39;, &#39;penny&#39;, &#39;people&#39;, &#39;pep&#39;, &#39;per&#39;, &#39;perdió&#39;, &#39;perfect&#39;, &#39;perfectly&#39;, &#39;perfecto&#39;, &#39;perform&#39;, &#39;performance&#39;, &#39;performed&#39;, &#39;performing&#39;, &#39;performs&#39;, &#39;perhaps&#39;, &#39;period&#39;, &#39;perk&#39;, &#39;permanently&#39;, &#39;persist&#39;, &#39;person&#39;, &#39;personal&#39;, &#39;personality&#39;, &#39;personalization&#39;, &#39;personalized&#39;, &#39;personally&#39;, &#39;persuasion&#39;, &#39;pets&#39;, &#39;phase&#39;, &#39;phenomenal&#39;, &#39;philip&#39;, &#39;philips&#39;, &#39;philipshue&#39;, &#39;phillip&#39;, &#39;phillips&#39;, &#39;philly&#39;, &#39;phone&#39;, &#39;phones&#39;, &#39;phonetically&#39;, &#39;photo&#39;, &#39;photographs&#39;, &#39;photos&#39;, &#39;phrase&#39;, &#39;pia&#39;, &#39;pick&#39;, &#39;picked&#39;, &#39;picking&#39;, &#39;picks&#39;, &#39;picky&#39;, &#39;pics&#39;, &#39;picture&#39;, &#39;pictures&#39;, &#39;piece&#39;, &#39;pin&#39;, &#39;pivoting&#39;, &#39;pixelated&#39;, &#39;pizza&#39;, &#39;place&#39;, &#39;placed&#39;, &#39;placement&#39;, &#39;places&#39;, &#39;placing&#39;, &#39;plain&#39;, &#39;plan&#39;, &#39;plane&#39;, &#39;planning&#39;, &#39;plans&#39;, &#39;platform&#39;, &#39;platforms&#39;, &#39;play&#39;, &#39;played&#39;, &#39;player&#39;, &#39;playing&#39;, &#39;playlist&#39;, &#39;playlists&#39;, &#39;plays&#39;, &#39;pleasantly&#39;, &#39;please&#39;, &#39;pleased&#39;, &#39;pleasedsimple&#39;, &#39;pleasure&#39;, &#39;plenty&#39;, &#39;plug&#39;, &#39;plugged&#39;, &#39;plugins&#39;, &#39;plugs&#39;, &#39;plus&#39;, &#39;pluto&#39;, &#39;pod&#39;, &#39;podcast&#39;, &#39;podcasts&#39;, &#39;point&#39;, &#39;pointed&#39;, &#39;pointless&#39;, &#39;politics&#39;, &#39;pool&#39;, &#39;poop&#39;, &#39;poor&#39;, &#39;pop&#39;, &#39;porch&#39;, &#39;port&#39;, &#39;portability&#39;, &#39;portable&#39;, &#39;portion&#39;, &#39;posed&#39;, &#39;position&#39;, &#39;positive&#39;, &#39;positives&#39;, &#39;possibilities&#39;, &#39;possible&#39;, &#39;possibly&#39;, &#39;post&#39;, &#39;poster&#39;, &#39;potential&#39;, &#39;pound&#39;, &#39;power&#39;, &#39;powercord&#39;, &#39;powerful&#39;, &#39;practical&#39;, &#39;practically&#39;, &#39;practicalthan&#39;, &#39;pray&#39;, &#39;pre&#39;, &#39;preciously&#39;, &#39;precise&#39;, &#39;prefer&#39;, &#39;preferences&#39;, &#39;preferred&#39;, &#39;premium&#39;, &#39;prepare&#39;, &#39;preparing&#39;, &#39;present&#39;, &#39;preset&#39;, &#39;press&#39;, &#39;presumably&#39;, &#39;prettier&#39;, &#39;pretty&#39;, &#39;prevent&#39;, &#39;prevents&#39;, &#39;preview&#39;, &#39;previous&#39;, &#39;previously&#39;, &#39;price&#39;, &#39;priced&#39;, &#39;prices&#39;, &#39;pricey&#39;, &#39;pricing&#39;, &#39;primarily&#39;, &#39;primary&#39;, &#39;prime&#39;, &#39;primeday&#39;, &#39;print&#39;, &#39;prior&#39;, &#39;privacy&#39;, &#39;prize&#39;, &#39;pro&#39;, &#39;probably&#39;, &#39;problem&#39;, &#39;problems&#39;, &#39;procedure&#39;, &#39;process&#39;, &#39;produc&#39;, &#39;product&#39;, &#39;producto&#39;, &#39;products&#39;, &#39;productsand&#39;, &#39;profiles&#39;, &#39;program&#39;, &#39;programing&#39;, &#39;programmed&#39;, &#39;programming&#39;, &#39;programs&#39;, &#39;project&#39;, &#39;projection&#39;, &#39;projects&#39;, &#39;promised&#39;, &#39;promoting&#39;, &#39;promotion&#39;, &#39;promp&#39;, &#39;prompt&#39;, &#39;prompts&#39;, &#39;proper&#39;, &#39;properly&#39;, &#39;props&#39;, &#39;pros&#39;, &#39;protected&#39;, &#39;protection&#39;, &#39;protocol&#39;, &#39;prove&#39;, &#39;proved&#39;, &#39;provee&#39;, &#39;provide&#39;, &#39;provided&#39;, &#39;provider&#39;, &#39;provides&#39;, &#39;providing&#39;, &#39;psychological&#39;, &#39;pueden&#39;, &#39;pull&#39;, &#39;pulling&#39;, &#39;pulsate&#39;, &#39;pulsed&#39;, &#39;punch&#39;, &#39;puny&#39;, &#39;pup&#39;, &#39;pur&#39;, &#39;purchase&#39;, &#39;purchased&#39;, &#39;purchaser&#39;, &#39;purchases&#39;, &#39;purchasing&#39;, &#39;pure&#39;, &#39;purely&#39;, &#39;purpose&#39;, &#39;purposes&#39;, &#39;push&#39;, &#39;pushed&#39;, &#39;put&#39;, &#39;puts&#39;, &#39;putting&#39;, &#39;puzzled&#39;, &#39;quality&#39;, &#39;qualty&#39;, &#39;que&#39;, &#39;quedó&#39;, &#39;queries&#39;, &#39;question&#39;, &#39;questionable&#39;, &#39;questions&#39;, &#39;quick&#39;, &#39;quicker&#39;, &#39;quickly&#39;, &#39;quiet&#39;, &#39;quit&#39;, &#39;quite&#39;, &#39;quiz&#39;, &#39;quot&#39;, &#39;quote&#39;, &#39;qvc&#39;, &#39;radio&#39;, &#39;rain&#39;, &#39;rainbow&#39;, &#39;raised&#39;, &#39;rambled&#39;, &#39;ran&#39;, &#39;random&#39;, &#39;randomly&#39;, &#39;range&#39;, &#39;ranger&#39;, &#39;rapidez&#39;, &#39;rare&#39;, &#39;rarely&#39;, &#39;rarity&#39;, &#39;rate&#39;, &#39;rather&#39;, &#39;rating&#39;, &#39;rattle&#39;, &#39;rattling&#39;, &#39;rcieved&#39;, &#39;re&#39;, &#39;reach&#39;, &#39;reached&#39;, &#39;reaching&#39;, &#39;reactive&#39;, &#39;read&#39;, &#39;reader&#39;, &#39;reading&#39;, &#39;reads&#39;, &#39;ready&#39;, &#39;real&#39;, &#39;realizando&#39;, &#39;realize&#39;, &#39;realized&#39;, &#39;realizing&#39;, &#39;really&#39;, &#39;reason&#39;, &#39;reasonable&#39;, &#39;reasons&#39;, &#39;reauthorize&#39;, &#39;reboot&#39;, &#39;rebooted&#39;, &#39;rebooting&#39;, &#39;reboots&#39;, &#39;reccomend&#39;, &#39;receivded&#39;, &#39;receive&#39;, &#39;received&#39;, &#39;receiver&#39;, &#39;receivers&#39;, &#39;receiving&#39;, &#39;recent&#39;, &#39;recently&#39;, &#39;reception&#39;, &#39;rechargeable&#39;, &#39;recharged&#39;, &#39;recipe&#39;, &#39;recipes&#39;, &#39;recipient&#39;, &#39;recognition&#39;, &#39;recognize&#39;, &#39;recognizes&#39;, &#39;recomendable&#39;, &#39;recommend&#39;, &#39;recommended&#39;, &#39;recommending&#39;, &#39;reconditioned&#39;, &#39;reconfigure&#39;, &#39;reconnect&#39;, &#39;reconnected&#39;, &#39;reconnecting&#39;, &#39;record&#39;, &#39;recorded&#39;, &#39;recording&#39;, &#39;recordings&#39;, &#39;rectangular&#39;, &#39;recurring&#39;, &#39;red&#39;, &#39;reduced&#39;, &#39;redundant&#39;, &#39;ref&#39;, &#39;refer&#39;, &#39;reference&#39;, &#39;references&#39;, &#39;referred&#39;, &#39;refers&#39;, &#39;refined&#39;, &#39;refund&#39;, &#39;refunds&#39;, &#39;refurb&#39;, &#39;refurbish&#39;, &#39;refurbished&#39;, &#39;refurbishedthought&#39;, &#39;refurbishing&#39;, &#39;refurbs&#39;, &#39;regard&#39;, &#39;regardless&#39;, &#39;regional&#39;, &#39;register&#39;, &#39;registered&#39;, &#39;regret&#39;, &#39;regrets&#39;, &#39;regular&#39;, &#39;regularly&#39;, &#39;reinstall&#39;, &#39;related&#39;, &#39;relatively&#39;, &#39;relaxing&#39;, &#39;relay&#39;, &#39;release&#39;, &#39;released&#39;, &#39;reliable&#39;, &#39;relief&#39;, &#39;rely&#39;, &#39;remaining&#39;, &#39;remains&#39;, &#39;remedial&#39;, &#39;remember&#39;, &#39;remembering&#39;, &#39;remind&#39;, &#39;reminded&#39;, &#39;reminder&#39;, &#39;reminders&#39;, &#39;reminding&#39;, &#39;reminds&#39;, &#39;remorse&#39;, &#39;remote&#39;, &#39;rename&#39;, &#39;rent&#39;, &#39;renting&#39;, &#39;reoccurring&#39;, &#39;reorder&#39;, &#39;rep&#39;, &#39;repair&#39;, &#39;repairs&#39;, &#39;repeat&#39;, &#39;repeated&#39;, &#39;repeating&#39;, &#39;repeats&#39;, &#39;repertoire&#39;, &#39;replace&#39;, &#39;replaced&#39;, &#39;replacement&#39;, &#39;replaces&#39;, &#39;replacing&#39;, &#39;replied&#39;, &#39;replying&#39;, &#39;report&#39;, &#39;reported&#39;, &#39;reports&#39;, &#39;reportsalarm&#39;, &#39;reputation&#39;, &#39;request&#39;, &#39;requesting&#39;, &#39;requests&#39;, &#39;require&#39;, &#39;required&#39;, &#39;requires&#39;, &#39;research&#39;, &#39;researched&#39;, &#39;researching&#39;, &#39;resembling&#39;, &#39;resemption&#39;, &#39;reset&#39;, &#39;resetting&#39;, &#39;resist&#39;, &#39;resistant&#39;, &#39;resolution&#39;, &#39;resolved&#39;, &#39;resolves&#39;, &#39;respond&#39;, &#39;responding&#39;, &#39;responds&#39;, &#39;response&#39;, &#39;responses&#39;, &#39;responsive&#39;, &#39;responsiveness&#39;, &#39;respuesta&#39;, &#39;rest&#39;, &#39;restart&#39;, &#39;restrictions&#39;, &#39;restrictive&#39;, &#39;result&#39;, &#39;results&#39;, &#39;resume&#39;, &#39;retired&#39;, &#39;return&#39;, &#39;returned&#39;, &#39;returnef&#39;, &#39;returning&#39;, &#39;review&#39;, &#39;reviewing&#39;, &#39;reviews&#39;, &#39;revise&#39;, &#39;rewards&#39;, &#39;rid&#39;, &#39;rides&#39;, &#39;ridiculous&#39;, &#39;ridiculously&#39;, &#39;right&#39;, &#39;ring&#39;, &#39;rings&#39;, &#39;rivers&#39;, &#39;road&#39;, &#39;rock&#39;, &#39;rocks&#39;, &#39;roku&#39;, &#39;roll&#39;, &#39;room&#39;, &#39;roomba&#39;, &#39;rooms&#39;, &#39;rotate&#39;, &#39;rotates&#39;, &#39;rotation&#39;, &#39;rotations&#39;, &#39;rough&#39;, &#39;round&#39;, &#39;route&#39;, &#39;router&#39;, &#39;routine&#39;, &#39;routinely&#39;, &#39;routines&#39;, &#39;row&#39;, &#39;rub&#39;, &#39;rubber&#39;, &#39;run&#39;, &#39;running&#39;, &#39;runs&#39;, &#39;s8&#39;, &#39;s9&#39;, &#39;sad&#39;, &#39;sadly&#39;, &#39;safe&#39;, &#39;said&#39;, &#39;sale&#39;, &#39;sales&#39;, &#39;salsa&#39;, &#39;same&#39;, &#39;samsung&#39;, &#39;sang&#39;, &#39;sanity&#39;, &#39;satellite&#39;, &#39;satisfied&#39;, &#39;satisified&#39;, &#39;save&#39;, &#39;saved&#39;, &#39;saving&#39;, &#39;savvy&#39;, &#39;savy&#39;, &#39;saw&#39;, &#39;say&#39;, &#39;saying&#39;, &#39;says&#39;, &#39;scared&#39;, &#39;scenes&#39;, &#39;scent&#39;, &#39;schedule&#39;, &#39;scheduled&#39;, &#39;schedules&#39;, &#39;scheduling&#39;, &#39;school&#39;, &#39;science&#39;, &#39;scooped&#39;, &#39;scores&#39;, &#39;scottish&#39;, &#39;scoured&#39;, &#39;scratch&#39;, &#39;scratched&#39;, &#39;screamig&#39;, &#39;screaming&#39;, &#39;screen&#39;, &#39;screenless&#39;, &#39;screens&#39;, &#39;screenselect&#39;, &#39;screw&#39;, &#39;script&#39;, &#39;scroll&#39;, &#39;scrolling&#39;, &#39;scrolls&#39;, &#39;se&#39;, &#39;sealed&#39;, &#39;seamless&#39;, &#39;seamlessly&#39;, &#39;seams&#39;, &#39;search&#39;, &#39;searches&#39;, &#39;searching&#39;, &#39;season&#39;, &#39;second&#39;, &#39;seconds&#39;, &#39;secret&#39;, &#39;secretary&#39;, &#39;section&#39;, &#39;security&#39;, &#39;see&#39;, &#39;seeing&#39;, &#39;seem&#39;, &#39;seemed&#39;, &#39;seems&#39;, &#39;seen&#39;, &#39;seldom&#39;, &#39;select&#39;, &#39;selection&#39;, &#39;selections&#39;, &#39;self&#39;, &#39;selfies&#39;, &#39;sell&#39;, &#39;selling&#39;, &#39;semana&#39;, &#39;semi&#39;, &#39;send&#39;, &#39;sending&#39;, &#39;sends&#39;, &#39;senior&#39;, &#39;sense&#39;, &#39;sensitive&#39;, &#39;sensitivity&#39;, &#39;sent&#39;, &#39;sentence&#39;, &#39;separate&#39;, &#39;separately&#39;, &#39;seprately&#39;, &#39;series&#39;, &#39;serious&#39;, &#39;seriously&#39;, &#39;serius&#39;, &#39;serve&#39;, &#39;served&#39;, &#39;service&#39;, &#39;services&#39;, &#39;set&#39;, &#39;sets&#39;, &#39;setting&#39;, &#39;settings&#39;, &#39;settingshome&#39;, &#39;settins&#39;, &#39;settle&#39;, &#39;setup&#39;, &#39;setups&#39;, &#39;sever&#39;, &#39;several&#39;, &#39;sewing&#39;, &#39;sh&#39;, &#39;shaking&#39;, &#39;shape&#39;, &#39;sharing&#39;, &#39;sharp&#39;, &#39;she&#39;, &#39;shell&#39;, &#39;shelled&#39;, &#39;shifting&#39;, &#39;shine&#39;, &#39;shining&#39;, &#39;ship&#39;, &#39;shipment&#39;, &#39;shipped&#39;, &#39;shipping&#39;, &#39;shocked&#39;, &#39;shooting&#39;, &#39;shop&#39;, &#39;shopping&#39;, &#39;short&#39;, &#39;shortcomings&#39;, &#39;shorted&#39;, &#39;shorter&#39;, &#39;shortly&#39;, &#39;should&#39;, &#39;shouldn&#39;, &#39;shout&#39;, &#39;show&#39;, &#39;shower&#39;, &#39;showering&#39;, &#39;showing&#39;, &#39;showman&#39;, &#39;shown&#39;, &#39;shows&#39;, &#39;showtime&#39;, &#39;shuffle&#39;, &#39;shut&#39;, &#39;shuts&#39;, &#39;shutting&#39;, &#39;sibling&#39;, &#39;side&#39;, &#39;sigh&#39;, &#39;sight&#39;, &#39;sign&#39;, &#39;significant&#39;, &#39;silly&#39;, &#39;silver&#39;, &#39;similar&#39;, &#39;simple&#39;, &#39;simpler&#39;, &#39;simplicity&#39;, &#39;simplified&#39;, &#39;simplify&#39;, &#39;simply&#39;, &#39;simultaneously&#39;, &#39;sin&#39;, &#39;since&#39;, &#39;sincerely&#39;, &#39;sing&#39;, &#39;singing&#39;, &#39;single&#39;, &#39;singley&#39;, &#39;sink&#39;, &#39;sinqued&#39;, &#39;siri&#39;, &#39;sirius&#39;, &#39;sirrius&#39;, &#39;sister&#39;, &#39;sit&#39;, &#39;site&#39;, &#39;sits&#39;, &#39;sitting&#39;, &#39;situations&#39;, &#39;six&#39;, &#39;size&#39;, &#39;sized&#39;, &#39;skeptical&#39;, &#39;skill&#39;, &#39;skills&#39;, &#39;skips&#39;, &#39;skype&#39;, &#39;sleek&#39;, &#39;sleep&#39;, &#39;sleeper&#39;, &#39;sleeping&#39;, &#39;sleeps&#39;, &#39;sleepy&#39;, &#39;sliced&#39;, &#39;slide&#39;, &#39;slideshow&#39;, &#39;slight&#39;, &#39;slightly&#39;, &#39;sling&#39;, &#39;slow&#39;, &#39;slowly&#39;, &#39;sm&#39;, &#39;small&#39;, &#39;smaller&#39;, &#39;smart&#39;, &#39;smartbon&#39;, &#39;smarter&#39;, &#39;smarthome&#39;, &#39;smartphone&#39;, &#39;smartthing&#39;, &#39;smartthings&#39;, &#39;smells&#39;, &#39;smiths&#39;, &#39;smooth&#39;, &#39;smoothly&#39;, &#39;snap&#39;, &#39;snarls&#39;, &#39;sneaky&#39;, &#39;snell&#39;, &#39;snooze&#39;, &#39;snoozed&#39;, &#39;snoozes&#39;, &#39;so&#39;, &#39;soaked&#39;, &#39;soaking&#39;, &#39;soccer&#39;, &#39;social&#39;, &#39;socket&#39;, &#39;sofa&#39;, &#39;soft&#39;, &#39;softly&#39;, &#39;software&#39;, &#39;sold&#39;, &#39;solely&#39;, &#39;solid&#39;, &#39;solo&#39;, &#39;solución&#39;, &#39;solution&#39;, &#39;solved&#39;, &#39;solves&#39;, &#39;some&#39;, &#39;somebody&#39;, &#39;somehow&#39;, &#39;someone&#39;, &#39;something&#39;, &#39;sometime&#39;, &#39;sometimes&#39;, &#39;somewhat&#39;, &#39;son&#39;, &#39;song&#39;, &#39;songs&#39;, &#39;sonos&#39;, &#39;sons&#39;, &#39;sony&#39;, &#39;soon&#39;, &#39;sooner&#39;, &#39;sooo&#39;, &#39;sooooo&#39;, &#39;sooooooo&#39;, &#39;sopt&#39;, &#39;sore&#39;, &#39;sorely&#39;, &#39;sorprendió&#39;, &#39;sorry&#39;, &#39;sort&#39;, &#39;sound&#39;, &#39;soundbar&#39;, &#39;sounded&#39;, &#39;sounding&#39;, &#39;soundlink&#39;, &#39;sounds&#39;, &#39;soundtouch&#39;, &#39;source&#39;, &#39;sources&#39;, &#39;southern&#39;, &#39;spa&#39;, &#39;space&#39;, &#39;spaces&#39;, &#39;spacing&#39;, &#39;spam&#39;, &#39;span&#39;, &#39;spanish&#39;, &#39;spanking&#39;, &#39;spark&#39;, &#39;sparks&#39;, &#39;speak&#39;, &#39;speaker&#39;, &#39;speakers&#39;, &#39;speaking&#39;, &#39;speaks&#39;, &#39;special&#39;, &#39;specially&#39;, &#39;specific&#39;, &#39;specifically&#39;, &#39;specifily&#39;, &#39;specify&#39;, &#39;specifying&#39;, &#39;specs&#39;, &#39;spectacular&#39;, &#39;speech&#39;, &#39;speed&#39;, &#39;speeds&#39;, &#39;speedy&#39;, &#39;spell&#39;, &#39;spelling&#39;, &#39;spend&#39;, &#39;spending&#39;, &#39;spent&#39;, &#39;spiel&#39;, &#39;spilled&#39;, &#39;spin&#39;, &#39;spins&#39;, &#39;split&#39;, &#39;spoiled&#39;, &#39;spoke&#39;, &#39;spoken&#39;, &#39;sport&#39;, &#39;sports&#39;, &#39;spot&#39;, &#39;spotify&#39;, &#39;spotlight&#39;, &#39;spots&#39;, &#39;spouse&#39;, &#39;sprinkler&#39;, &#39;sprint&#39;, &#39;spur&#39;, &#39;spying&#39;, &#39;square&#39;, &#39;squirms&#39;, &#39;sry&#39;, &#39;ssdi&#39;, &#39;st&#39;, &#39;staff&#39;, &#39;stage&#39;, &#39;staging&#39;, &#39;stairs&#39;, &#39;stand&#39;, &#39;standalone&#39;, &#39;standard&#39;, &#39;standards&#39;, &#39;standing&#39;, &#39;stands&#39;, &#39;star&#39;, &#39;stark&#39;, &#39;stars&#39;, &#39;start&#39;, &#39;started&#39;, &#39;starting&#39;, &#39;starts&#39;, &#39;stat&#39;, &#39;state&#39;, &#39;statement&#39;, &#39;states&#39;, &#39;station&#39;, &#39;stationary&#39;, &#39;stationed&#39;, &#39;stations&#39;, &#39;stay&#39;, &#39;stayed&#39;, &#39;staying&#39;, &#39;steaming&#39;, &#39;steep&#39;, &#39;stellar&#39;, &#39;step&#39;, &#39;steps&#39;, &#39;stereo&#39;, &#39;stick&#39;, &#39;sticks&#39;, &#39;still&#39;, &#39;stimulus&#39;, &#39;stinks&#39;, &#39;stoled&#39;, &#39;stop&#39;, &#39;stopped&#39;, &#39;stops&#39;, &#39;storage&#39;, &#39;store&#39;, &#39;stories&#39;, &#39;storm&#39;, &#39;story&#39;, &#39;stove&#39;, &#39;straight&#39;, &#39;straightforward&#39;, &#39;strange&#39;, &#39;stream&#39;, &#39;streaming&#39;, &#39;streamline&#39;, &#39;strictly&#39;, &#39;string&#39;, &#39;strips&#39;, &#39;strong&#39;, &#39;strongly&#39;, &#39;structure&#39;, &#39;struggle&#39;, &#39;stubborn&#39;, &#39;stuck&#39;, &#39;students&#39;, &#39;stuff&#39;, &#39;stump&#39;, &#39;stupid&#39;, &#39;sturdy&#39;, &#39;style&#39;, &#39;stylish&#39;, &#39;su&#39;, &#39;sub&#39;, &#39;subject&#39;, &#39;subpar&#39;, &#39;subscriber&#39;, &#39;subscribing&#39;, &#39;subscription&#39;, &#39;subscriptiondoes&#39;, &#39;subscriptions&#39;, &#39;subsequently&#39;, &#39;substitute&#39;, &#39;success&#39;, &#39;successful&#39;, &#39;successfully&#39;, &#39;successor&#39;, &#39;such&#39;, &#39;suck&#39;, &#39;sucks&#39;, &#39;suffer&#39;, &#39;sufficient&#39;, &#39;suffolk&#39;, &#39;suggest&#39;, &#39;suggested&#39;, &#39;suggesting&#39;, &#39;suggestions&#39;, &#39;suggests&#39;, &#39;suitable&#39;, &#39;summoning&#39;, &#39;sunroom&#39;, &#39;supberb&#39;, &#39;super&#39;, &#39;superb&#39;, &#39;superior&#39;, &#39;supplied&#39;, &#39;supplying&#39;, &#39;support&#39;, &#39;supported&#39;, &#39;supporting&#39;, &#39;supports&#39;, &#39;suppose&#39;, &#39;supposed&#39;, &#39;sure&#39;, &#39;surely&#39;, &#39;surface&#39;, &#39;surprise&#39;, &#39;surprised&#39;, &#39;surprising&#39;, &#39;surprisingly&#39;, &#39;surround&#39;, &#39;survived&#39;, &#39;sweet&#39;, &#39;swell&#39;, &#39;swipe&#39;, &#39;swiping&#39;, &#39;switch&#39;, &#39;switched&#39;, &#39;switches&#39;, &#39;switching&#39;, &#39;sync&#39;, &#39;synced&#39;, &#39;synching&#39;, &#39;syncing&#39;, &#39;system&#39;, &#39;systems&#39;, &#39;table&#39;, &#39;tablet&#39;, &#39;tablets&#39;, &#39;tad&#39;, &#39;tailor&#39;, &#39;take&#39;, &#39;taken&#39;, &#39;takes&#39;, &#39;taking&#39;, &#39;tales&#39;, &#39;talk&#39;, &#39;talked&#39;, &#39;talking&#39;, &#39;talks&#39;, &#39;tall&#39;, &#39;taller&#39;, &#39;tap&#39;, &#39;tape&#39;, &#39;taping&#39;, &#39;tapped&#39;, &#39;tardis&#39;, &#39;tasha&#39;, &#39;task&#39;, &#39;tasks&#39;, &#39;teacher&#39;, &#39;teams&#39;, &#39;tear&#39;, &#39;tec&#39;, &#39;tech&#39;, &#39;techie&#39;, &#39;technical&#39;, &#39;technically&#39;, &#39;technicians&#39;, &#39;techno&#39;, &#39;technologically&#39;, &#39;technology&#39;, &#39;techy&#39;, &#39;teenagers&#39;, &#39;teeth&#39;, &#39;tekkie&#39;, &#39;telephone&#39;, &#39;television&#39;, &#39;tell&#39;, &#39;telling&#39;, &#39;tells&#39;, &#39;temp&#39;, &#39;temperature&#39;, &#39;temps&#39;, &#39;tempting&#39;, &#39;ten&#39;, &#39;tend&#39;, &#39;tends&#39;, &#39;terminology&#39;, &#39;terrible&#39;, &#39;terrific&#39;, &#39;test&#39;, &#39;tested&#39;, &#39;testing&#39;, &#39;texas&#39;, &#39;text&#39;, &#39;texts&#39;, &#39;tg&#39;, &#39;tge&#39;, &#39;than&#39;, &#39;thank&#39;, &#39;thanks&#39;, &#39;that&#39;, &#39;thats&#39;, &#39;the&#39;, &#39;theater&#39;, &#39;theecho&#39;, &#39;their&#39;, &#39;theirs&#39;, &#39;them&#39;, &#39;themes&#39;, &#39;themselves&#39;, &#39;then&#39;, &#39;theories&#39;, &#39;there&#39;, &#39;therefore&#39;, &#39;thermostat&#39;, &#39;these&#39;, &#39;thestand&#39;, &#39;thete&#39;, &#39;they&#39;, &#39;thick&#39;, &#39;thing&#39;, &#39;things&#39;, &#39;think&#39;, &#39;thinking&#39;, &#39;third&#39;, &#39;this&#39;, &#39;thongs&#39;, &#39;thorough&#39;, &#39;thoroughly&#39;, &#39;those&#39;, &#39;thou&#39;, &#39;though&#39;, &#39;thought&#39;, &#39;thoughts&#39;, &#39;thousands&#39;, &#39;three&#39;, &#39;thrilled&#39;, &#39;through&#39;, &#39;throughout&#39;, &#39;throw&#39;, &#39;thrown&#39;, &#39;thru&#39;, &#39;thu&#39;, &#39;thumb&#39;, &#39;thumbs&#39;, &#39;thunderstorm&#39;, &#39;thursday&#39;, &#39;ti&#39;, &#39;tickled&#39;, &#39;tiempo&#39;, &#39;tiene&#39;, &#39;ties&#39;, &#39;til&#39;, &#39;till&#39;, &#39;time&#39;, &#39;timer&#39;, &#39;timers&#39;, &#39;times&#39;, &#39;timing&#39;, &#39;tin&#39;, &#39;ting&#39;, &#39;tinker&#39;, &#39;tinkering&#39;, &#39;tinny&#39;, &#39;tiny&#39;, &#39;tipping&#39;, &#39;tips&#39;, &#39;tired&#39;, &#39;title&#39;, &#39;tivo&#39;, &#39;to&#39;, &#39;toda&#39;, &#39;today&#39;, &#39;toddler&#39;, &#39;together&#39;, &#39;toilet&#39;, &#39;told&#39;, &#39;tomorrow&#39;, &#39;tomy&#39;, &#39;ton&#39;, &#39;tones&#39;, &#39;tons&#39;, &#39;tony&#39;, &#39;too&#39;, &#39;took&#39;, &#39;tool&#39;, &#39;tools&#39;, &#39;tooth&#39;, &#39;top&#39;, &#39;topic&#39;, &#39;tosca&#39;, &#39;total&#39;, &#39;totallly&#39;, &#39;totally&#39;, &#39;tou&#39;, &#39;touch&#39;, &#39;touching&#39;, &#39;touted&#39;, &#39;toward&#39;, &#39;towards&#39;, &#39;tower&#39;, &#39;town&#39;, &#39;toy&#39;, &#39;tp&#39;, &#39;track&#39;, &#39;traditional&#39;, &#39;traffic&#39;, &#39;trailer&#39;, &#39;trailers&#39;, &#39;trained&#39;, &#39;trainees&#39;, &#39;training&#39;, &#39;transferring&#39;, &#39;travel&#39;, &#39;traveling&#39;, &#39;travelling&#39;, &#39;través&#39;, &#39;treadmill&#39;, &#39;treat&#39;, &#39;treble&#39;, &#39;trek&#39;, &#39;tremendous&#39;, &#39;trending&#39;, &#39;trial&#39;, &#39;tricks&#39;, &#39;tricky&#39;, &#39;tried&#39;, &#39;tries&#39;, &#39;trigger&#39;, &#39;trip&#39;, &#39;trivia&#39;, &#39;trouble&#39;, &#39;troubleshooting&#39;, &#39;troublesome&#39;, &#39;troubling&#39;, &#39;true&#39;, &#39;truly&#39;, &#39;trust&#39;, &#39;try&#39;, &#39;trying&#39;, &#39;tube&#39;, &#39;tubi&#39;, &#39;tune&#39;, &#39;tunein&#39;, &#39;tunes&#39;, &#39;turn&#39;, &#39;turned&#39;, &#39;turning&#39;, &#39;turns&#39;, &#39;tv&#39;, &#39;tvs&#39;, &#39;tweeter&#39;, &#39;tweeters&#39;, &#39;twice&#39;, &#39;twist&#39;, &#39;twitter&#39;, &#39;two&#39;, &#39;ty&#39;, &#39;type&#39;, &#39;typed&#39;, &#39;types&#39;, &#39;typical&#39;, &#39;typically&#39;, &#39;typing&#39;, &#39;títulos&#39;, &#39;udefulness&#39;, &#39;ugly&#39;, &#39;uhyour&#39;, &#39;ummm&#39;, &#39;un&#39;, &#39;unable&#39;, &#39;unacceptable&#39;, &#39;unavailable&#39;, &#39;unbelievable&#39;, &#39;uncle&#39;, &#39;under&#39;, &#39;underestimated&#39;, &#39;understand&#39;, &#39;understanding&#39;, &#39;understands&#39;, &#39;understood&#39;, &#39;unexpected&#39;, &#39;unfortunately&#39;, &#39;unhappy&#39;, &#39;unhelpful&#39;, &#39;unico&#39;, &#39;unimportant&#39;, &#39;uninstall&#39;, &#39;unique&#39;, &#39;unit&#39;, &#39;units&#39;, &#39;universal&#39;, &#39;unless&#39;, &#39;unlike&#39;, &#39;unlimited&#39;, &#39;unlocking&#39;, &#39;unnannounced&#39;, &#39;unnecessary&#39;, &#39;unobtrusive&#39;, &#39;unplug&#39;, &#39;unplugged&#39;, &#39;unresponsive&#39;, &#39;unsettling&#39;, &#39;untapped&#39;, &#39;until&#39;, &#39;unusable&#39;, &#39;unused&#39;, &#39;unwitty&#39;, &#39;unwrapped&#39;, &#39;up&#39;, &#39;upcoming&#39;, &#39;update&#39;, &#39;updated&#39;, &#39;updates&#39;, &#39;updating&#39;, &#39;upgrade&#39;, &#39;upgraded&#39;, &#39;upgrades&#39;, &#39;upgrading&#39;, &#39;upload&#39;, &#39;upon&#39;, &#39;upset&#39;, &#39;upsetting&#39;, &#39;upstairs&#39;, &#39;urge&#39;, &#39;us&#39;, &#39;usa&#39;, &#39;usable&#39;, &#39;usage&#39;, &#39;usb&#39;, &#39;usde&#39;, &#39;use&#39;, &#39;used&#39;, &#39;useful&#39;, &#39;useless&#39;, &#39;user&#39;, &#39;users&#39;, &#39;uses&#39;, &#39;using&#39;, &#39;usual&#39;, &#39;usually&#39;, &#39;utility&#39;, &#39;utilización&#39;, &#39;utilize&#39;, &#39;utilizing&#39;, &#39;vacation&#39;, &#39;vacations&#39;, &#39;vacuum&#39;, &#39;value&#39;, &#39;variant&#39;, &#39;variety&#39;, &#39;various&#39;, &#39;vast&#39;, &#39;ve&#39;, &#39;vehicle&#39;, &#39;verbal&#39;, &#39;verbalize&#39;, &#39;verbally&#39;, &#39;versa&#39;, &#39;versatile&#39;, &#39;versatility&#39;, &#39;verse&#39;, &#39;verses&#39;, &#39;version&#39;, &#39;versions&#39;, &#39;versus&#39;, &#39;very&#39;, &#39;vetted&#39;, &#39;vez&#39;, &#39;via&#39;, &#39;vibrating&#39;, &#39;vice&#39;, &#39;viceo&#39;, &#39;video&#39;, &#39;videos&#39;, &#39;view&#39;, &#39;viewed&#39;, &#39;viewing&#39;, &#39;views&#39;, &#39;vintage&#39;, &#39;viola&#39;, &#39;virtual&#39;, &#39;virtually&#39;, &#39;visa&#39;, &#39;visible&#39;, &#39;vision&#39;, &#39;visiting&#39;, &#39;visits&#39;, &#39;visual&#39;, &#39;visuals&#39;, &#39;vlan&#39;, &#39;voice&#39;, &#39;voices&#39;, &#39;voiceview&#39;, &#39;voila&#39;, &#39;voltage&#39;, &#39;voltson&#39;, &#39;volume&#39;, &#39;vs&#39;, &#39;vudu&#39;, &#39;wait&#39;, &#39;waited&#39;, &#39;waiting&#39;, &#39;waits&#39;, &#39;wake&#39;, &#39;wakes&#39;, &#39;waking&#39;, &#39;walk&#39;, &#39;walked&#39;, &#39;walking&#39;, &#39;walks&#39;, &#39;wall&#39;, &#39;walls&#39;, &#39;want&#39;, &#39;wanted&#39;, &#39;wanting&#39;, &#39;warehouse&#39;, &#39;warning&#39;, &#39;warns&#39;, &#39;warranty&#39;, &#39;was&#39;, &#39;wasconcerned&#39;, &#39;wasn&#39;, &#39;wasnt&#39;, &#39;waste&#39;, &#39;wasted&#39;, &#39;watch&#39;, &#39;watched&#39;, &#39;watching&#39;, &#39;water&#39;, &#39;wattage&#39;, &#39;wave&#39;, &#39;way&#39;, &#39;ways&#39;, &#39;we&#39;, &#39;weak&#39;, &#39;wealth&#39;, &#39;wear&#39;, &#39;weary&#39;, &#39;weather&#39;, &#39;web&#39;, &#39;website&#39;, &#39;websites&#39;, &#39;wedding&#39;, &#39;week&#39;, &#39;weekday&#39;, &#39;weekdays&#39;, &#39;weekend&#39;, &#39;weekly&#39;, &#39;weeks&#39;, &#39;weight&#39;, &#39;weird&#39;, &#39;welcome&#39;, &#39;well&#39;, &#39;wellfour&#39;, &#39;went&#39;, &#39;were&#39;, &#39;weren&#39;, &#39;what&#39;, &#39;whatever&#39;, &#39;whats&#39;, &#39;whatsoever&#39;, &#39;whe&#39;, &#39;when&#39;, &#39;whenever&#39;, &#39;where&#39;, &#39;wherever&#39;, &#39;whether&#39;, &#39;which&#39;, &#39;while&#39;, &#39;whisper&#39;, &#39;whistles&#39;, &#39;white&#39;, &#39;who&#39;, &#39;whole&#39;, &#39;whom&#39;, &#39;whos&#39;, &#39;whose&#39;, &#39;why&#39;, &#39;wi&#39;, &#39;wide&#39;, &#39;widespread&#39;, &#39;wife&#39;, &#39;wifi&#39;, &#39;wikipedia&#39;, &#39;will&#39;, &#39;willing&#39;, &#39;wind&#39;, &#39;window&#39;, &#39;winds&#39;, &#39;wink&#39;, &#39;wireless&#39;, &#39;wish&#39;, &#39;wished&#39;, &#39;wishing&#39;, &#39;with&#39;, &#39;within&#39;, &#39;without&#39;, &#39;woke&#39;, &#39;woken&#39;, &#39;won&#39;, &#39;wonder&#39;, &#39;wonderful&#39;, &#39;wonderfully&#39;, &#39;wonders&#39;, &#39;wont&#39;, &#39;woofer&#39;, &#39;woofers&#39;, &#39;woohoo&#39;, &#39;word&#39;, &#39;words&#39;, &#39;work&#39;, &#39;workarounds&#39;, &#39;worked&#39;, &#39;worker&#39;, &#39;working&#39;, &#39;workout&#39;, &#39;workreat&#39;, &#39;works&#39;, &#39;world&#39;, &#39;worried&#39;, &#39;worry&#39;, &#39;worse&#39;, &#39;worst&#39;, &#39;worth&#39;, &#39;worthless&#39;, &#39;worthy&#39;, &#39;would&#39;, &#39;wouldn&#39;, &#39;wow&#39;, &#39;writes&#39;, &#39;writing&#39;, &#39;wrong&#39;, &#39;www&#39;, &#39;xbox&#39;, &#39;xfinity&#39;, &#39;xm&#39;, &#39;yale&#39;, &#39;yard&#39;, &#39;yards&#39;, &#39;yeah&#39;, &#39;year&#39;, &#39;years&#39;, &#39;yell&#39;, &#39;yelling&#39;, &#39;yellow&#39;, &#39;yep&#39;, &#39;yes&#39;, &#39;yesterday&#39;, &#39;yet&#39;, &#39;yhe&#39;, &#39;york&#39;, &#39;you&#39;, &#39;young&#39;, &#39;younger&#39;, &#39;youngest&#39;, &#39;your&#39;, &#39;yourself&#39;, &#39;youtube&#39;, &#39;yr&#39;, &#39;yrs&#39;, &#39;yup&#39;, &#39;zero&#39;, &#39;zigbee&#39;, &#39;zonked&#39;, &#39;zzzz&#39;, &#39;zzzzzzz&#39;, &#39;útil&#39;]
</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;verified_reviews&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>feedback</th>
      <th>Black  Dot</th>
      <th>Black  Plus</th>
      <th>Black  Show</th>
      <th>Black  Spot</th>
      <th>Charcoal Fabric</th>
      <th>Configuration: Fire TV Stick</th>
      <th>Heather Gray Fabric</th>
      <th>Oak Finish</th>
      <th>Sandstone Fabric</th>
      <th>Walnut Finish</th>
      <th>White</th>
      <th>White  Dot</th>
      <th>White  Plus</th>
      <th>White  Show</th>
      <th>White  Spot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">encoded_reviews</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">alexa_countcetorizer</span><span class="o">.</span><span class="n">toarray</span><span class="p">())</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">alexa_df</span><span class="p">,</span> <span class="n">encoded_reviews</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[34]:</div>



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
      <th>feedback</th>
      <th>Black  Dot</th>
      <th>Black  Plus</th>
      <th>Black  Show</th>
      <th>Black  Spot</th>
      <th>Charcoal Fabric</th>
      <th>Configuration: Fire TV Stick</th>
      <th>Heather Gray Fabric</th>
      <th>Oak Finish</th>
      <th>Sandstone Fabric</th>
      <th>...</th>
      <th>4034</th>
      <th>4035</th>
      <th>4036</th>
      <th>4037</th>
      <th>4038</th>
      <th>4039</th>
      <th>4040</th>
      <th>4041</th>
      <th>4042</th>
      <th>4043</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4060 columns</p>
</div>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[35]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(3150, 4060)</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;feedback&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
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
      <th>Black  Dot</th>
      <th>Black  Plus</th>
      <th>Black  Show</th>
      <th>Black  Spot</th>
      <th>Charcoal Fabric</th>
      <th>Configuration: Fire TV Stick</th>
      <th>Heather Gray Fabric</th>
      <th>Oak Finish</th>
      <th>Sandstone Fabric</th>
      <th>Walnut Finish</th>
      <th>...</th>
      <th>4034</th>
      <th>4035</th>
      <th>4036</th>
      <th>4037</th>
      <th>4038</th>
      <th>4039</th>
      <th>4040</th>
      <th>4041</th>
      <th>4042</th>
      <th>4043</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4059 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;feedback&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
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

<div class="prompt output_prompt">Out[41]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    1
1    1
2    1
3    1
4    1
Name: feedback, dtype: int64</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Create train Test Split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span>
                                                    <span class="n">y</span><span class="p">,</span>
                                                    <span class="n">test_size</span><span class="o">=</span><span class="mf">0.20</span><span class="p">,</span>
                                                    <span class="n">random_state</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
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
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">classification_report</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">randomforest_classifier</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">100</span><span class="p">,</span>
                                                 <span class="n">criterion</span> <span class="o">=</span><span class="s1">&#39;entropy&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
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

<div class="prompt output_prompt">Out[47]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>RandomForestClassifier(bootstrap=True, class_weight=None, criterion=&#39;entropy&#39;,
            max_depth=None, max_features=&#39;auto&#39;, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[58]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Evaluate</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[59]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict</span> <span class="o">=</span> <span class="n">randomforest_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[60]:</div>
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
<div class="prompt input_prompt">In&nbsp;[61]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cm</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[61]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[ 15,  36],
       [  1, 578]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[62]:</div>
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

<div class="prompt output_prompt">Out[62]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a3b58b438&gt;</pre>
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
<div class="prompt input_prompt">In&nbsp;[64]:</div>
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

          0       0.94      0.29      0.45        51
          1       0.94      1.00      0.97       579

avg / total       0.94      0.94      0.93       630

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
<h3 id="Model-Improve">Model Improve<a class="anchor-link" href="#Model-Improve">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../datasets/amazon/amazon_alexa.tsv&#39;</span><span class="p">,</span> <span class="n">sep</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\t</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[67]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">alexa_df</span><span class="p">,</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">alexa_countcetorizer</span><span class="o">.</span><span class="n">toarray</span><span class="p">())],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[68]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[68]:</div>



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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>...</th>
      <th>4034</th>
      <th>4035</th>
      <th>4036</th>
      <th>4037</th>
      <th>4038</th>
      <th>4039</th>
      <th>4040</th>
      <th>4041</th>
      <th>4042</th>
      <th>4043</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>31-Jul-18</td>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Music</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I received the echo as a gift. I needed anothe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>31-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Without having a cellphone, I cannot use many ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I think this is the 5th one I've purchased. I'...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>looks great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Love it! I’ve listened to songs I haven’t hear...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I sent it to my 85 year old Dad, and he talks ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it! Learning knew things with it eveyda...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>I purchased this for my mother who is having k...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love, Love, Love!!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>Just what I expected....</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I love it, wife hates it.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Really happy with this purchase.  Great speake...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>We have only been using Alexa for a couple of ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love the size of the 2nd generation echo. S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I liked the original Echo. This is the same bu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love the Echo and how good the music sounds pl...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love Alexa! We use her to play music, play ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Have only had it set up for a few days. Still ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it. It plays my sleep sounds immediatel...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I got a second unit for the bedroom, I was exp...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Amazing product</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love my Echo. It's easy to operate, loads of...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Sounds great!! Love them!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Fun item to play with and get used to using.  ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Just like the other one</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I like the hands free operation vs the Tap. We...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I dislike that it confuses my requests all the...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3123</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Love my Alexa! Actually have 3 throughout the ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>This product is easy to use and very entertain...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>works great but speaker is not the good for mu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Outstanding product - easy to use.  works great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>We have six of these throughout our home and t...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Use the product for music and it’s great!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Easy to set-up and to use.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>It works great!!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I like having more Alexa devices in my house a...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>PHENOMENAL</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I loved it does exactly what it says</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I used it to control my smart home devices. Wo...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Very convenient</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Este producto llegó y a la semana se quedó sin...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Easy to set up Ready to use in minutes.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Barry</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>My three year old loves it.  Good for doing ba...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Awesome device wish I bought one ages ago.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>love it</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Perfect for kids, adults and everyone in betwe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Listening to music, searching locations, check...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I do love these things, i have them running my...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Only complaint I have is that the sound qualit...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>4</td>
      <td>29-Jul-18</td>
      <td>Black  Dot</td>
      <td>Good</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 4049 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[70]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;length&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;verified_reviews&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="nb">len</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">alexa_df</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[71]:</div>



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
      <th>rating</th>
      <th>date</th>
      <th>variation</th>
      <th>verified_reviews</th>
      <th>feedback</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>...</th>
      <th>4035</th>
      <th>4036</th>
      <th>4037</th>
      <th>4038</th>
      <th>4039</th>
      <th>4040</th>
      <th>4041</th>
      <th>4042</th>
      <th>4043</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love my Echo!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Loved it!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>31-Jul-18</td>
      <td>Walnut Finish</td>
      <td>Sometimes while playing a game, you can answer...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I have had a lot of fun with this thing. My 4 ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Music</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I received the echo as a gift. I needed anothe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>31-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Without having a cellphone, I cannot use many ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>31-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I think this is the 5th one I've purchased. I'...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>221</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>looks great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Love it! I’ve listened to songs I haven’t hear...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>114</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I sent it to my 85 year old Dad, and he talks ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it! Learning knew things with it eveyda...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>169</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>I purchased this for my mother who is having k...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>290</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love, Love, Love!!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Oak Finish</td>
      <td>Just what I expected....</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>I love it, wife hates it.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Really happy with this purchase.  Great speake...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>We have only been using Alexa for a couple of ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>216</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love the size of the 2nd generation echo. S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I liked the original Echo. This is the same bu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>451</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Love the Echo and how good the music sounds pl...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>246</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>We love Alexa! We use her to play music, play ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>385</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Heather Gray Fabric</td>
      <td>Have only had it set up for a few days. Still ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love it. It plays my sleep sounds immediatel...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>I got a second unit for the bedroom, I was exp...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>179</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Sandstone Fabric</td>
      <td>Amazing product</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>I love my Echo. It's easy to operate, loads of...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Sounds great!! Love them!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Fun item to play with and get used to using.  ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>133</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Charcoal Fabric</td>
      <td>Just like the other one</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I like the hands free operation vs the Tap. We...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>84</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I dislike that it confuses my requests all the...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3123</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Love my Alexa! Actually have 3 throughout the ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>128</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>This product is easy to use and very entertain...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>works great but speaker is not the good for mu...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Outstanding product - easy to use.  works great</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
    </tr>
    <tr>
      <th>3129</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>We have six of these throughout our home and t...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>352</td>
    </tr>
    <tr>
      <th>3130</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Use the product for music and it’s great!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3131</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Easy to set-up and to use.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>It works great!!</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I like having more Alexa devices in my house a...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>102</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>PHENOMENAL</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>I loved it does exactly what it says</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I used it to control my smart home devices. Wo...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Very convenient</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Este producto llegó y a la semana se quedó sin...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3139</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Easy to set up Ready to use in minutes.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Barry</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>3</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3142</th>
      <td>4</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>My three year old loves it.  Good for doing ba...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
    </tr>
    <tr>
      <th>3143</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Awesome device wish I bought one ages ago.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
    </tr>
    <tr>
      <th>3144</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>love it</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3145</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Perfect for kids, adults and everyone in betwe...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3146</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>Listening to music, searching locations, check...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>135</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>Black  Dot</td>
      <td>I do love these things, i have them running my...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>441</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>5</td>
      <td>30-Jul-18</td>
      <td>White  Dot</td>
      <td>Only complaint I have is that the sound qualit...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>380</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>4</td>
      <td>29-Jul-18</td>
      <td>Black  Dot</td>
      <td>Good</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 4050 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[72]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;rating&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;date&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;variation&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;verified_reviews&#39;</span><span class="p">,</span>
                   <span class="s1">&#39;feedback&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[73]:</div>
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

<div class="prompt output_prompt">Out[73]:</div>



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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>4035</th>
      <th>4036</th>
      <th>4037</th>
      <th>4038</th>
      <th>4039</th>
      <th>4040</th>
      <th>4041</th>
      <th>4042</th>
      <th>4043</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4045 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[74]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">alexa_df</span><span class="p">[</span><span class="s1">&#39;feedback&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[76]:</div>
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

<div class="prompt output_prompt">Out[76]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    1
1    1
2    1
3    1
4    1
Name: feedback, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[77]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Create train Test Split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span>
                                                    <span class="n">y</span><span class="p">,</span>
                                                    <span class="n">test_size</span><span class="o">=</span><span class="mf">0.20</span><span class="p">,</span>
                                                    <span class="n">random_state</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                                                    <span class="n">stratify</span><span class="o">=</span><span class="n">y</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[78]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">classification_report</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[79]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">randomforest_classifier</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">100</span><span class="p">,</span> <span class="n">criterion</span> <span class="o">=</span><span class="s1">&#39;entropy&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[80]:</div>
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

<div class="prompt output_prompt">Out[80]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>RandomForestClassifier(bootstrap=True, class_weight=None, criterion=&#39;entropy&#39;,
            max_depth=None, max_features=&#39;auto&#39;, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[81]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_predict</span> <span class="o">=</span> <span class="n">randomforest_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[82]:</div>
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
<div class="prompt input_prompt">In&nbsp;[83]:</div>
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

<div class="prompt output_prompt">Out[83]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a3c80c400&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWAAAAD8CAYAAABJsn7AAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAFOtJREFUeJzt3XucV3Wdx/HXZwa8poCI3A3d2LyspUlGeVnNHpq2ia55
6SYVNWU3qbbNtNpHZatZaplFoaCoeSHThRAvhLqWN0RzrXQL8sYIQshFF0iZme/+MUcaZZj5jfOb
+c7v+Hr6OI8553vOnPPlEb799jnfc06klJAk9b663B2QpNcqA1iSMjGAJSkTA1iSMjGAJSkTA1iS
MjGAJSkTA1iSMjGAJSmTfj19ge23G+OjdtrMDlttm7sL6oOeWfNodPccG1c+VnHm9N95925frzsc
AUtSJj0+ApakXtXSnLsHFTOAJZVLc1PuHlTMAJZUKim15O5CxQxgSeXSYgBLUh6OgCUpE2/CSVIm
joAlKY/kLAhJysSbcJKUiSUIScrEm3CSlIkjYEnKxJtwkpSJN+EkKY+UrAFLUh7WgCUpE0sQkpSJ
I2BJyqR5Y+4eVMwAllQuliAkKRNLEJKUiSNgScrEAJakPFIN3YSry90BSaqq1FL50omIeCIifh8R
D0XEwqJtp4iYFxGLip+DivaIiAsjYnFEPBwRb+ns/AawpHJpaal8qcxhKaV9U0rjiu3TgfkppbHA
/GIb4ChgbLE0AFM6O7EBLKlcqjgC3oIJwIxifQZwbJv2y1Ore4GBETG8oxMZwJLKpboj4ATcGhEP
RERD0TY0pbQMoPi5S9E+EljS5ncbi7Yt8iacpHLpwsi2CNWGNk1TU0pT22wfmFJaGhG7APMi4n87
Ol17veno+gawpHJpqvyF7EXYTu1g/9Li54qIuAE4AFgeEcNTSsuKEsOK4vBGYHSbXx8FLO3o+pYg
JJVLlWrAEbF9ROzw0jpwBPAHYDYwsThsIjCrWJ8NnFLMhhgPrH2pVLEljoAllUv1HsQYCtwQEdCa
lVellG6OiPuBmRExCXgKOKE4fi5wNLAYWA98tLMLGMCSyqVK74JIKT0GvLmd9meBw9tpT8BnunIN
A1hSufgosiRl4tvQJCmTLsyCyM0AllQuqcOpt32KASypXKwBS1ImBrAkZeJNOEnKpLk5dw8qZgBL
KhdLEJKUiQEsSZlYA5akPFKL84AlKQ9LEJKUibMgJCmTGhoB+0WMHjLlp+fyxBMLuf/+Wza1nXHm
ZBYtvpd77p3LPffO5cgjD83XQWWx9dZbcdP8a5n/2xv473t+xZe/+tlN+07/2mnctfAm7rxvDpM+
+aGMvaxx1f8sfY9xBNxDrrziOn720xlcfPH5L2u/6EfT+OEPL87UK+X2wgsvcvwxH2X9uvX069eP
2Tdfyfx5v+Ef37g7I0cN56C3Hk1KiZ133il3V2tXmV7GExF70Pq9+5G0fuFzKTA7pfRoD/etpt11
1wJ23XVU7m6oD1q/bj0A/fv3o1///qSUmPixkzn1418mFeGxcuWqnF2sbX1gZFupDksQEfEV4Bpa
P7e8ALi/WL86Ik7v+e6Vzyc/NZH77ruJKT89l4EDd8zdHWVQV1fHr39zPX9Y9FvuvP1ufvfAw7x+
t12Z8K9Hccvtv+CqX/yM3XZ/fe5u1q6WVPmSWWc14EnAW1NK56SUriyWc2j9NPOknu9euVxy8ZX8
096HMH780TzzzArOPudrubukDFpaWnjXwf/Kfnsfxn7778Mee45l663688ILL3DkYSdw5eXXccFF
Z+XuZu1qbq58yayzAG4BRrTTPrzY166IaIiIhRGxsKnp+e70r1RWrFhJS0sLKSUunX4N4/bf7Ht/
eg15bu3z3P3bBRx2+EEsXbqcG2ffCsDcX81jr73fmLl3tSu1tFS85NZZAE8G5kfETRExtVhuBuYD
p23pl1JKU1NK41JK4/r126Ga/a1pw4YN2bR+zDFH8sdH/pyxN8ph8OBB7Dig9d+JbbbZmoP/+e0s
XvQ4N984n4MOGQ/AOw56K4/95YmMvaxxNVSC6PAmXErp5oj4R1pLDiNprf82AvenlPKP3/uwyy67
kIMPGc/gwYP486J7OOusCzjk4PG86U17kVLiyaca+fznzsjdTfWyXYYN4cIpZ1NfX09d1DH7v25m
3i13cN+9D/CTqd+j4dSJrFu3ni9+/uu5u1q7auhdEJF6eMrG9tuNyf+fGfU5O2y1be4uqA96Zs2j
0d1zrPvWByvOnO2/8fNuX687nAcsqVyaauf/nBvAksqlhkoQBrCkcukDN9cqZQBLKpW+ML2sUgaw
pHKpoRGwb0OTVC5VngccEfUR8buImFNs7xYR90XEooi4NiK2Ktq3LrYXF/vHdHZuA1hSuVT/UeTT
gLYvH/sucEFKaSywmr+/lmESsDql9AbgguK4DhnAkkoltaSKl85ExCjgPcAlxXYA7wSuKw6ZARxb
rE8otin2H14cv0UGsKRyqW4J4gfAv/P3d98MBtaklJqK7UZanxKm+LkEoNi/tjh+iwxgSeXShS9i
tH1xWLE0vHSaiPgXYEVK6YE2Z29vRJsq2NcuZ0FIKpcuzIJIKU0Fpm5h94HAMRFxNLANsCOtI+KB
EdGvGOWOovUjFdA6Gh4NNEZEP2AA0OGb9R0BSyqXKpUgUkpfTSmNSimNAU4GbkspfRC4HXhfcdhE
YFaxPrvYpth/W+rkZTuOgCWVSmru8QcxvgJcExFnAb8DphXt04ArImIxrSPfkzs7kQEsqVx64EGM
lNIdwB3F+mO0vqL3lcf8DTihK+c1gCWVSiXTy/oKA1hSuRjAkpRJ7byLxwCWVC6pqXYS2ACWVC61
k78GsKRy8SacJOXiCFiS8nAELEm5OAKWpDw2vSiyBhjAkkqlhr5KbwBLKhkDWJLycAQsSZkYwJKU
SWru8DuYfYoBLKlUHAFLUiapxRGwJGXhCFiSMknJEbAkZeEIWJIyaXEWhCTl4U04ScrEAJakTFLt
vA7YAJZULo6AJSkTp6FJUibNzoKQpDxqaQRcl7sDklRNqSUqXjoSEdtExIKI+J+I+GNEfLNo3y0i
7ouIRRFxbURsVbRvXWwvLvaP6ayvBrCkUkmp8qUTLwDvTCm9GdgXeHdEjAe+C1yQUhoLrAYmFcdP
AlanlN4AXFAc1yEDWFKpVGsEnFr9X7HZv1gS8E7guqJ9BnBssT6h2KbYf3hEdHgRA1hSqTS31FW8
dCYi6iPiIWAFMA/4C7AmpU3fXm4ERhbrI4ElAMX+tcDgjs5vAEsqla6UICKiISIWtlkaXn6u1JxS
2hcYBRwA7NneJYuf7Y12Oyx0OAtCUqm0dGEWREppKjC1guPWRMQdwHhgYET0K0a5o4ClxWGNwGig
MSL6AQOAVR2d1xGwpFJJKSpeOhIRQyJiYLG+LfAu4FHgduB9xWETgVnF+uxim2L/bSl1fKvPEbCk
UqniuyCGAzMiop7WwerMlNKciHgEuCYizgJ+B0wrjp8GXBERi2kd+Z7c2QV6PIBfaNrY05dQDVrz
1G25u6CS6koJoiMppYeB/dppf4zWevAr2/8GnNCVazgCllQqlcxu6CsMYEmlUkNvozSAJZVLtUoQ
vcEAllQqtfQyHgNYUqnU0EeRDWBJ5ZLafSCtbzKAJZVKkyUIScrDEbAkZWINWJIycQQsSZk4Apak
TJodAUtSHp18aahPMYAllUqLI2BJysOX8UhSJt6Ek6RMWjr+EnyfYgBLKpXm3B3oAgNYUqk4C0KS
MnEWhCRl4iwIScrEEoQkZeI0NEnKpNkRsCTl4QhYkjIxgCUpkxr6JJwBLKlcHAFLUia19ChyXe4O
SFI1tUTlS0ciYnRE3B4Rj0bEHyPitKJ9p4iYFxGLip+DivaIiAsjYnFEPBwRb+msrwawpFJp6cLS
iSbgSymlPYHxwGciYi/gdGB+SmksML/YBjgKGFssDcCUzi5gAEsqlWoFcEppWUrpwWL9eeBRYCQw
AZhRHDYDOLZYnwBcnlrdCwyMiOEdXcMAllQqqQtLpSJiDLAfcB8wNKW0DFpDGtilOGwksKTNrzUW
bVtkAEsqla7UgCOiISIWtlkaXnm+iHgd8EtgckrpuQ4u3V5VucOcdxaEpFLpyiyIlNJUYOqW9kdE
f1rD9+cppeuL5uURMTyltKwoMawo2huB0W1+fRSwtKPrOwKWVCotpIqXjkREANOAR1NK57fZNRuY
WKxPBGa1aT+lmA0xHlj7UqliSxwBSyqVKj6IcSDwYeD3EfFQ0XYGcA4wMyImAU8BJxT75gJHA4uB
9cBHO7uAASypVKr1QvaU0m9pv64LcHg7xyfgM125hgEsqVR8FFmSMmmK2vkokQEsqVRqJ34NYEkl
YwlCkjLpbHpZX2IASyqV2olfA1hSyViCkKRMmmtoDGwASyoVR8CSlElyBCxJeTgC1maOPOJQzj//
W9TX1TH90qs593s/zt0lvUpHHD+R7bfbjrq6Ourr65k5/cKX7X/+/9Zx+rfOZdnyv9Lc1MxHPnA8
x73niG5dc+1zz/Olr5/N0meWM2LYUM779lcZsOMOzLnlNqb9/BcAbLfttnz93z7LHmN379a1al0t
TUPzdZS9oK6ujgt/+B3+5b0fYp83H8ZJJx3LnnuOzd0tdcP0H53DL2f8eLPwBbj6l7/iH8bsyvUz
fsKlF32X7/3oYjZu3FjReRc8+DBnnnXeZu2XXDGT8eP2Ze610xg/bl+mXTkTgJEjhnHZRedyw+VT
+NRH3s83z928P681PfFFjJ5iAPeCA966H3/5yxM8/vhTbNy4kZkzZ3HMe4/M3S31kIhg3foNpJRY
v+FvDNhxB+rr6wGY/vPrOGnS5znulFO56JIrKj7n7b+5hwlHvQuACUe9i9vuvAeA/fbZiwE77gDA
m/beg+UrVlb5T1N7mkgVL7m96gCOiE7fdalWI0YOY0nj31+M3/j0MkaMGJaxR+qOiKDhC2dy4sc+
xy9mzd1s/weOfy+PPbGEwyZ8kONOOZXTJ3+Kuro67rrvAZ5qfJprLvkhv7zsxzzyp8UsfOj3FV3z
2dVrGLLzTgAM2XknVq1Zu9kx18+5hYPGj+veH64EUhf+ya07NeBvApe2t6P4rlIDQNQPoK5u+25c
pva1vlj/5VpfHapadMWU89hlyGCeXb2GT0w+g91eP5px++6zaf9dCx5gj7G7M/1H57Dk6WV8YvIZ
7P/mvbn7/ge5e8GDvO8jnwVg/YYNPLlkKeP23Yf3f2IyL764kfUbNrD2uec5fmLra2W/+OmPceDb
9u+0Twse+B+un3MrV0z5fs/8oWtIaW7CRcTDW9oFDN3S77X9zlK/rUa+5pPm6cZljB41YtP2qJHD
WbZsecYeqTt2GTIYgMGDBnL4Ie/g94/86WUBfMON8/j4h04kIth11AhGDh/G4082QoKPf/gkTjz2
6M3OefXFPwBaa8Cz5s7jO1/70sv2Dx40kL+uXMWQnXfirytXsdPAAZv2/Wnx43zjnB/w0/O+zcAB
O/bEH7mm9IWRbaU6K0EMBU4B3tvO8mzPdq087l/4EG94w26MGTOa/v37c+KJE/jVnFtzd0uvwvoN
f2PduvWb1u9e8CBjdx/zsmOGDx3CvQ+0fsFm5arVPPFUI6NGDOMdB7yFG268lfXrNwCw/K8reXb1
moque+hB45l1068BmHXTrzns4LcDsOyZFUw+49uc/Y0vM2bXUdX4I9a8li4suXVWgpgDvC6l9NAr
d0TEHT3SoxJqbm7mtMlfY+6NV1FfV8dlM67lkUf+nLtbehWeXbWa0874NgDNTc0cfcShHDR+HNfe
cCMAJx33Hj71kQ9w5nfO47gPn0pKiS98+mMMGjiAA9+2P489uYQPfvKLAGy37Tac/Y0vM3jQwE6v
+/EPn8iXvv6fXD/nFoYPHcL5Z50JwJRLr2Ltc89z1vdbpzW2Ny3utaa5hsp70dO1SEsQas+Gpb/J
3QX1Qf133n1L32Cr2Adef1zFmXPVkzd0+3rd4YMYkkqllmrABrCkUukLtd1KGcCSSqWWHkU2gCWV
iiUIScqklmZBGMCSSsUShCRl4k04ScrEGrAkZVJLJQjfByypVFJKFS+diYjpEbEiIv7Qpm2niJgX
EYuKn4OK9oiICyNicUQ8HBFv6ez8BrCkUmkmVbxU4DLg3a9oOx2Yn1IaC8wvtgGOAsYWSwMwpbOT
G8CSSqWFVPHSmZTSncCqVzRPAGYU6zOAY9u0X55a3QsMjIjhHZ3fAJZUKl0pQUREQ0QsbLM0VHCJ
oSmlZcW1lgG7FO0jgSVtjmss2rbIm3CSSqUrN+HafjyiCtp7s1qHnXEELKlUeuGbcMtfKi0UP1cU
7Y3A6DbHjQKW0gEDWFKpNKdU8fIqzQYmFusTgVlt2k8pZkOMB9a+VKrYEksQkkqlmvOAI+Jq4FBg
54hoBP4DOAeYGRGTgKeAE4rD5wJHA4uB9UCnX443gCWVSjUDOKX0/i3sOrydYxPwma6c3wCWVCo9
/Zm1ajKAJZVKLT2KbABLKhVfxiNJmTSn2nkhpQEsqVSsAUtSJtaAJSkTa8CSlEmLJQhJysMRsCRl
4iwIScrEEoQkZWIJQpIycQQsSZk4ApakTJpTc+4uVMwAllQqPoosSZn4KLIkZeIIWJIycRaEJGXi
LAhJysRHkSUpE2vAkpSJNWBJysQRsCRl4jxgScrEEbAkZeIsCEnKpJZuwtXl7oAkVVNKqeKlMxHx
7oj4U0QsjojTq91XA1hSqaQu/NORiKgHfgwcBewFvD8i9qpmXw1gSaVSxRHwAcDilNJjKaUXgWuA
CdXsqzVgSaVSxRrwSGBJm+1G4G3VOjn0QgA3vfh09PQ1akVENKSUpubuh/oW/15UV1cyJyIagIY2
TVPb/G/R3nmqeofPEkTvauj8EL0G+fcik5TS1JTSuDZL2/8QNgKj22yPApZW8/oGsCS1735gbETs
FhFbAScDs6t5AWvAktSOlFJTRHwWuAWoB6anlP5YzWsYwL3LOp/a49+LPiqlNBeY21Pnj1p6blqS
ysQasCRlYgD3kp5+pFG1JyKmR8SKiPhD7r4oDwO4F/TGI42qSZcB787dCeVjAPeOHn+kUbUnpXQn
sCp3P5SPAdw72nukcWSmvkjqIwzg3tHjjzRKqj0GcO/o8UcaJdUeA7h39PgjjZJqjwHcC1JKTcBL
jzQ+Csys9iONqj0RcTVwD/DGiGiMiEm5+6Te5ZNwkpSJI2BJysQAlqRMDGBJysQAlqRMDGBJysQA
lqRMDGBJysQAlqRM/h98i9HaAU0YwAAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[84]:</div>
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

          0       1.00      0.29      0.45        51
          1       0.94      1.00      0.97       579

avg / total       0.95      0.94      0.93       630

</pre>
</div>
</div>

</div>
</div>

</div>