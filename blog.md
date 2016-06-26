---
layout: default
title: Blog
permalink: /blog/
---

<div class="home">

  <h1 class="page-heading">Posts:</h1>

  <ul class="post-list">
    {% for post in site.categories.post %}
      <li>
        <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>

        <h2>
          <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
          <div class="post_meta">
            <span class="post_author">{{post.author}}</span>
            <span class="post_sep"> | </span>
            <span class="post_date">{{ post.date | date: "%b %-d, %Y" }}</span>
            <span class="post_sep"> | </span>
            {% if post.tags %}
              {% for tag in post.tags %}
                <span class="post_tag"><a href="{{ site.baseurl }}{{ site.tag_page }}#{{ tag | slugify }}" class="post-tag">{{ tag }}</a></span>
              {% endfor %}
            {% endif %}


            <span class="post_excerpt"> {{ post.excerpt }} </span>


            <br>
          <a class="btn" href="{{post.url | prepend: site.baseurl }}"> Read more</a>

          </div>
        </h2>
      </li>
    {% endfor %}
  </ul>

  <p class="rss-subscribe">subscribe <a href="{{ "/feed.xml" | prepend: site.baseurl }}">via RSS</a></p>

</div>
