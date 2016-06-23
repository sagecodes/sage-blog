---
layout: default
title: Projects
permalink: /projects/
---

<div class="home">

  <h1 class="page-heading">Projects:</h1>

  <ul class="post-list">
    {% for post in site.categories.project %}
      <li>
        <h2>
          <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
          <div class="post_meta">
            <span class="post_excerpt"> {{ post.excerpt }} </span>

            {% if post.tags %}
              {% for tag in post.tags %}
                <span class="post_tag"><a href="{{ site.baseurl }}{{ site.tag_page }}#{{ tag | slugify }}" class="post-tag">{{ tag }}</a></span>
              {% endfor %}
            {% endif %}
            <br>
          <a href="{{post.url | prepend: site.baseurl }}">Demo</a>
          <a href="{{post.url | prepend: site.baseurl }}">Github</a>

          </div>
        </h2>
      </li>
    {% endfor %}
  </ul>
</div>