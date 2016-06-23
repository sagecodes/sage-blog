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

            <span class="post_tag">{{ post.tags | join: ', ' }}</span>
            <br>
          <a href="{{post.url | prepend: site.baseurl }}">Demo</a>
          <a href="{{post.url | prepend: site.baseurl }}">Github</a>

          </div>
        </h2>
      </li>
    {% endfor %}
  </ul>
</div>