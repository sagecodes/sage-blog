---
layout: post
title:  "Git & Github commands"
permalink: /git-github-commands/
date:   2018-06-11
categories: post
tags: git
author: Sage Elliott
published: true
---

I forget a lot of things including git commands...
Here is a reference I made mostly for myself or when setting people with git & github for the first time:

## The usual suspects
commands often used:

- `git add .` add all unstaged files
- `git commit -m"message here"` commit message
- `git push origin featureBranch` push to remote branch
- `git stash` save changes not added and remove them from branch
- `git stash pop` puts the "saved" changes back in branch/directory
- `git rebase -i origin/master` use `f` on commits to change to `fixup`(squash)
	- `r` to reword commit message
	- `i` to enter interactive mode in vim
esc to get back to where to save and exit `:wq` (enter)


## First time setup

### Config
- `git config --global user.email email@example.com`
- `git config --global user.name first last`

### repo setup
- `git init`
- `git add .` to add all files in directory or use `git add fileName` to add individual files
- `git commit -m"initial commit"`
- `git remote add origin github_url`
- `git push -u origin master`

## .gitignore
create file in directory called `.gitignore`

### File Contents
- `file_name` will not add file here to 
- `**/folderName` anything in `folderName` 
- `*.fileType` anything that is `fileType`

### Common contents in .gitignore:

`.DS_Store`