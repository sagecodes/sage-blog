---
layout: post
title:  "Intro to Flyte Tasks & Workflows"
permalink: /intro-flyte-task-and-workflows/
date:   2023-03-27
categories: note
tags: mlops flyte workflow
author: Sage Elliott
published: true
---

** 
This post is a "note". I use notes to keep track of useful things for myself, but if you stumbled upon this page and find it useful, great!

Check out my blog posts for more in-depth content.
**

<br>

## Getting Started with Flyte: Simplify Your Workflows with Code Examples

Flyte is an open-source tool for orchestrating reliable and scalable workflows. If you're looking for a solution to simplify and streamline your data processing and machine learning tasks, you've come to the right place. In this blog, we'll introduce you to the basics of Flyte and provide code examples to help you hit the ground running. Let's dive in!

## Getting Started with Flyte:

### Flyte Installation:

To get started, first, install Flyte in a Python environment by running the following command:

```bash
pip install flytekit
```

## Creating a Simple Flyte Task:

Next, let's create a simple task using the Flyte SDK.

```python
from flytekit import task

@task
def hello_world(name: str) -> str:
    return f"Hello, {name}!"
```

Here, we've defined a simple `hello_world` function, decorated it with the `@task` decorator, and provided input and output type annotations.

## Registering and Running the Task:

Flyte tasks only takes keyword args

```
hello_world(name="Flyte")
```

You should see the output `Hello, Flyte!` printed on the console above.

## Creating a Workflow:

Now let's create a workflow that combines multiple tasks.

```python
from flytekit import task, workflow

@task
def add(a: int, b: int) -> int:
    return a + b

@task
def multiply(a: int, b: int) -> int:
    return a * b


@workflow
def my_workflow(x: int, y: int) -> int:
    addition_result = add(a=x, b=y)
    multiplication_result = multiply(a=x, b=y)
    added_results =  add(a=addition_result, b=multiplication_result)
    return added_results
```

run the workflow
```
my_workflow(x=3, y=4)
```
`>>> 19`

In this example, we've defined two tasks (add and multiply) and a workflow (my_workflow) that uses these tasks.

Running the Workflow:
To run the workflow, execute the following command:

You should see the output 25 printed on the console, which is the result of the combined tasks.

Conclusion:

Congratulations, you've successfully created and run a simple task and workflow using Flyte! With the basics under your belt, you're ready to explore more advanced features and build complex, scalable workflows. Stay tuned for more blog posts on Flyte's powerful capabilities!

Learn more about Flyte tasks and workflows in the Flyte docs:
https://docs.flyte.org/projects/cookbook/en/latest/getting_started/tasks_and_workflows.html