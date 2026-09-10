# **Welcome to ShrinkAI's documentation**

<div style="display: flex; gap: 2.5rem; align-items: flex-start; margin-bottom: 2rem; flex-wrap: wrap;">

  <div style="flex: 0 0 200px; display: flex; flex-direction: column; align-items: center; text-align: center;">
    <img src="assets/shrinkai-logo-nobackground.png" alt="ShrinkAI Logo" style="width: 750; height: auto; margin-bottom: 1rem;" />
    <div style="display: flex; flex-direction: column; gap: 0.4rem; align-items: center;">
      <a href="https://pypi.org/project/shrinkai/">
        <img src="https://img.shields.io/pypi/v/shrinkai?color=2860BE&style=flat-square" alt="PyPI version" />
      </a>
      <img src="https://img.shields.io/pypi/pyversions/shrinkai?color=1595BC&style=flat-square" alt="Python versions" />
      <a href="https://github.com/elouanzer/shrinkai">
        <img src="https://img.shields.io/badge/GitHub-Repository-151922?logo=github&style=flat-square" alt="GitHub Repo" />
      </a>
      <img src="https://img.shields.io/github/license/elouanzer/shrinkai?color=7B8494&style=flat-square" alt="License" />
    </div>
  </div>

  <div style="flex: 1; min-width: 280px;">
    <div class="md-typeset" style="flex: 1; min-width: 320px;">
    <p>
      <strong>ShrinkAI</strong> is a package for reducing neural networks size, making them ideal to run on small devices such as smartphones or robots, and for speeding up inference, particularly interesting for edge AI.
    </p>
    <p>
      This package includes, among other things, numerous distillation and compression techniques, all wrapped in an API that is easy to use for users familiar with PyTorch.
    </p>
  </div>
  </div>
</div>


## **Installation**

ShrinkAI is on pypi and can be installed with the following command:
```bash
pip install shrinkai
```

The package is compatible with Python 3.11+, and depends `torch`, `torchvision`, `rich`, `tqdm` and `psutil`.

## **Contact**

You can report an [issue](https://github.com/elouanzer/shrinkai/issues) directly in GitHub.

## **License**

[MIT License](https://github.com/elouanzer/shrinkai/blob/main/LICENSE)