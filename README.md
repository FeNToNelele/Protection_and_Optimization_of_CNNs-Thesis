# :hungary:
Ez a repository tartalmazza szakdolgozatom programját, melynek címe: **Szoftverfejlesztés Deep Learning rendszerekben - A neurális hálózatok védelme**

A repo több `TensorFlow`-ban készült konvolúciós modellt mutat be. Céljuk annak bemutatása, hogy milyen módszerekkel lehet a neurális hálózatokat robosztusabbá tenni, azaz hogy érjük el, hogy zajos vagy megtévesztő képekkel szemben is helyesen klasszifikáljon.

A modelleket egy `Tkinter` keretrendszerben készült grafikus felülettel lehet könnyen kipróbálni. A teszt képekre opcionálisan zajcsökkentés helyezhető, így interaktívan bevonva a felhasználót a kutatói munkába. 

# Előkövetelmény
- Anaconda Prompt: a virtuális környezet felállításához.

## Virtuális környezet telepítése
1. Anaconda Prompt-on belül lépjünk a repo gyökérkönyvtárába.
2. Futtassuk a `conda env create -f ./venv/venv.yaml` parancsot. Ekkor létrejön theisenv néven a virtuális környezet

## Használat
1. Anaconda Prompt-on belül lépjünk a repository gyökérkönyvtárába.
2. Indítsuk a GUI-t `start.bat` paranccsal.

# :uk:
This repository contains the project for my thesis called **Software Development in Deep Learning Systems - Protection of Neural Networks**

This repo has multiple CNNs made in `TensorFlow`. Their goal is to represent how we can achieve robustness in neural networks, so it still classifies correctly when they meet with noisy or misleading pictures.

You can try them out with a GUI made in `Tkinter`. Noise reduction can be applied to your own images optinally, so you are also a part of this research.

# Prerequisite
- Anaconda Prompt: to create virtual environment.

## Installation of virtual environment
1. In Anaconda Prompt navigate to the root of the repo.
2. Run `conda env create -f ./venv/venv.yaml`. The virtual environment is created under name theisenv.

## Usage
1. In Anaconda Prompt navigate to the root of the repo.
2. Run GUI with `start.bat`.
