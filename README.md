# :hungary:
Ez a repository tartalmazza szakdolgozatom programját, melynek címe: **Konvolúciós neurális hálózatok védelme és optimalizálása**

A repo több `TensorFlow`-ban készült konvolúciós modellt mutat be. Céljuk annak bemutatása, hogy milyen módszerekkel lehet a neurális hálózatokat robosztusabbá tenni, azaz hogy érjük el, hogy zajos vagy megtévesztő képekkel szemben is helyesen klasszifikáljon.

A modelleket egy `Tkinter` keretrendszerben készült grafikus felülettel lehet könnyen kipróbálni. A teszt képekre opcionálisan zajcsökkentés helyezhető, így interaktívan bevonva a felhasználót a kutatói munkába. 

# Előkövetelmény
- Anaconda Prompt: a virtuális környezet felállításához.

## Virtuális környezet telepítése
1. Anaconda Prompt-on belül lépjünk a repository gyökérkönyvtárába.
2. Futtassuk a `conda env create -f ./venv/venv.yaml` parancsot. Ekkor létrejön `theisenv` néven a virtuális környezet

## Használat
1. Anaconda Prompt-on belül lépjünk a repository gyökérkönyvtárába.
2. Indítsuk a GUI-t `start.bat` paranccsal.

# :uk:
This repository contains the project for my thesis titled **Protection and Optimization of Convolutional Neural Networks**

The repository has multiple CNNs made in `TensorFlow`. The purpose of these CNNS is to represent how robustness can be achieved in neural networks, enabling them to correctly classify images even in the presence of noise or misleading visual cues.

You can try them out with a GUI made in `Tkinter`. Optionally, you can also apply noise reduction, making you a part of the research.

# Prerequisites
- Anaconda Prompt: for creating the virtual environment.

## Installation of virtual environment
1. In Anaconda Prompt navigate to the root of the repository.
2. Run `conda env create -f ./venv/venv.yaml` to create a virtual environment named `theisenv`.

## Usage
1. In Anaconda Prompt navigate to the root of the repository.
2. Run the GUI using `start.bat`.
