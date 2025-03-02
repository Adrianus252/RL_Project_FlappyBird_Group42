

# 1 Install Python 

## Version 3.9
- https://www.python.org/downloads/release/python-390/



# 2 Virtual Environment
```python
cd <projectDirectory> 
```
## 2.1 Create 

```python
py -3.9 -m venv flappy_env_3.9
```

## 2.2 Allow running scripts

```python
Set-ExecutionPolicy Unrestricted -Scope Process
```

## 2.2 Activate virtual env

```python
flappy_env_3.9\Scripts\activate
```

## 2.3 Deactivate Virtual Env
```python
deactivate
```

# 3 Installation

### 3.2 Upgrade Pip,Setuptools, Wheel
```python
# upgrade the pip, setuptools and wheel 
pip cache purge
python -m pip install --upgrade pip==24.0 setuptools wheel
```
- 3.9
- Installing collected packages: pip, setuptools, wheel
    - pip-24.0
    - setuptools-75.8.2
    - wheel-0.45.1

## 3.1 Install Flappy Bird Gym

```python
pip install flappy-bird-gym
```

```python
#alternative
pip install --no-cache-dir flappy-bird-gym
```
- Python 3.9 
Successfully installed 
    - Pillow-8.2.0
    - cloudpickle-1.6.0
    - flappy-bird-gym-0.3.0
    - gym-0.18.3
    - numpy-1.19.5
    - pygame-2.0.3
    - pyglet-1.5.15
    - scipy-1.13.1

```python
# for 3.9
pip install scipy==1.7.3
pip install numpy==1.22.4
```

## 3.3 Install Stable Baselines (not working with flappy bird gym)

```python
# install stable baseline 3
pip install stable-baselines3[extra]
```
- Installing collected packages: 
    - pytz
    - mpmath
    - farama-notifications
    - zipp
    - tzdata
    - typing-extensions
    - tensorboard-data-server
    - sympy
    - six
    - pyparsing
    - pygments
    - pygame
    - psutil
    - protobuf
    - pillow
    - packaging
    - numpy
    - networkx
    - mdurl
    - MarkupSafe,
    - kiwisolver
    - grpcio
    - fsspec
    - fonttools
    - filelock
    - cycler
    - colorama
    - cloudpickle
    - absl-py
    - werkzeug
    - tqdm
    - python-dateutil
    - opencv-python
    - markdown-it-py
    - jinja2
    - importlib-resources
    - importlib-metadata
    - contourpy, torch
    - rich
    - pandas
    - matplotlib
    - markdown
    - gymnasium
    - ale-py
    - tensorboard
    - stable-baselines3
- Successfully installed
    - MarkupSafe-3.0.2 
    - absl-py-2.1.0
    - ale-py-0.10.2
    - cloudpickle-3.1.1
    - colorama-0.4.6
    - contourpy-1.3.0
    - cycler-0.12.1
    - farama-notifications-0.0.4
    - filelock-3.17.0
    - fonttools-4.56.0
    - fsspec-2025.2.0
    - grpcio-1.70.0
    - gymnasium-1.0.0
    - importlib-metadata-8.6.1
    - importlib-resources-6.5.2
    - jinja2-3.1.5
    - kiwisolver-1.4.7
    - markdown-3.7
    - markdown-it-py-3.0.0
    - matplotlib-3.9.4
    - mdurl-0.1.2
    - mpmath-1.3.0 
    - networkx-3.2.1
    - numpy-2.0.2
    - opencv-python-4.11.0.86
    - packaging-24.2
    - pandas-2.2.3
    - pillow-11.1.0
    - protobuf-5.29.3
    - psutil-7.0.0
    - pygame-2.6.1
    - pygments-2.19.1
    - pyparsing-3.2.1
    - python-dateutil-2.9.0.post0 
    - pytz-2025.1 
    - rich-13.9.4 six-1.17.0 
    - stable-baselines3-2.5.0 
    - sympy-1.13.1 
    - tensorboard-2.19.0 
    - tensorboard-data-server-0.7.2 
    - torch-2.6.0 
    - tqdm-4.67.1 
    - typing-extensions-4.12.2 
    - tzdata-2025.1 
    - werkzeug-3.1.3 
    - zipp-3.21.0


## Specific Version if pip depedency resolver is not working
```python
pip install numpy==1.16.5
```

# 4 Reinforcement Library Skrl (not working with flappy bird gym)

```python
pip install skrl
```

- Successfully installed 
- MarkupSafe-3.0.2 
- absl-py-2.1.0 
- colorama-0.4.6 
- farama-notifications-0.0.4
- grpcio-1.70.0 
- gymnasium-1.1.0 
- importlib-metadata-8.6.1 
- markdown-3.7 numpy-2.0.2 
- packaging-24.2 
- protobuf-5.29.3 
- six-1.17.0 skrl-1.4.1 
- tensorboard-2.19.0 
- tensorboard-data-server-0.7.2 
- tqdm-4.67.1 
- typing-extensions-4.12.2 
- werkzeug-3.1.3 
- zipp-3.21.0

```python
pip install numpy==1.16.5
```

## Install Torch
```python
pip install torch torchvision torchaudio
```
- Successfully installed 
    - filelock-3.17.0 
    - fsspec-2025.2.0 
    - jinja2-3.1.5 
    - mpmath-1.3.0 
    - networkx-3.2.1 
    - sympy-1.13.1 
    - torch-2.6.0 
    - torchaudio-2.6.0 
    - torchvision-0.21.0

