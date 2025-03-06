

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

# 3 Installation

## 3.1 Minimal Working Installation

```python
pip list | findstr "pip setuptools wheel"
#pip               20.2.3
#setuptools        49.2.1
pip install flappy-bird-gym
pip install scipy==1.7.3
pip install numpy==1.19.5
cd ..\run_flappy\
python .\flappy_run_human.py
cd ..\ppo\ 
pip install torch torchvision torchaudio
python train_myPPO.py
python test_myPPO.py
```