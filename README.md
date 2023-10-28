# Neural Machine Translation

One of the most popular tasks in Natural Language Processing is translating sequence of text from one language to another. The **Sequence-to-Sequence** architecture is used to perform this task. In my graduate NLP course at the _University of Rochester_, I was introduced to this architecture and the popular attention mechanism. I wanted to independently study the seq2seq models and this repository contains code of my experiments. In particular, I implement the seq2seq architecture using
* Only Recurrent Neural Networks (RNNs)
* RNNs with attention mechanism
* Transformers architecture

![Python](https://img.shields.io/badge/Python-20232A?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/jupyter-20232A?style=for-the-badge&logo=jupyter)
![PyTorch](https://img.shields.io/badge/PyTorch-20232A?style=for-the-badge&logo=pytorch)
![TorchText](https://img.shields.io/badge/TorchText-20232A?style=for-the-badge&logo=pytorch)
![Matplotlib](https://img.shields.io/badge/matplotlib-20232A?style=for-the-badge&logo=matplotlib)

# Installing Dependencies
```python
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source ./venv/bin/activate
# Update pip
pip install --upgrade pip
# Install from requirements.txt
pip install -r requirements.txt
```

# Acknowledgement
1. My Natural Language Processing course instructor [Hangfeng He](https://hornhehhf.github.io/). 
2. [Dive into Deep Learning](https://d2l.ai/) authors and community