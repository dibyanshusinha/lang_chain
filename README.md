# Building a RAG application from scratch

This is a GPT Model that querying your own files for context and provides answers to your questions

## Setup

1. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ python chat.py
```

2. Rename a `.env_template` file with to `.env` and update the variables as required:

```bash
OPENAI_API_KEY='YOUR_OPENAI_API_KEY'

```