# smol-tools

A collection of lightweight AI-powered tools built with LLaMA.cpp and small language models. These tools are designed to run locally on your machine without requiring expensive GPU resources. They can also run offline, without any internet connection.

## Features

### SmolSummarizer
- Quick text summarization using SmolLM2-1.7B Instruct
- Maintains key points while providing concise summaries
- Able to reply to follow-up questions

### SmolRewriter
- Rewrites text to be more professional and approachable
- Maintains the original message's intent and key points
- Perfect for email and message drafting

### SmolAgent
- An AI agent that can perform various tasks through tool integration
- Built-in tools include:
  - Weather lookup
  - Random number generation
  - Current time
  - Web browser control
- Extensible tool system for adding new capabilities

## Installation

1. Clone the repository:

```bash
git clone https://github.com/huggingface/smollm.git
cd smollm/smol_tools
```

2. Install dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you're on mac, and you don't have tkinter installed, you can install it with:

```bash
brew install python-tk@3.11
```

For linux, you can install it with:

```bash
sudo apt-get install python3-tk
```

On Windows, when you install python you need to check the option to also install the tkinter library.

## Usage

### GUI Demo
Run the Tkinter-based demo application:

```bash
python demo_tkinter.py
```

The demo provides a user-friendly interface with the following shortcuts:
- `F1`: Open SmolDraft interface
- `F2`: Summarize selected text
- `F5`: Open SmolChat interface
- `F10`: Open SmolAgent interface

### Programmatic Usage

```python
from smol_tools.summarizer import SmolSummarizer
from smol_tools.rewriter import SmolRewriter
from smol_tools.agent import SmolToolAgent
# Initialize tools
summarizer = SmolSummarizer()
rewriter = SmolRewriter()
agent = SmolToolAgent()
# Generate a summary
for summary in summarizer.process("Your text here"):
    print(summary)
# Rewrite text
for improved in rewriter.process("Your text here"):
    print(improved)
# Use the agent
for response in agent.process("What's the weather in London?"):
    print(response)
```


## Models

The tools use the following models:
- SmolSummarizer: SmolLM2-1.7B Instruct

All models are quantized to 16-bit floating-point (F16) for efficient inference. Training was done on BF16, but in our tests, this format provides slower inference on Mac M-series chips.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.