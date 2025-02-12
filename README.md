# LLM Robot

An opinionated robotics framework built on zenoh.

## Installation

```bash
pip install -e .
```

## Usage

To read from the zenoh network:

```bash
llm-robot read
```

To connect to a specific zenoh peer:

```bash
llm-robot read --connect tcp/192.168.1.1:7447
```

## Project Structure

- `src/llm_robot/`: Main package
  - `cli.py`: Command-line interface
  - `models/`: Data structure definitions
- `robots/`: Robot implementations
  - `small/`: Small robot package

## Development

This project uses `pyproject.toml` for dependency management. Make sure you have Python 3.8 or newer installed. 