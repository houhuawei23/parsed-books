# Parsed Books

A collection of parsed technical books in Markdown format, organized by subject category. Each book directory contains structured markdown chapters, original images, and—where available—a `metadata.yaml` file with bibliographic information.

## Categories

| Category | Books | Index |
|----------|-------|-------|
| [AI](AI) | 1 | — |
| [Causal](Causal) | 5 | [List.md](Causal/List.md) |
| [CPC](CPC) | 0 | — |
| [Marx](Marx) | 5 | [List.md](Marx/List.md) |
| [Math](Math) | 1 | — |
| [ML](ML) | 6 | — |

## Books

### AI

- [AIMA-人工智能-现代方法-zh-4th](AI/AIMA-人工智能-现代方法-zh-4th)

### Causal Inference

Sorted by publication year:

- [Causation, Prediction, and Search: Second Edition](Causal/2021-Causation_Prediction_and_Search-2nd-Peter_Spirtes-Clark_Glymour) — Peter Spirtes, Clark Glymour, Richard Scheines (2001)
- [Causality: Models, Reasoning, and Inference](Causal/2009-Causality_2nd_Pearl) — Judea Pearl (2009)
- [Causal Inference in Statistics: A Primer](Causal/2016-Causal_Inference_in_Statistics-A_Primer_Jewell-Pearl) — Judea Pearl, Madelyn Glymour, Nicholas P. Jewell (2016)
- [The Book of Why: The New Science of Cause and Effect](Causal/2018-The_Book_of_Why-Judea_Pearl-Dana_Mackenzie) — Judea Pearl, Dana Mackenzie (2018)
- [Causal Inference: What If](Causal/2020-Causal-Inference-What-If-Hernán_MA-Robins_JM) — Miguel A. Hernán, James M. Robins (2020)

### Marx

- [中国马克思主义与当代-2024年版](Marx/中国马克思主义与当代-2024年版)
- [自然辩证法概论-2025版](Marx/自然辩证法概论-2025版)
- [资本论-中共中央马克思恩格斯列宁斯大林著作编译局-编译](Marx/资本论-中共中央马克思恩格斯列宁斯大林著作编译局-编译)
- [马克思恩格斯列宁哲学经典著作导读-第2版](Marx/马克思恩格斯列宁哲学经典著作导读-第2版)
- [马克思恩格斯列宁经典著作选读-2025年版](Marx/马克思恩格斯列宁经典著作选读-2025年版)

### Math

- [数理逻辑-王兵山](Math/数理逻辑-王兵山)

### Machine Learning

- [强化学习的数学原理-赵世钰](ML/强化学习的数学原理-赵世钰_parsed)
- [机器学习-Machine-Learning-周志华](ML/机器学习-Machine-Learning-周志华)
- [机器学习理论导引-周志华-王魏-高尉-张利军](ML/机器学习理论导引-周志华-王魏-高尉-张利军)
- [统计学习方法-第2版-李航](ML/统计学习方法-第2版-李航)
- [流形上的分析-J.R.曼克勒斯](ML/流形上的分析-J.R.曼克勒斯.pdf) (PDF)
- [高维数据的流形学习分析方法-李波](ML/高维数据的流形学习分析方法-李波.pdf) (PDF)

## Generate Book Indexes

The repository includes a Python script [`generate_list.py`](generate_list.py) that scans `metadata.yaml` files and generates a Markdown index (`List.md`) for each category.

### Requirements

- Python 3
- PyYAML (`pip install pyyaml`)

### Usage

```bash
# Generate indexes for every category under the script's root
python3 generate_list.py

# Generate an index for a specific category only
python3 generate_list.py Causal

# Preview what would be generated without writing files
python3 generate_list.py --all-categories --dry-run

# Use a custom output filename and title
python3 generate_list.py -o README.md -t "My Library"
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `target` | Target directory to index. If omitted, all categories are indexed. |
| `-o, --output` | Output filename (default: `List.md`). |
| `-t, --title` | Custom title for the index (default: derived from directory name). |
| `-a, --all-categories` | Generate an index for every category subdirectory of `target`. |
| `-r, --recursive` | Recursively scan for `metadata.yaml` files. |
| `-n, --dry-run` | Print what would be generated without writing files. |
| `-q, --quiet` | Suppress non-error output. |

## Purpose

This repository contains structured markdown versions of technical books for easier:

- Searching and referencing
- Note-taking and annotation
- Cross-linking between concepts
- Integration with knowledge management systems

## Structure

Each book is organized into individual markdown files by chapter, with:

- Original content preserved in structured format
- Mathematical notation maintained
- Code examples and algorithms included
- A `metadata.yaml` file (when available) containing title, author, date, keywords, and abstract

## License

See [LICENSE](LICENSE) for details.
