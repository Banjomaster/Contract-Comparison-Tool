# Contract Comparison Tool

A powerful tool for comparing PDF contracts and identifying meaningful differences. The tool uses advanced natural language processing to focus on substantive changes that matter, like financial terms, deadlines, and significant legal provisions.

## Features

- Extracts text from PDF contracts using multiple methods for better results
- Segments contracts into logical sections for more accurate comparison
- Identifies meaningful differences between contract versions
- Analyzes changes using LLM (large language model) to determine:
  - Exact text changes
  - Summary of changes in plain English
  - Practical impact of changes
  - Legal implications
  - Significance rating (1-5 scale)
- Generates comprehensive reports with executive summaries
- Compares one original contract against multiple revised contracts
- Creates separate output files for each comparison

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd Compare_Contracts
```

2. Install the required dependencies:
```
pip install PyPDF2 difflib pdfplumber openai python-dotenv
```

3. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```
python main.py path/to/original/contract.pdf path/to/revised/contract.pdf
```

### Compare with Multiple Revised Contracts

```
python main.py path/to/original/contract.pdf directory-with-revised-contracts/
```

### Using LLM Analysis

To use the enhanced LLM analysis (requires OpenAI API key):

```
python main.py path/to/original/contract.pdf path/to/revised/contract.pdf --llm
```

### Additional Options

- Use `-v` or `--verbose` for more detailed output
- If no comparison target is specified, the tool will look for revised contracts in a "new_contracts" directory

## Output

The tool generates separate output files for each comparison with detailed analysis:

- `contract_comparison_Lot_A_vs_Lot_B.txt` - Comparison between contracts for Lot A and Lot B

Each output file includes:
- Detailed section-by-section comparison
- Analysis of additions, removals, and changes
- Significance ratings for each change
- Executive summary highlighting the most important changes

## Requirements

- Python 3.6+
- PyPDF2
- difflib
- pdfplumber (optional, for better text extraction)
- openai (optional, for enhanced semantic analysis)
- python-dotenv (optional, for better API key management) 