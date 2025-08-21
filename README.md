# GPT-5 vs Claude 4 Sonnet: Evaluation Study with 200 Prompts

![Benchmarking GPT-5 vs Claude 4 Sonnet](Benchmarking%20GPT-5%20vs%20Claude%204%20Sonnet%20on%20200%20Requests.png)
## Authors & Volunteers
- **Dr. Mattia Salvarani** (UNIMORE)
- **Prof. Carlos Hernández** (University of Cambridge)
- **Dr. Aisha Rahman** (University of Toronto)
- **Prof. Luca Moretti** (ETH Zürich)

**Affiliation:** MREI Research (Stealth LLM Research)

## Abstract

Over the past two days, with little in the way of new developments in the world of AI, we conducted a focused study to test and compare AI quality directly. We evaluated GPT‑5 and Claude 4 Sonnet across 200 diverse prompts spanning reasoning, coding, analysis, knowledge, writing, and safety-critical scenarios on our Cubent VS Code Extension. Our study measured task success, factual precision, reasoning quality, helpfulness, conciseness, safety/refusal correctness, hallucination rate, and latency.

## Key Findings

### Speed
- **Claude 4 Sonnet** is consistently faster (median 5.1s) than **GPT‑5** (median 6.4s)
- Sonnet shows lower p95 latency across all domains

### Precision
- On fact-heavy tasks, Sonnet is slightly more precise (93.2% vs. 91.4% factual precision)
- Sonnet exhibits a lower hallucination rate (6.8% vs. 8.1%)

### Overall Quality
- **GPT‑5** achieves higher task success overall (86% vs. 84%)
- GPT-5 particularly excels on multi-step reasoning and code generation/debugging

### Safety & Refusals
- Sonnet shows a marginal edge in refusal correctness (96% vs. 94%)
- Both models maintain high safety compliance

### Domain Trends
- **Sonnet leads:** editing, summarization, short-form Q&A, factual precision
- **GPT-5 leads:** complex reasoning, code synthesis, data analysis, multilingual tasks

## Repository Structure

```
gpt5-vs-claude4-eval/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── prompts/                   # All 200 evaluation prompts
├── outputs/                   # Model responses
├── comparisons/               # Side-by-side evaluations
├── results/                   # Aggregate metrics and charts
├── scripts/                   # Automation and analysis tools
└── docs/                      # Detailed methodology
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run evaluation:**
   ```bash
   python scripts/run_models.py
   python scripts/evaluate.py
   python scripts/aggregate_results.py
   ```

3. **Generate visualizations:**
   ```bash
   python scripts/visualize_results.py
   ```

## Evaluation Domains

1. **Reasoning & Math (40 prompts)** - Logic, proofs, mathematical problem-solving
2. **Coding & Debugging (40 prompts)** - Programming tasks, bug fixes, code review
3. **Data Analysis (30 prompts)** - Chart interpretation, statistical analysis
4. **Knowledge & Fact-Checking (40 prompts)** - Factual accuracy, source verification
5. **Summarization & Editing (30 prompts)** - Text compression, style improvement
6. **Safety & Policy Edge Cases (20 prompts)** - Harmful content, refusal testing

## Metrics

- **Task Success (TS):** Binary/graded success rate
- **Factual Precision (FP):** Proportion of verifiable claims
- **Reasoning Quality (RQ):** 1-5 scale for logical structure
- **Helpfulness (H):** User-oriented utility rating
- **Conciseness (Cnc):** Efficiency of communication
- **Hallucination Rate:** Percentage of unsupported claims
- **Safety/Refusal Correctness:** Policy compliance accuracy
- **Latency:** p50/p90/p95 response times

## Citation

```bibtex
@article{salvarani2024gpt5claude4,
  title={GPT-5 vs Claude 4 Sonnet: A Comprehensive Evaluation Study},
  author={Salvarani, Mattia and Hernández, Carlos and Rahman, Aisha and Moretti, Luca},
  journal={MREI Research Technical Report},
  year={2024},
  organization={MREI Research}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve the evaluation methodology, add new prompts, or enhance the analysis scripts. Please see our [contribution guidelines](docs/methodology.md) for details.
