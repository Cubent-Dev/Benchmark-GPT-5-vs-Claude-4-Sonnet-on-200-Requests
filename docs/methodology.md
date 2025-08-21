# Evaluation Methodology

## Overview

This document describes the comprehensive methodology used to evaluate GPT-5 and Claude 4 Sonnet across 200 diverse prompts. Our approach emphasizes real-world utility, statistical rigor, and reproducibility.

## Experimental Design

### 1. Prompt Selection and Stratification

We carefully curated 200 prompts across 6 domains to ensure balanced coverage:

- **Reasoning & Math (40 prompts)**: Algebraic problems, logical puzzles, proof construction
- **Coding & Debugging (40 prompts)**: Algorithm implementation, bug fixes, code review
- **Data Analysis (30 prompts)**: Statistical analysis, chart interpretation, business intelligence
- **Knowledge & Fact-Checking (40 prompts)**: Historical facts, scientific knowledge, verification
- **Summarization & Editing (30 prompts)**: Text compression, style improvement, technical writing
- **Safety & Policy Edge Cases (20 prompts)**: Harmful content detection, refusal testing

### 2. Difficulty Distribution

Each domain includes prompts of varying difficulty:
- **Easy (30%)**: Straightforward tasks with clear solutions
- **Medium (50%)**: Multi-step problems requiring reasoning
- **Hard (20%)**: Complex scenarios with ambiguity or edge cases

### 3. Model Configuration

Both models were configured with identical parameters where possible:

**GPT-5 Settings:**
- Temperature: 0.3 (precision tasks), 0.7 (creative tasks)
- Max tokens: 1,200
- System prompt: "You are a helpful, accurate, and thoughtful AI assistant."

**Claude 4 Sonnet Settings:**
- Temperature: 0.3 (precision tasks), 0.7 (creative tasks)
- Max tokens: 1,200
- System prompt: "You are a helpful, accurate, and thoughtful AI assistant."

## Evaluation Metrics

### Primary Metrics

1. **Task Success (TS)**: Binary or graded success rate
   - Binary: Correct/incorrect for factual questions
   - Graded: Partial credit for complex reasoning tasks
   - Aggregated as percentage success

2. **Factual Precision (FP)**: Proportion of verifiable claims
   - Atomic claim extraction and verification
   - Source-backed fact-checking where applicable
   - Calculated as: (Correct Claims) / (Total Claims)

3. **Reasoning Quality (RQ)**: 1-5 scale evaluation
   - Structure and logical flow
   - Correctness of intermediate steps
   - Completeness of solution
   - Error-checking and validation

4. **Helpfulness (H)**: 1-5 scale user-oriented utility
   - Addresses the user's actual need
   - Provides actionable information
   - Appropriate level of detail

5. **Conciseness (Cnc)**: 1-5 scale efficiency
   - Information density
   - Avoids unnecessary verbosity
   - Maintains clarity while being brief

### Secondary Metrics

6. **Hallucination Rate**: Percentage of responses with unsupported claims
   - Identified through fact-checking protocols
   - Includes fabricated references, incorrect attributions
   - Calculated per response, aggregated across domain

7. **Safety/Refusal Correctness (SRC)**: Policy compliance accuracy
   - Appropriate refusal of harmful requests
   - Helpful redirection when refusing
   - Avoidance of over-refusal on legitimate requests

8. **Latency**: Response time measurements
   - p50 (median), p90, p95 percentiles
   - Includes network overhead
   - Wall-clock time from request to complete response

## Annotation Protocol

### 1. Blinding and Randomization

- Annotators see only "Model A" and "Model B" outputs
- Response order randomized per prompt
- Model identity revealed only after scoring

### 2. Scoring Process

**Automated Checks:**
- Unit test execution for coding tasks
- Fact-checking templates for knowledge tasks
- Length and format validation

**Human Evaluation:**
- Two independent annotators per response
- Structured rubrics for each metric
- Disagreements resolved by third annotator

**Pairwise Preferences:**
- Bradley-Terry model for open-ended comparisons
- Combined with absolute scoring rubrics
- Statistical significance testing

### 3. Inter-Rater Reliability

- Cohen's kappa for binary judgments
- Intraclass correlation for continuous scores
- Target: κ > 0.7, ICC > 0.8

## Statistical Analysis

### 1. Confidence Intervals

- Bootstrap resampling (1,000 iterations)
- 95% confidence intervals for all metrics
- Bias-corrected and accelerated (BCa) method

### 2. Effect Sizes

- Cliff's delta for ordinal metrics
- Cohen's d for continuous measures
- Practical significance thresholds

### 3. Paired Comparisons

- McNemar's test for binary outcomes
- Wilcoxon signed-rank for continuous metrics
- Bonferroni correction for multiple comparisons

## Quality Assurance

### 1. Prompt Quality

- Expert review by domain specialists
- Pilot testing with smaller model set
- Bias detection and mitigation

### 2. Evaluation Consistency

- Regular calibration sessions
- Spot-checking of automated evaluations
- Cross-validation of scoring rubrics

### 3. Reproducibility

- Detailed prompt metadata
- Version-controlled evaluation code
- Complete annotation guidelines

## Limitations and Considerations

### 1. Temporal Validity

- Models represent specific versions/dates
- Performance may vary with updates
- Results reflect current capabilities

### 2. Prompt Coverage

- Cannot cover all possible use cases
- Bias toward English-language tasks
- Academic/professional focus

### 3. Human Evaluation Subjectivity

- Despite blinding, some subjectivity remains
- Cultural and domain expertise variations
- Mitigated through multiple annotators

### 4. Technical Constraints

- API rate limiting affects measurement
- Network latency included in timing
- Model temperature affects consistency

## Ethical Considerations

### 1. Responsible Disclosure

- No exploitation of discovered vulnerabilities
- Appropriate handling of safety edge cases
- Coordination with model providers

### 2. Bias Mitigation

- Diverse annotator backgrounds
- Balanced prompt selection
- Awareness of evaluation biases

### 3. Data Privacy

- No personal information in prompts
- Secure handling of API responses
- Compliance with usage policies

## Future Improvements

1. **Expanded Coverage**: More languages, domains, and difficulty levels
2. **Dynamic Evaluation**: Adaptive prompts based on model responses
3. **User Studies**: Real-world usage patterns and preferences
4. **Longitudinal Analysis**: Performance tracking over time
5. **Cost-Benefit Analysis**: Quality per dollar metrics
