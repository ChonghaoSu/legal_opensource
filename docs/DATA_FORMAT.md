# Data Format Guide

## Fine-tuning Data Format

Fine-tuning data should be in JSONL format (one JSON object per line) with the following structure:

```json
{
  "prompt": "Your prompt text here...",
  "completion": "Expected completion/summary with citations..."
}
```

### Example

```json
{
  "prompt": "Analyze the following legal document excerpt and create a summary with accurate citations.\n\nDocument Excerpt:\nThe court held that the defendant's motion to dismiss should be granted. See Smith v. Jones, 123 F.3d 456 (2020).\n\nCitations found: Smith v. Jones, 123 F.3d 456\n\nSummary:",
  "completion": "The court granted the defendant's motion to dismiss (Smith v. Jones, 123 F.3d 456 (2020)) because the plaintiff failed to state a claim."
}
```

## DPO Data Format

DPO (Direct Preference Optimization) data requires pairs of chosen (preferred) and rejected (less preferred) responses:

```json
{
  "prompt": "Your prompt text here...",
  "chosen": "Good response with accurate citations...",
  "rejected": "Bad response without citations or with hallucinations..."
}
```

### Example

```json
{
  "prompt": "Analyze the following legal document excerpt...\n\nSummary:",
  "chosen": "The court granted the motion to dismiss (Smith v. Jones, 123 F.3d 456 (2020)) because the plaintiff failed to state a claim under Rule 12(b)(6).",
  "rejected": "The court granted the motion to dismiss because the plaintiff's claim was insufficient. The court cited several cases including Brown v. White, 789 F.2d 123 (2019) in support."
}
```

**Key differences:**
- `chosen`: Response with accurate citations from the source document
- `rejected`: Response with missing citations, hallucinated citations, or incorrect information

## Creating DPO Pairs

To create DPO pairs from your fine-tuning data:

1. Generate multiple responses from your fine-tuned model
2. Evaluate each response for citation accuracy
3. Label responses as "chosen" (accurate) or "rejected" (inaccurate/hallucinated)
4. Pair them with the same prompt

You can use the `scripts/create_dpo_pairs.py` script to help automate this process.

## Citation Patterns

The system recognizes common legal citation formats:

- Case citations: `123 F.3d 456`, `Smith v. Jones`
- Federal Reporter: `123 F.2d 456`
- Federal Supplement: `123 F. Supp. 456`
- Rules: `Fed. R. Evid. 802`, `Fed. R. Civ. P. 12(b)(6)`

Make sure your training data includes proper citations in these formats.


