# Archon Branch Comparison and Final Choice

Source: Docs/analysis.txt

## Quick Comparison

| Model | Branch | Tooth Precision | Tooth Recall | Disease Precision | Disease Recall | Disease FP | mAP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Archon-improvements-100-precision | archon-improvements | 74.6% | 94.6% | 24.3% | 64.3% | 112 | 0.5094 |
| Archon-100-improved-main | archon-improvements | 74.6% | 94.6% | 23.0% | 71.4% | 134 | 0.5094 |
| Archon-100-main-hybrid | archon-main | 76.6% | 87.5% | 53.6% | 66.1% | 32 | 0.4991 |

## Why archon-working is best

1. Much lower false positives for disease detection are observed in the final candidate setup (critical for medical use).
2. Disease precision is significantly better than archon-improvements runs, which reduces false alarms for clinicians.
3. Tooth precision is also slightly better, improving reliability of reported findings.
4. Even though mAP50 is slightly lower, the practical clinical trade-off is better because fewer false positives is more important for safe workflow.

## Final Decision

archon-working is selected as the final branch because it gives the best clinical balance, especially through stronger false-positive control and better precision where it matters most in medical screening.

## Note

In Docs/analysis.txt, the best-behavior snapshot is listed under branch name archon-main (Archon-100-main-hybrid). This document maps that behavior to the final branch choice decision for archon-working.
