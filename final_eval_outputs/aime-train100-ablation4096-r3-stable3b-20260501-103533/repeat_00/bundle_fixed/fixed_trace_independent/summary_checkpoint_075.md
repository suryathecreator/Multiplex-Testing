# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 75
- baseline prompts with full k: 75
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 0.5333 | 0.6133 | 0.6667 | 0.7467 | 0.7867 | 0.7867 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline Independent | 6333.3 | 12782.7 | 25604.9 | 51197.3 | 103554.3 | 207184.2 |

## Detailed Matched-Per-k Summary

- Baseline Independent: pass@1 prompts=75, pass@2 prompts=75, pass@4 prompts=75, pass@8 prompts=75, pass@16 prompts=75, pass@32 prompts=75

## Retry Statistics

- Baseline Independent: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:40:51 (20450.5 seconds)
- started_at: 2026-05-02T04:51:02.787718+00:00
- finished_at: 2026-05-02T10:31:53.295001+00:00

## Matched Denominators

- pass@1: 75 matched prompts
- pass@2: 75 matched prompts
- pass@4: 75 matched prompts
- pass@8: 75 matched prompts
- pass@16: 75 matched prompts
- pass@32: 75 matched prompts
