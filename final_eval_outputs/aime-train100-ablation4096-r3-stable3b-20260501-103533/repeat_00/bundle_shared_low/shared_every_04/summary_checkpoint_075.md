# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 75
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.5067 | 0.5467 | 0.6133 | 0.6933 | 0.7600 | 0.7867 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 6499.7 | 9006.0 | 13601.4 | 26218.4 | 53097.6 | 106641.5 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=75, pass@2 prompts=75, pass@4 prompts=75, pass@8 prompts=75, pass@16 prompts=75, pass@32 prompts=75

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 5:09:24 (18563.5 seconds)
- started_at: 2026-05-04T03:33:16.560027+00:00
- finished_at: 2026-05-04T08:42:40.083942+00:00

## Matched Denominators

- pass@1: 75 matched prompts
- pass@2: 75 matched prompts
- pass@4: 75 matched prompts
- pass@8: 75 matched prompts
- pass@16: 75 matched prompts
- pass@32: 75 matched prompts
