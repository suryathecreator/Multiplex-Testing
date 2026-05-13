# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 85
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.4588 | 0.5294 | 0.5647 | 0.6118 | 0.6235 | 0.6824 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6502.8 | 8812.3 | 13494.3 | 22927.1 | 41642.7 | 82317.7 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=85, pass@2 prompts=85, pass@4 prompts=85, pass@8 prompts=85, pass@16 prompts=85, pass@32 prompts=85

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:23:08 (8588.4 seconds)
- started_at: 2026-05-02T08:25:36.107096+00:00
- finished_at: 2026-05-02T10:48:44.517281+00:00

## Matched Denominators

- pass@1: 85 matched prompts
- pass@2: 85 matched prompts
- pass@4: 85 matched prompts
- pass@8: 85 matched prompts
- pass@16: 85 matched prompts
- pass@32: 85 matched prompts
