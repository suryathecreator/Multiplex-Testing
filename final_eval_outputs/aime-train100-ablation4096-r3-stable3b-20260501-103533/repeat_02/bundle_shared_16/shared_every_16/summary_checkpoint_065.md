# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 65
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.5077 | 0.5846 | 0.6154 | 0.6615 | 0.6769 | 0.7077 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6447.4 | 8741.4 | 13361.9 | 22807.4 | 41358.6 | 81496.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=65, pass@2 prompts=65, pass@4 prompts=65, pass@8 prompts=65, pass@16 prompts=65, pass@32 prompts=65

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 2:23:08 (8588.4 seconds)
- started_at: 2026-05-02T08:25:36.107096+00:00
- finished_at: 2026-05-02T10:48:44.517281+00:00

## Matched Denominators

- pass@1: 65 matched prompts
- pass@2: 65 matched prompts
- pass@4: 65 matched prompts
- pass@8: 65 matched prompts
- pass@16: 65 matched prompts
- pass@32: 65 matched prompts
