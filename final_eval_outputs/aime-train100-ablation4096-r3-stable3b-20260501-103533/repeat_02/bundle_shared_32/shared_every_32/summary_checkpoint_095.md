# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 95
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.5158 | 0.5684 | 0.5789 | 0.6211 | 0.6211 | 0.6421 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6659.5 | 9178.7 | 14128.3 | 24115.5 | 44042.3 | 83779.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=95, pass@2 prompts=95, pass@4 prompts=95, pass@8 prompts=95, pass@16 prompts=95, pass@32 prompts=95

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:48:10 (2890.0 seconds)
- started_at: 2026-05-02T07:02:14.976121+00:00
- finished_at: 2026-05-02T07:50:25.024598+00:00

## Matched Denominators

- pass@1: 95 matched prompts
- pass@2: 95 matched prompts
- pass@4: 95 matched prompts
- pass@8: 95 matched prompts
- pass@16: 95 matched prompts
- pass@32: 95 matched prompts
