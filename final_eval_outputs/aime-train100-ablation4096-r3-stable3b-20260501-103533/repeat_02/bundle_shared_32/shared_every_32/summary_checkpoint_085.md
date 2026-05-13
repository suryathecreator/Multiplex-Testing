# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 85
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.5059 | 0.5647 | 0.5765 | 0.6235 | 0.6235 | 0.6353 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6681.4 | 9243.2 | 14204.6 | 24255.7 | 44270.6 | 84177.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=85, pass@2 prompts=85, pass@4 prompts=85, pass@8 prompts=85, pass@16 prompts=85, pass@32 prompts=85

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:48:10 (2890.0 seconds)
- started_at: 2026-05-02T07:02:14.976121+00:00
- finished_at: 2026-05-02T07:50:25.024598+00:00

## Matched Denominators

- pass@1: 85 matched prompts
- pass@2: 85 matched prompts
- pass@4: 85 matched prompts
- pass@8: 85 matched prompts
- pass@16: 85 matched prompts
- pass@32: 85 matched prompts
