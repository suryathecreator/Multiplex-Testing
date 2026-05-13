# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 55
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 0.6182 | 0.6545 | 0.6909 | 0.7091 | 0.7091 | 0.7455 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_32 | 6045.3 | 8040.1 | 12189.6 | 20178.5 | 36057.3 | 67920.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_32: pass@1 prompts=55, pass@2 prompts=55, pass@4 prompts=55, pass@8 prompts=55, pass@16 prompts=55, pass@32 prompts=55

## Retry Statistics

- shared_trace_group_32: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 1:12:09 (4329.2 seconds)
- started_at: 2026-05-02T01:24:15.235665+00:00
- finished_at: 2026-05-02T02:36:24.449743+00:00

## Matched Denominators

- pass@1: 55 matched prompts
- pass@2: 55 matched prompts
- pass@4: 55 matched prompts
- pass@8: 55 matched prompts
- pass@16: 55 matched prompts
- pass@32: 55 matched prompts
