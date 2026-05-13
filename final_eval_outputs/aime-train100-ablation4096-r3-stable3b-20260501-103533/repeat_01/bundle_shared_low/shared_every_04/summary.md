# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 100
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 0.4700 | 0.4800 | 0.4900 | 0.5900 | 0.7000 | 0.7400 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_4 | 6598.0 | 9159.8 | 14261.3 | 28274.2 | 56053.2 | 111761.4 |

## Detailed Matched-Per-k Summary

- shared_trace_group_4: pass@1 prompts=100, pass@2 prompts=100, pass@4 prompts=100, pass@8 prompts=100, pass@16 prompts=100, pass@32 prompts=100

## Retry Statistics

- shared_trace_group_4: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:00:00 (0.2 seconds)
- started_at: 2026-05-04T17:49:55.193870+00:00
- finished_at: 2026-05-04T17:49:55.416761+00:00

## Matched Denominators

- pass@1: 100 matched prompts
- pass@2: 100 matched prompts
- pass@4: 100 matched prompts
- pass@8: 100 matched prompts
- pass@16: 100 matched prompts
- pass@32: 100 matched prompts
