# AIME 2024 Pass@k Comparison

## Clean Summary

- matched prompts with full k=32: 45
- baseline prompts with full k: 0
- shared-trace prompts with full k: 0
- standard-generation prompts with full k: 0

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 0.4889 | 0.6000 | 0.6222 | 0.6222 | 0.6667 | 0.8000 |

| Method | cost@1 | cost@2 | cost@4 | cost@8 | cost@16 | cost@32 |
| --- | --- | --- | --- | --- | --- | --- |
| shared_trace_group_16 | 6516.7 | 8892.8 | 13567.1 | 23187.1 | 42462.6 | 82095.2 |

## Detailed Matched-Per-k Summary

- shared_trace_group_16: pass@1 prompts=45, pass@2 prompts=45, pass@4 prompts=45, pass@8 prompts=45, pass@16 prompts=45, pass@32 prompts=45

## Retry Statistics

- shared_trace_group_16: attempts=0, accepted=0, rejected=0, tokens=0, rejected_tokens=0

## Wall Clock

- duration: 0:25:37 (1536.7 seconds)
- started_at: 2026-05-02T07:36:02.320454+00:00
- finished_at: 2026-05-02T08:01:39.046168+00:00

## Matched Denominators

- pass@1: 45 matched prompts
- pass@2: 45 matched prompts
- pass@4: 45 matched prompts
- pass@8: 45 matched prompts
- pass@16: 45 matched prompts
- pass@32: 45 matched prompts
