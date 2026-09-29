# Trading AI rules

- Signal at close t.
- Execution at open t+1.
- No overnight positions.
- Stop has priority if stop and target are hit on the same candle.
- No synthetic OHLCV data.
- Do not call a model score a probability unless it is calibrated.
- Keep walk-forward research separate from locked final evaluation.
- Prefer reproducible negative results over optimistic claims.
- Do not modify data/ or runs/.
- Run tests before committing changes.
