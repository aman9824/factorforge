---
id: factor-timing-overfitting
title: Factor Behavior Across Regimes and the Overfitting Problem
source: factorforge:research-notes
authors: Harvey; Liu; Zhu; López de Prado
published: 2026-01-20
---

# Factor Behavior Across Regimes and the Overfitting Problem

## Summary
Factor premia vary across macroeconomic regimes, which tempts researchers to time factors. But the
same flexibility that enables factor timing also enables overfitting, and most published factors do
not survive honest multiple-testing corrections.

## Regime dependence

### Inflation
The value factor tends to outperform in high-inflation regimes, while long-duration growth stocks
suffer. Inflation is one of the most important regimes for factor behavior.

### Recession
Defensive factors such as quality and low-volatility outperform in a recession, while the size
factor and the value factor often underperform as fragile and distressed firms sell off.

### Market recovery
The momentum factor is vulnerable in a market recovery because prior losers rebound, and the size
factor often leads as small-cap stocks recover first.

## The overfitting problem

### Multiple testing
Harvey, Liu, and Zhu (2016) argued that most claimed factors are false discoveries because
researchers test hundreds of candidates and report only the winners. A t-statistic above 2.0 is no
longer sufficient evidence once multiple testing is accounted for.

### Backtest overfitting
López de Prado warned that backtests are routinely overfit through repeated tuning on the same data.
The deflated Sharpe ratio and the probability of backtest overfitting are designed to discount a
result by the number of configurations that were tried.

## Caveats
A factor that looks strong in a single in-sample backtest should be distrusted until it survives an
out-of-sample test, a deflated Sharpe ratio that accounts for the number of trials, and a check of
its sensitivity to parameter choices. Factor timing is especially prone to data mining.

## References
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. Review of Financial Studies.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. Journal of Computational Finance.
