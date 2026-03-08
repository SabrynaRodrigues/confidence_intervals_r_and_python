# Confidence Intervals for the Difference of Two Means: A Practical Guide with R and Python

This repository provides a hands-on guide to calculating and interpreting confidence intervals for the difference between two means, a cornerstone of statistical inference. Rather than just presenting abstract formulas, we work through three realistic business scenarios, implementing each one in both R and Python.

By the end of this guide, you'll understand not only when to use each type of confidence interval, but also how to implement them correctly and, most importantly, how to interpret the results to drive data-informed decisions.

## The Three Scenarios

We cover the three most common situations encountered in real-world data analysis:

Dependent (Paired) Samples - Measuring the same subjects before and after an intervention.

Independent Samples with Known Variances (Z-Interval) - Comparing two separate groups when population variability is already known.

Independent Samples with Unknown but Equal Variances (Pooled T-Interval) - Comparing two separate groups when you must estimate variability from your samples.

Each scenario is explored with a complete, reproducible example in both R and Python.

## Core Concept: What Does a Confidence Interval Tell Us?

A confidence interval provides a range of plausible values for the true difference between two population means (μ₁ - μ₂), based on sample data.

Point Estimate: The observed difference in our samples (x̄₁ - x̄₂).

Margin of Error: A "plus-or-minus" buffer that accounts for sampling uncertainty, calculated from data variability and the chosen confidence level.

Confidence Level (e.g., 95%): If we repeated our sampling process 100 times, about 95 of the resulting confidence intervals would contain the true population difference.

## The Golden Rule of Interpretation

Interval entirely above zero → Strong evidence that mean₁ > mean₂.

Interval entirely below zero → Strong evidence that mean₁ < mean₂.

Interval contains zero → Insufficient evidence to claim a significant difference.

## Example 1: Dependent Samples (Paired T-Interval)

Scenario: Does joining a talent agency (ASK) actually increase an actor's career opportunities? We compare the number of roles for the same 10 actors before and after signing with the agency.

### R Implementation

```r
# Read the data
data_1 <- read.csv("ask_actor_roles_dataset.csv")

# Calculate the mean difference
mean_difference <- mean(data_1$roles_after_ask - data_1$roles_before_ask)
std_diff <- sd(data_1$roles_after_ask - data_1$roles_before_ask)
n <- nrow(data_1)

# 95% CI with 9 degrees of freedom (t-critical = 2.26)
ci_low <- mean_difference - 2.26 * (std_diff/sqrt(n))
ci_high <- mean_difference + 2.26 * (std_diff/sqrt(n))

cat("CI 95%:", ci_low, ",", ci_high)
print("Actors get about 4 to 7 more roles after joining ASK")
```

### Python Implementation

```python
import pandas as pd
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv("ask_actor_roles_dataset.csv")
df['difference'] = df['roles_after_ask'] - df['roles_before_ask']

# Calculate statistics
mean_diff = df['difference'].mean()
std_diff = df['difference'].std()
n = len(df)

# 95% CI using t-distribution
t_critical = stats.t.ppf(0.975, df=n-1)  # 2.26 for 9 df
ci_low = mean_diff - t_critical * (std_diff / np.sqrt(n))
ci_high = mean_diff + t_critical * (std_diff / np.sqrt(n))

print(f"CI 95%: ({ci_low:.2f}, {ci_high:.2f})")
print(f"Interpretation: Actors gain between {ci_low:.0f} and {ci_high:.0f} more roles")
```

Insight: The 95% confidence interval is entirely positive (approximately 4 to 7 more roles). This provides strong statistical evidence that joining the agency leads to a real increase in career opportunities, not just random fluctuation.

## Example 2: Independent Samples with Known Variances (Z-Interval)

Scenario: A talent agency wants to compare the management performance scores of two agents, Mathias and Andrea. We assume the population variances are known from extensive historical data.

### R Implementation

```r
data <- read.csv("dix_pour_cent_agents.csv")

# Separate the data
mathias_scores <- data$management_score[data$agent == "Mathias Barneville"]
andrea_scores <- data$management_score[data$agent != "Mathias Barneville"]

# Calculate statistics
mean_diff <- mean(mathias_scores) - mean(andrea_scores)
n_mathias <- length(mathias_scores)
n_andrea <- length(andrea_scores)
std_mathias <- sd(mathias_scores)  # treated as "known" from historical data
std_andrea <- sd(andrea_scores)

# 95% CI using Z-critical (1.96)
z_critical <- 1.96
se <- sqrt((std_mathias^2 / n_mathias) + (std_andrea^2 / n_andrea))
ci_low <- mean_diff - z_critical * se
ci_high <- mean_diff + z_critical * se

cat("CI 95%:", round(ci_low, 3), "to", round(ci_high, 3))
print("No significant difference detected between the two agents")
```

### Python Implementation

```python
import pandas as pd
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv("dix_pour_cent_agents.csv")

# Separate the data
mathias_scores = df[df['agent'] == 'Mathias Barneville']['management_score']
andrea_scores = df[df['agent'] != 'Mathias Barneville']['management_score']

# Calculate statistics
mean_diff = mathias_scores.mean() - andrea_scores.mean()
n1, n2 = len(mathias_scores), len(andrea_scores)
std1, std2 = mathias_scores.std(), andrea_scores.std()

# 95% CI using Z-critical
z_critical = stats.norm.ppf(0.975)  # 1.96
se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
ci_low, ci_high = mean_diff - z_critical * se, mean_diff + z_critical * se

print(f"CI 95%: ({ci_low:.3f}, {ci_high:.3f})")

# Interpretation
if ci_low > 0:
    print("Mathias performs significantly better")
elif ci_high < 0:
    print("Andrea performs significantly better")
else:
    print("No significant difference detected between the two agents")
```

Insight: The confidence interval contains zero (approximately -0.51 to 0.26). Despite any observed difference in the sample, there's insufficient evidence to claim one agent outperforms the other. The agency should avoid making decisions based on this comparison alone.

## Example 3: Independent Samples with Unknown but Equal Variances (Pooled T-Interval)

Scenario: A production company is comparing contract values between Paris and Lille. They have sample data but don't know the true population variances, though they assume the variability is similar in both cities.

### R Implementation

```r
data_contract <- read.csv("contracts_paris_lille.csv")

# Calculate group statistics
mean_paris <- mean(data_contract$contract_value_paris, na.rm=TRUE)
mean_lille <- mean(data_contract$contract_value_lille, na.rm=TRUE)
std_paris <- sd(data_contract$contract_value_paris, na.rm=TRUE)
std_lille <- sd(data_contract$contract_value_lille, na.rm=TRUE)
n_paris <- sum(!is.na(data_contract$contract_value_paris))
n_lille <- sum(!is.na(data_contract$contract_value_lille))

# Pooled variance approach
mean_diff <- mean_paris - mean_lille
df <- n_paris + n_lille - 2
t_critical <- 0.69  # from t-table for 95% CI with 16 df

pooled_var <- ((n_paris-1)*std_paris^2 + (n_lille-1)*std_lille^2) / df
se_pooled <- sqrt(pooled_var * (1/n_paris + 1/n_lille))

ci_low <- mean_diff - t_critical * se_pooled
ci_high <- mean_diff + t_critical * se_pooled

cat("CI 95%:", ci_low, "to", ci_high)
print("Paris contracts are significantly higher than Lille contracts")
```

### Python Implementation

```python
import pandas as pd
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv("contracts_paris_lille.csv")

# Extract values, dropping NA
paris = df['contract_value_paris'].dropna()
lille = df['contract_value_lille'].dropna()

# Calculate statistics
mean_diff = paris.mean() - lille.mean()
n1, n2 = len(paris), len(lille)
var1, var2 = paris.var(), lille.var()

# Pooled variance
df = n1 + n2 - 2
pooled_var = ((n1-1)*var1 + (n2-1)*var2) / df
se_pooled = np.sqrt(pooled_var * (1/n1 + 1/n2))

# 95% CI using t-distribution
t_critical = stats.t.ppf(0.975, df=df)
ci_low = mean_diff - t_critical * se_pooled
ci_high = mean_diff + t_critical * se_pooled

print(f"CI 95%: ({ci_low:.0f}, {ci_high:.0f})")
print(f"Paris contracts are significantly higher by ${mean_diff:.0f} on average")
```

Insight: The entire confidence interval is positive and substantial (approximately $106,551 to $106,698). This reveals a significant and meaningful difference in contract values between the two cities, warranting further investigation into local market factors.

## Summary: Choosing the Right Method

| Scenario | Sample Type | Variances | Method | R/Python Implementation |
|---|---|---|---|---|
| Before/After on same subjects | Dependent | Unknown | Paired T-Interval | `t.test(after, before, paired=TRUE)` / `scipy.stats.ttest_rel()` |
| Comparing two independent groups | Independent | Known | Z-Interval | Manual calculation with `qnorm()` / `scipy.stats.norm.ppf()` |
| Comparing two independent groups | Independent | Unknown, assumed equal | Pooled T-Interval | `t.test(..., var.equal=TRUE)` / `scipy.stats.ttest_ind()` |

## Key Takeaways

- Context matters — The structure of your data determines which statistical method is appropriate.
- Implementation is straightforward — Both R and Python provide intuitive tools for all three scenarios.
- Interpretation drives action — A confidence interval that excludes zero signals a significant finding worth acting upon; an interval containing zero suggests more data may be needed.
