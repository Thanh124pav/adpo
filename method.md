# Method: Budget Allocation with Value Sharing and Pruning

## Motivation

In tree-based policy optimization, a fixed maximum depth and branch factor define a large nominal search space. Expanding this tree uniformly is wasteful: many partial reasoning prefixes are either redundant with previously explored prefixes or unlikely to improve the final answer. Our method treats inference-time tree expansion as a budget allocation problem. Instead of spending the same generation budget on every node, we use online value estimates to decide whether a newly generated segment should be expanded, shared with an existing segment, or pruned.

The two central mechanisms are:

- **Value Sharing**: if two partial trajectories induce nearly the same distribution over plausible final solutions, they can share downstream value.
- **Pruning**: if a child trajectory is significantly worse than its parent under the same answer-set probe, we stop expanding it.

Together, these mechanisms reallocate generation budget from redundant or low-value branches to branches that are more likely to contribute useful training signal.

## Setup

For each problem \(x\), the model constructs a reasoning tree. A node \(s\) corresponds to a partial trajectory:

\[
\tau(s) = x \oplus z_1 \oplus \cdots \oplus z_d,
\]

where \(z_i\) are generated reasoning segments. A vanilla SPO-style tree expands each non-terminal node according to a branch factor schedule until a maximum depth is reached. In contrast, our method decides online whether each newly generated node should continue to receive budget.

For each problem, we first construct a finite answer set:

\[
Y = \{y_1, \ldots, y_m\}.
\]

Intuitively, \(Y\) is a probe set of plausible complete solutions for the problem. The model is then asked: under the partial trajectory \(\tau(s)\), how likely are these final solutions?

For each segment/node \(s\), define:

\[
LP_i(s) = \log \pi_\theta(y_i \mid \tau(s)).
\]

The vector

\[
LP(s) = [LP_1(s), \ldots, LP_m(s)]
\]

is a compact signature of what the current partial trajectory believes about the final answer space.

Because scoring all \(m\) answers for every node is expensive, the method uses a two-stage check:

- a fast estimate using only \(K \ll m\) answers;
- a full verification using all \(m\) answers only when a trigger is likely.

We denote:

\[
\text{AvgLP}_K(s) = \frac{1}{K} \sum_{i=1}^{K} LP_i(s),
\]

and, after full scoring,

\[
\text{AvgLP}_m(s) = \frac{1}{m} \sum_{i=1}^{m} LP_i(s).
\]

## Residual Mass Outside the Answer Set

The finite answer set \(Y\) does not cover the full completion space. For a node \(s\), define the residual probability mass outside \(Y\):

\[
\delta(s) = \log \left(1 - \sum_{i=1}^{m} \exp(LP_i(s))\right).
\]

Thus:

\[
\exp(\delta(s))
\]

is the probability mass assigned to completions not represented in \(Y\). If \(\exp(\delta(s))\) is close to 1, then the answer set poorly covers the model's conditional distribution at node \(s\). In that case, value-sharing and pruning decisions should be conservative because the observed answer-set probabilities describe only a tiny part of the model's distribution.

In the implementation, the threshold computation uses the average residual mass across fully scored rows:

\[
\overline{\Delta} = \mathbb{E}_s[\exp(\delta(s))].
\]

This quantity reduces the trust placed in answer-set comparisons when \(Y\) has poor coverage.

## Thresholds

The method uses two thresholds:

\[
\eta = \frac{\epsilon}{R_{\max}} - \overline{\Delta},
\]

clamped to a small positive minimum, and

\[
\tau(K,\eta,\alpha) = \eta + \sqrt{\frac{\log(2/\alpha)}{2K}}.
\]

Here:

- \(\epsilon\) is the tolerated value error.
- \(R_{\max}\) bounds the reward scale.
- \(\overline{\Delta}\) is the average residual mass outside the answer set.
- \(K\) is the number of fast probe answers.
- \(\alpha\) controls the confidence band.

The fast threshold \(\tau\) is looser than \(\eta\) because it accounts for uncertainty from using only \(K\) samples. The full threshold \(\eta\) is used after all \(m\) answers have been scored.

## Value Sharing

### Intuition

Two partial trajectories may look different at the token level but imply nearly the same future answer distribution. Expanding both subtrees wastes budget. Value Sharing detects this redundancy and lets one node reuse the value estimate of another.

For a new segment \(s\), the method searches for a candidate target \(t\), typically the nearest previously expanded segment under \(\text{AvgLP}_K\). Other choices are possible, such as sharing with the parent or root.

The fast check is:

\[
|\text{AvgLP}_K(s) - \text{AvgLP}_K(t)| < \tau.
\]

If this passes, the method computes the full \(m\)-answer vectors and estimates an upper bound on total variation distance:

\[
TV_m(s,t)
= \frac{1}{2} \sum_{i=1}^{m}
\left| \exp(LP_i(s)) - \exp(LP_i(t)) \right|
+ \frac{1}{2}\left(\exp(\delta(s)) + \exp(\delta(t))\right).
\]

The first term compares the probability assigned to the finite answer set. The second term accounts for probability mass outside \(Y\). This makes the comparison conservative: if \(Y\) does not cover the model distribution well, the residual term becomes large and sharing is less likely.

The share rule is:

\[
TV_m(s,t) \leq \eta.
\]

If this condition holds, the method marks \(s\) as **SHARE** and stops expanding it. Its downstream value can be inherited from the target segment \(t\).

### Why This Preserves Budget

If \(TV_m(s,t)\) is small, then for any bounded reward function \(r\),

\[
|\mathbb{E}[r \mid \tau(s)] - \mathbb{E}[r \mid \tau(t)]|
\]

is small. Therefore, expanding both subtrees is unlikely to produce meaningfully different value estimates. Sharing avoids redundant expansion while keeping the node in the tree as an edge for training-time accounting.

## Pruning

### Intuition

Pruning removes branches that are significantly worse than their parent. The key comparison is between a child segment \(s\) and its parent \(pa(s)\). If conditioning on the child prefix makes the model much less likely to complete into the plausible answer set, then this branch is unlikely to lead to high-value rollouts.

The fast prune check is:

\[
\text{AvgLP}_K(pa(s)) - \text{AvgLP}_K(s) > \tau.
\]

If the child is much worse than the parent under the \(K\)-answer probe, the method scores the full answer set and checks:

\[
\text{AvgLP}_m(pa(s)) - \text{AvgLP}_m(s) > \eta.
\]

If this condition holds, the method marks \(s\) as **PRUNE** and stops expanding it.

### Interpretation

The parent represents the state before committing to the child segment. If the child sharply reduces likelihood over plausible final solutions, then the child has moved the trajectory toward a lower-value region. Continuing to generate descendants from this prefix spends budget on a branch that is already unlikely to be useful.

Pruning is therefore a budget allocation rule:

- do not spend future tokens on branches whose answer-set likelihood has dropped substantially;
- redirect the expansion budget to branches that remain competitive.

## Online Decision Rule

For each newly generated non-terminal segment \(s\):

1. Score \(K\) answer-set log probabilities \(LP_i(s)\).
2. Try **Value Sharing**:
   - find a candidate target \(t\);
   - if \(|\text{AvgLP}_K(s) - \text{AvgLP}_K(t)| < \tau\), score full \(m\);
   - if \(TV_m(s,t) \leq \eta\), mark \(s\) as SHARE.
3. If not shared, try **Pruning**:
   - compare \(s\) to its parent \(pa(s)\);
   - if \(\text{AvgLP}_K(pa(s)) - \text{AvgLP}_K(s) > \tau\), score full \(m\);
   - if \(\text{AvgLP}_m(pa(s)) - \text{AvgLP}_m(s) > \eta\), mark \(s\) as PRUNE.
4. If neither trigger fires, mark \(s\) as EXPAND and continue expanding its children.

The fast \(K\)-answer pass avoids full scoring for most nodes. Full \(m\)-answer scoring is used only when a decision is close enough to matter.

## Budget Allocation View

The method can be understood as adaptive allocation of model-evaluation budget. A uniform tree allocates budget according to the nominal search structure: every node at the same depth receives similar expansion effort. Our method allocates budget according to online evidence:

- redundant nodes share value instead of expanding;
- low-value nodes are pruned;
- remaining budget flows to branches whose answer-set signature remains promising and distinct.

This creates an effective tree that can be much smaller than the nominal tree while preserving or improving downstream performance.

Two natural budget metrics follow from this view:

- **Generated-token budget**: the number of tokens spent expanding tree nodes.
- **Model-evaluation budget**: generated tokens plus answer-set generation and log-probability scoring tokens.

The first metric measures search budget. The second measures total model compute more fairly, including the overhead introduced by Value Sharing and Pruning.

## Important Failure Mode: Poor Answer-Set Coverage

If \(\exp(LP_i(s))\) is extremely small for all answers \(y_i\), then:

\[
\sum_i \exp(LP_i(s)) \approx 0,
\]

so:

\[
\exp(\delta(s)) \approx 1.
\]

This means the answer set \(Y\) covers almost none of the model's conditional probability mass. In that regime, \(\eta\) can collapse to a very small value, making pruning overly aggressive. Practically, this may appear as prune rate near 1.0 at shallow depths.

Possible mitigations include:

- setting a positive \(\eta\) override;
- using shorter or better-aligned answer-set completions;
- reducing answer-set temperature;
- normalizing log probabilities by answer length;
- shortening early segments so depth-1 prefixes do not overcommit too strongly.

## Summary

Value Sharing and Pruning both use answer-set log probabilities as a low-dimensional probe of a partial trajectory's future value. Value Sharing removes redundant computation by merging nodes with similar answer distributions. Pruning removes branches whose answer likelihood degrades sharply relative to the parent. Together, they turn fixed tree search into adaptive budget allocation: the model spends fewer tokens on redundant or low-value branches and reserves more budget for promising, diverse reasoning paths.
