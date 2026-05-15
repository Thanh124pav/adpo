# Method: Sibling-Local Value Sharing and Pruning

## Motivation

Tree-based policy optimization spends a large generation budget on subtrees
that often become redundant. ValueShare asks whether two partial reasoning
states induce similar future model behavior. If two sibling nodes have nearly
the same local rollout distribution, expanding both subtrees is wasteful; one
node can reuse the value estimate of the other.

This version removes the global answer set from the ValueShare trigger. Sharing
is based on intrinsic model signals from sibling rollouts generated during tree
expansion. Pruning remains separate and can still use the answer-set probe.

## Tree Setup

For a problem prompt, the policy constructs a reasoning tree. A node `s`
corresponds to a partial trajectory:

$$
\tau(s) = x \oplus z_1 \oplus \cdots \oplus z_d
$$

Here, each `z_i` is a generated reasoning segment. Let the branch factor at a
parent node `p` be `W`. Expanding `p` produces siblings:

$$
S(p) = \{s_1, \ldots, s_W\}
$$

The new ValueShare trigger only compares nodes inside the same sibling set.
This keeps the comparison local and avoids a global nearest-neighbor index
based on an external probe set.

## Rollout Continuation Set

For each sibling `s_i`, generate a small rollout set from that node:

$$
G_i = \{g_{i,1}, \ldots, g_{i,m_i}\}
$$

The elements of `G_i` can be immediate children, grandchildren, or short
descendant rollouts from `s_i`. In the current implementation, the next expanded
layer is used as the rollout probe and is reused if `s_i` is not shared.

This means a node can still be cut after probe descendants are generated. The
descendants serve as Monte Carlo samples for the sharing test; if the node is
marked SHARE, the probe subtree is discarded and not used for training edges.

## Sampled Local TV

For two siblings `s_i` and `s_j`, define the shared candidate support:

$$
C_{ij} = G_i \cup G_j
$$

For every continuation `c` in the candidate support, score the same continuation
under both sibling prefixes:

$$
L_i(c) = \log \pi_\theta(c \mid \tau(s_i))
$$

$$
L_j(c) = \log \pi_\theta(c \mid \tau(s_j))
$$

Normalize the scores on the sampled support:

$$
\hat p_i(c)
=
\frac{\exp(L_i(c))}
{\sum_{c' \in C_{ij}} \exp(L_i(c'))}
$$

$$
\hat p_j(c)
=
\frac{\exp(L_j(c))}
{\sum_{c' \in C_{ij}} \exp(L_j(c'))}
$$

The sampled local total variation distance is:

$$
\widehat{TV}_{C_{ij}}(s_i,s_j)
=
\frac{1}{2}
\sum_{c \in C_{ij}}
\left|
\hat p_i(c) - \hat p_j(c)
\right|
$$

The implementation computes this with a stable log-softmax-style shift, so very
large or very small log probabilities do not overflow.

## Share Rule

Let the sampled support size be:

$$
n_{ij} = |C_{ij}|
$$

A Hoeffding-style concentration radius is:

$$
r(n_{ij}, \alpha)
=
\sqrt{
\frac{\log(2/\alpha)}
{2n_{ij}}
}
$$

Given tolerated value error `epsilon` and reward bound `R_max`, define:

$$
\eta_{\text{share}}
=
\frac{\epsilon}{R_{\max}}
$$

The conservative share rule is:

$$
\widehat{TV}_{C_{ij}}(s_i,s_j)
+
r(n_{ij}, \alpha)
\leq
\eta_{\text{share}}
$$

The implementation exposes the confidence term as an option. By default, the
online rule drops the radius because small rollout supports make the radius too
conservative for the current default epsilon:

$$
\widehat{TV}_{C_{ij}}(s_i,s_j)
\leq
\eta_{\text{share}}
$$

When the selected rule holds, one sibling is marked SHARE and points to the
other sibling as its value target. The shared node is kept in the tree for
accounting, but its probe descendants are removed and recursion below it stops.

## Candidate Pair Budget

Comparing all sibling pairs costs quadratic pair evaluations in `W`. To reduce
latency, the implementation evaluates only a fixed fraction of sibling pairs.
The default budget is:

$$
B
=
\min
\left(
\frac{W(W-1)}{2},
\left\lfloor 0.25 W^2 \right\rceil
\right)
$$

This is approximately:

$$
\left(\frac{W}{2}\right)^2
$$

Pairs are ranked by a cheap precomputed score. When the prune engine is active,
the score is the node's fast average log probability. Otherwise, the
implementation falls back to a simple text-length proxy. Only the closest `B`
pairs under this cheap score are fully scored with sampled local TV.

The expensive cross-scoring cost is therefore approximately:

$$
O(Bm)
$$

instead of:

$$
O(W^2m)
$$

where `m` is the sampled continuation budget.

## Value Error Bound

Let `P_H(. | s)` be the true distribution over `H`-step rollout continuations
from node `s`. For any bounded downstream value or reward function `f`:

$$
|f(c)| \leq R_{\max}
$$

The standard TV inequality gives:

$$
\left|
\mathbb{E}_{c \sim P_H(\cdot \mid s_i)}[f(c)]
-
\mathbb{E}_{c \sim P_H(\cdot \mid s_j)}[f(c)]
\right|
\leq
R_{\max}
TV(P_H(\cdot \mid s_i), P_H(\cdot \mid s_j))
$$

The implementation estimates this TV using the sampled support. The total
estimation error decomposes as:

$$
\left|
\widehat{TV}_{C_{ij}}
-
TV(P_H^i, P_H^j)
\right|
\leq
\epsilon_{\text{MC}}
+
\epsilon_H
+
\epsilon_B
$$

The terms are:

- `epsilon_MC`: Monte Carlo error from finitely many sampled continuations.
- `epsilon_H`: truncation error from finite-depth rollouts.
- `epsilon_B`: candidate-selection bias from evaluating only `B` sibling pairs.

A practical concentration term for the Monte Carlo component is:

$$
\epsilon_{\text{MC}}
=
O
\left(
\sqrt{
\frac{\log(1/\delta)}
{n_{ij}}
}
\right)
$$

With the confidence option enabled, a sufficient operational condition for
sharing is:

$$
R_{\max}
\left(
\widehat{TV}_{C_{ij}}
+
r(n_{ij}, \alpha)
+
\epsilon_H
+
\epsilon_B
\right)
\leq
\epsilon
$$

In code, the rollout-depth error and pair-selection bias are controlled by the
rollout budget and pair budget rather than explicitly estimated. The default
practical rule is:

$$
\widehat{TV}_{C_{ij}}
\leq
\frac{\epsilon}{R_{\max}}
$$

## Online Algorithm

For each parent node `p`:

1. Generate `W` sibling children.
2. Run the prune trigger for each child when enabled.
3. For each remaining expandable sibling, generate one probe descendant layer.
4. Rank sibling pairs by the cheap precomputed score.
5. Evaluate only the top `B` pairs, with `B` approximately `(W/2)^2`.
6. For each evaluated pair, compute sampled local TV on the union of their
   rollout continuations.
7. If the selected share rule passes, mark one sibling as SHARE and stop
   recursion below it.
8. Recurse only into siblings that remain EXPAND.

This spends a small probe budget to identify redundant siblings, then avoids
expanding redundant subtrees to full depth.

## Why This Replaces the Global Answer Set for ValueShare

The previous ValueShare trigger used a global set of full solutions and raw
sequence probabilities:

$$
\pi_\theta(y \mid \tau(s))
$$

For long full solutions, these probabilities can be extremely small. The
residual mass outside the answer set then dominates the TV estimate, making the
estimate too loose to justify sharing.

The sibling-local trigger instead compares the model's own local continuation
distribution around the nodes being considered. This better matches the desired
behavioral question: do these two sibling states lead the policy toward the
same rollout distribution?

The method is therefore an intrinsic policy-distribution signal for approximate
state merging in LLM reasoning trees.
