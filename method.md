# Method: Sibling-Local Value Sharing and Pruning

## Motivation

Tree-based policy optimization spends a large generation budget on subtrees
that often become redundant. The goal of ValueShare is not merely to ask
whether two partial reasoning states look textually similar. It asks whether
the model would behave similarly after those states. If two sibling nodes induce
nearly the same local continuation distribution, expanding both descendants is
unnecessary; one node can reuse the value of the other.

This version removes the global answer set used by the original ValueShare
trigger. Instead, sharing is based on intrinsic model signals from sibling
rollouts generated during tree expansion.

Pruning remains a separate trigger. It can still use the answer-set probe to
stop branches whose likelihood over plausible final solutions drops sharply
relative to their parent.

## Tree Setup

For a problem prompt, the policy constructs a reasoning tree. A node $$s$$
corresponds to a partial trajectory:

$$
\tau(s) = x \oplus z_1 \oplus \cdots \oplus z_d
$$

where each $$z_i$$ is a generated reasoning segment. Let the branch factor at a
parent node $$p$$ be $$W$$. Expanding $$p$$ produces siblings:

$$
S(p) = \{s_1, \ldots, s_W\}
$$

The new ValueShare trigger only compares nodes inside the same sibling set
$$S(p)$$. This keeps the comparison local and avoids a global nearest-neighbor
index based on an external probe set.

## Rollout Continuation Set

For each sibling $$s_i$$, we first generate a small rollout set from that node:

$$
G_i = \{g_{i,1}, \ldots, g_{i,m_i}\}
$$

The elements of $$G_i$$ can be immediate children, grandchildren, or short
descendant rollouts from $$s_i$$. In practice, the implementation uses the next
expanded layer as the rollout probe and reuses those nodes if $$s_i$$ is not
shared.

Using descendants rather than only immediate next tokens gives a better Monte
Carlo estimate of whether two states have similar downstream behavior. A node
can still be cut after these probe descendants are generated. The descendants
serve as samples for the sharing test; if the node is marked SHARE, the probe
subtree is discarded and not used for training edges.

## Sampled Local TV

For two siblings $$s_i$$ and $$s_j$$, define the shared candidate support:

$$
C_{ij} = G_i \cup G_j
$$

For every continuation $$c \in C_{ij}$$, score the same continuation under both
prefixes:

$$
L_i(c) = \log \pi_\theta(c \mid \tau(s_i))
$$

$$
L_j(c) = \log \pi_\theta(c \mid \tau(s_j))
$$

The scores are normalized on the sampled support:

$$
\hat p_i(c) =
\frac{\exp(L_i(c))}
{\sum_{c' \in C_{ij}} \exp(L_i(c'))}
$$

$$
\hat p_j(c) =
\frac{\exp(L_j(c))}
{\sum_{c' \in C_{ij}} \exp(L_j(c'))}
$$

The sampled local total variation distance is:

$$
\widehat{TV}_{C_{ij}}(s_i,s_j)
= \frac{1}{2}\sum_{c \in C_{ij}}
\left|\hat p_i(c) - \hat p_j(c)\right|
$$

The implementation computes the normalization with a stable log-softmax-style
shift, so very large or very small log probabilities do not overflow.

## Share Rule

Let $$n_{ij} = |C_{ij}|$$. A concentration radius is:

$$
r(n_{ij}, \alpha)
= \sqrt{\frac{\log(2/\alpha)}{2n_{ij}}}
$$

Given a tolerated value error $$\epsilon$$ and reward bound $$R_{\max}$$, define:

$$
\eta_{\text{share}} =
\frac{\epsilon}{R_{\max}}
$$

or use an explicit override in ablations.

The conservative share rule is:

$$
\widehat{TV}_{C_{ij}}(s_i,s_j)
+ r(n_{ij}, \alpha)
\leq \eta_{\text{share}}
$$

The implementation exposes this confidence term as an option. By default, the
online rule uses $$\widehat{TV}_{C_{ij}} \leq \eta_{\text{share}}$$ because
small rollout supports make the Hoeffding radius overly conservative for the
current default $$\epsilon$$. When the confidence option is enabled, the rule
above is used.

When the selected rule holds, one sibling is marked SHARE and points to the
other sibling as its value target. The shared node is kept in the tree for
accounting, but its probe descendants are removed and recursion below it stops.

## Candidate Pair Budget

Comparing all sibling pairs costs $$O(W^2)$$ pair evaluations. To reduce
latency, the implementation evaluates only a fixed fraction of the pairs. The
default budget is:

$$
B =
\min\left(
\frac{W(W-1)}{2},
\left\lfloor 0.25 W^2 \right\rceil
\right)
$$

This is approximately $$\left(W/2\right)^2$$ candidate pairs.

Pairs are ranked by a cheap precomputed score. When the prune engine is active,
the score is the node's fast average log probability $$\text{AvgLP}_K$$.
Otherwise the implementation falls back to a simple text-length proxy. Only the
closest $$B$$ pairs under this cheap score are fully scored with sampled local
TV.

This keeps the expensive cross-scoring cost near:

$$
O(Bm)
$$

instead of:

$$
O(W^2m)
$$

where $$m$$ is the sampled continuation budget.

## Value Error Bound

Let $$P_H(\cdot \mid s)$$ be the true distribution over $$H$$-step rollout
continuations from node $$s$$. For any bounded downstream value or reward
function $$f$$ with:

$$
|f(c)| \leq R_{\max}
$$

the standard TV inequality gives:

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

The implementation estimates this TV using the sampled support $$C_{ij}$$. The
total estimation error decomposes into:

$$
\left|
\widehat{TV}_{C_{ij}} - TV(P_H^i, P_H^j)
\right|
\leq
\epsilon_{\text{MC}}
+ \epsilon_H
+ \epsilon_B
$$

where:

- $$\epsilon_{\text{MC}}$$ is the Monte Carlo error from using finitely many
  sampled continuations.
- $$\epsilon_H$$ is the truncation error from using finite-depth rollouts rather
  than full completions.
- $$\epsilon_B$$ is the candidate-selection bias from evaluating only $$B$$
  sibling pairs instead of all pairs.

A practical concentration term for the Monte Carlo component is:

$$
\epsilon_{\text{MC}}
=
O\left(
\sqrt{\frac{\log(1/\delta)}{n_{ij}}}
\right)
$$

The conservative trigger can use the explicit radius $$r(n_{ij}, \alpha)$$ as
a finite-sample guard. Therefore a sufficient operational condition for sharing
is:

$$
R_{\max}
\left(
\widehat{TV}_{C_{ij}}
+ r(n_{ij}, \alpha)
+ \epsilon_H
+ \epsilon_B
\right)
\leq
\epsilon
$$

In code, $$\epsilon_H$$ and $$\epsilon_B$$ are controlled by rollout depth,
rollout budget, and pair budget rather than explicitly estimated. With the
confidence option enabled, the online rule uses:

$$
\widehat{TV}_{C_{ij}} + r(n_{ij}, \alpha)
\leq
\frac{\epsilon}{R_{\max}}
$$

as the actionable share criterion. With the default confidence option disabled,
the practical rule drops the radius and uses:

$$
\widehat{TV}_{C_{ij}}
\leq
\frac{\epsilon}{R_{\max}}
$$

## Online Algorithm

For each parent node $$p$$:

1. Generate $$W$$ sibling children.
2. Run the prune trigger for each child when enabled.
3. For each remaining expandable sibling, generate one probe descendant layer.
4. Rank sibling pairs by the cheap precomputed score.
5. Evaluate only the top $$B \approx (W/2)^2$$ pairs.
6. For each evaluated pair, compute sampled local TV on $$G_i \cup G_j$$.
7. If the conservative share rule passes, mark one sibling as SHARE and stop
   recursion below it.
8. Recurse only into siblings that remain EXPAND.

This preserves the main budget-saving behavior: the method may spend a small
probe budget to identify redundant siblings, but it avoids expanding redundant
subtrees to full depth.

## Why This Replaces the Global Answer Set for ValueShare

The previous ValueShare trigger used a global set of full solutions $$Y$$ and
raw sequence probabilities $$\pi_\theta(y \mid \tau(s))$$. For long full
solutions, these probabilities can be extremely small, making the residual mass
outside $$Y$$ dominate the TV estimate. In that regime the estimate becomes too
loose to justify sharing.

The sibling-local trigger instead compares the model's own local continuation
distribution around the nodes being considered. This better matches the desired
behavioral question:

$$
\text{Do these two sibling states lead the policy toward the same next rollout
distribution?}
$$

The method is therefore an intrinsic policy-distribution signal for approximate
state merging in LLM reasoning trees.
