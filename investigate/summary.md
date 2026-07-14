# ILP model behind `solve_cover_ics_oneshot`

> Companion to `ilp_cover_ics.py`. Explains the ILP that both `solve_cover_ics`
> (lazy cuts) and `solve_cover_ics_oneshot` (all cuts upfront) solve — they
> encode the *same* feasible region, just build it at different times. See
> `dl-separable/docs.md` §1.3 for where Cover-ICS sits relative to GISMO/GICS.

## 1. What problem is being modeled

Given a graph $\Gamma = (V, E)$, a max simultaneous-fire count $k$, and a
precision budget $B \ge k$, find the smallest sensor set $D \subseteq V$ such
that for every possible fire $U \subseteq V$ with $|U| \le k$:

1. **Soundness** — $U \subseteq \varphi_D(s_U)$: the decoder never rules out
   the true fire.
2. **Precision** — $|\varphi_D(s_U)| \le B$: the decoder's guess is never
   bigger than $B$ rooms.
3. **Non-triviality** — $s_U \neq \emptyset$ whenever $U \neq \emptyset$: a
   real fire always trips at least one sensor.

where $s_U = N^+_1(U) \cap D$ is the **signature** (which sensors in $D$ fire),
and $\varphi_D(s) = \bigcup \{W : |W|\le k,\ s_W = s\}$ is the natural decoder
(union of every fire set that would produce signature $s$).

This is `Cover-ICS`, a relaxation of the exact identifying-code problem
(GICS): GICS demands every fire set have a *unique* signature; Cover-ICS only
demands that fire sets sharing a signature don't collectively get too big
(≤ B).

## 2. Decision variables

$$z_v \in \{0, 1\} \quad \text{for each } v \in V$$

$z_v = 1$ means "place a sensor at $v$." The solved-for sensor set is
$D = \{v : z_v = 1\}$.

## 3. Objective

$$\min \sum_{v \in V} z_v$$

Smallest possible sensor set (`prob += pulp.lpSum(z.values())` in code).

## 4. Constraints

### 4.1 Non-triviality → domination (always encoded)

Proven equivalent to "$D$ dominates $\Gamma$" (see `dl-separable`/plan.md
§2.5, referenced in the module docstring): every vertex must have some sensor
in its closed neighborhood.

$$\sum_{u \in N^+_1(v)} z_u \ge 1 \qquad \forall v \in V$$

```python
for v in nodes:
    prob += pulp.lpSum(z[u] for u in closed_neighborhood(v, adj)) >= 1
```

### 4.2 Soundness (never encoded — automatic)

$U$ is always one of the sets unioned into $\varphi_D(s_U)$ by definition, so
Soundness holds for *any* $D$. No constraint needed.

### 4.3 Precision (the hard one)

This is where the two solvers differ in *when* they build the constraint,
not *what* the constraint says.

**The naive idea and why it's wrong.** You might think: for every pair of
fire sets $U_1, U_2$ with $|U_1 \cup U_2| > B$, force some sensor to tell them
apart. This is only *necessary*, not *sufficient* — Precision bounds the
union of the *whole* equivalence class of fire sets sharing a signature, and
a class can have every pairwise union $\le B$ while its total union exceeds
$B$.

**Counterexample** (star graph $K_{1,m}$, hub $h$, leaves $l_1,\dots,l_m$,
$k=1$, $B=2$, $D=\{h\}$): every single-leaf fire $\{l_i\}$ has signature
$\{h\}$ — identical for all $m$ leaves. Every *pair* $\{l_i\},\{l_j\}$ has
union size 2 (not $>B$), so pairwise-only checking adds no constraint and
wrongly accepts $D=\{h\}$. But the real decoded class is *all $m$ leaves at
once* — size $m$. This is `solve_cover_ics_naive()` in the code, kept only to
demonstrate the gap.

**The correct condition.** $D$ satisfies Precision iff for *every* subset $W$
of fire-sets with $|\bigcup W| > B$, at least one pair inside $W$ is
separated by $D$ (a sensor $v$ "separates" $U_1,U_2$ if exactly one of them
hits $v$'s closed neighborhood — `distinguishing_set()` in code). This holds
because the actual signature-equivalence classes under $D$ partition the
fire-sets, and two fire-sets sit in different classes *iff* $D$ separates
them — so an oversized $W$ either sits inside one class (whose own union is
then also oversized, violating Precision directly) or spans ≥2 classes,
which forces a separated pair inside $W$.

**Only minimal $W$ need checking.** It's enough to enforce this for
*inclusion-minimal* violating $W$ (no proper subset of $W$ also exceeds $B$):
any larger oversized $W$ contains a minimal-violating subset, and that
subset's separated pair is also a pair inside the larger $W$.

**Turning "some pair separated" into a linear constraint.** For a violating
group $\{m_1, \dots, m_t\}$, the condition is
$\text{OR}_{i<j}\big(\sum_{v \in \text{Dist}(m_i, m_j)} z_v \ge 1\big)$.
Because each disjunct is itself a "some sum $\ge 1$" statement over binary
variables, an OR of such statements is *exactly* equivalent to one sum over
the *union* of all the sets involved:

$$\sum_{v \,\in\, \bigcup_{i<j} \text{Dist}(m_i, m_j)} z_v \ge 1$$

This is the one linear cut added per violating group — no stronger, no
weaker than necessary. (An earlier version added a separate constraint per
*pair* instead of unioning them, which is strictly stronger than necessary
and can overshoot the true optimum — verified on $K_{1,4}$: it found size 4
instead of the true optimum 3.)

## 5. Where the two solvers diverge

| | `solve_cover_ics` (lazy) | `solve_cover_ics_oneshot` (this one) |
|---|---|---|
| When are Precision cuts found? | Discovered on demand: solve with domination only, inspect the resulting $D$'s actual signature classes (`find_violating_groups`), add cuts for any that violate $B$, re-solve. Repeat. | Computed once, before the model is ever solved, via `minimal_violating_groups()` — the graph-structural (not $D$-dependent) enumeration of every inclusion-minimal violating fire-set group. |
| `prob.solve()` calls | One per round (2–6 in this module's demos) | Exactly one |
| Cost driver | Per-round CBC subprocess spawn (this is what usually reads as "slow") | Enumerating `minimal_violating_groups()` — itself a real combinatorial search (Apriori-style: union-size-$\le B$ is downward closed, so minimal violators are found level by level, same monotonicity trick as frequent-itemset mining) |
| Scaling risk | Cuts are added lazily, so only ever as many as an actual candidate $D$ produced — bounded in practice | Can blow up even on modest graphs: a loose $B$ relative to $k$ leaves almost every fire-set pair "compatible," so the search frontier stays wide for longer. `max_frontier` (default 20,000) fails fast with a clear `RuntimeError` instead of hanging. |
| Feasible region solved | Identical — both are the same ILP, built at different times | Identical |

**Empirically** (this module's small demo graphs, 5–9 nodes): one-shot was
faster (~13ms vs 85–290ms), because most of the lazy version's wall time was
CBC subprocess startup per round, not ILP solve time itself. On a 12-node
random graph with a loose $B$, one-shot's upfront enumeration hit the
`max_frontier` guard in <1s where lazy would have kept working fine — the
crossover depends on how tight $B$ is relative to $k$, not graph size alone.

## 6. `CoverICSInfeasible`

Both solvers can raise this: if a violating group's fire-sets are *true
twins* (every pair inside it has an empty `distinguishing_set` — identical
closed neighborhoods), no sensor placement, at any size, can ever separate
them, and their union already exceeds $B$. No valid $D$ exists for that
$(k, B)$.
