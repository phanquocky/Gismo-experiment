# Research: approaches for Cover-ICS sensor placement + fast decoding

> Scope: literature/approach survey only, no implementation. Companion to
> `summary.md` (which documents the ILP already implemented in
> `ilp_cover_ics.py`).

## 1. Problem reframed

`summary.md` defines Cover-ICS as: graph $\Gamma=(V,E)$, max simultaneous
fire count $k$, precision budget $B \ge k$, find smallest $D \subseteq V$
such that for every fire $U \subseteq V$, $|U|\le k$:

1. **Soundness** — $U \subseteq \varphi_D(s_U)$ (automatic, no constraint needed)
2. **Precision** — $|\varphi_D(s_U)| \le B$
3. **Non-triviality** — $s_U \neq \emptyset$ whenever $U \neq \emptyset$ (= domination)

where $s_U = N^+_1(U) \cap D$.

This maps almost 1:1 onto two adjacent bodies of literature:

- **Identifying codes in graphs** (Karpovsky–Chakrabarty–Levitin 1998):
  domination + unique signature per vertex/set. `summary.md`'s Cover-ICS is a
  *relaxation* of the exact case — instead of requiring unique signatures
  (GICS), it only bounds the union of any signature-equivalence class by $B$.
  When $B=k$ this reduces to a **$(1,\le k)$-identifying code**
  (Ben-Haim & Litsyn 2005, "On a New Class of Codes for Identifying Vertices
  in Graphs").
- **Non-adaptive combinatorial group testing (CGT)**: each sensor $d\in D$ is
  a *test* with test set $T_d = N^+_1(d)$ (closed neighborhood); a test is
  positive iff $U \cap T_d \neq \emptyset$. $U$ is the "defective set"
  ($|U|\le k$). The Precision requirement ($|\varphi_D(s_U)|\le B$) is
  *exactly* the defining property of a **$(k,B)$-list-disjunct matrix**: a
  design where the decoder always outputs a superset of size $\le B$
  containing all true defectives.
- Because test sets are locked to closed neighborhoods of a graph rather than
  freely chosen subsets, this is specifically **graph-constrained group
  testing** (Karbasi–Zadimoghaddam–Cheraghchi–Mohajer, "Graph-Constrained
  Group Testing", IEEE Trans. Info. Theory 2012, arXiv:1001.1445).

Repo file names already in progress (`dl-separable/`, `Using_DL_Disjunct*`)
suggest a "d-separable vs d-disjunct" framing has been explored before —
this is precisely the group-testing vocabulary above.

The task splits into two genuinely separate subproblems:

- **(A)** offline design of $D$ (already has an ILP solution in
  `ilp_cover_ics.py`)
- **(B)** online decode: given observed $s_U$, compute $K \supseteq U$,
  $|K|\le B$, *fast* — this is the "encode algorithm" asked about.

## 2. Approaches for (A): designing D

| Approach | Idea | Trade-off |
|---|---|---|
| **ILP/CP (current)** | Domination + minimal-violating-group cuts, lazy or one-shot | Exact optimum; NP-hard in general (Charon–Hudry–Lobstein 2003 proved minimizing identifying/locating-dominating codes is NP-hard) — doesn't scale |
| **Naive SAT / MaxSAT encoding** (superseded, see §3) | Translate domination + minimal-violating-group cuts directly into CNF, solve with weighted partial MaxSAT (RC2) | Same encoding-size problem as ILP: still needs `minimal_violating_groups()` enumerated combinatorially upfront. SAT solver is faster per-call than CBC, but doesn't fix the real bottleneck |
| **GISMO-style Grouped Independent Support (recommended, see §3–4)** | Polynomial-size CNF ($O(k\|V\|+\|E\|)$), reduce sensor selection to finding a minimal independent-support group set via Padoa's theorem + incremental SAT | 40× larger graphs, up to 520× faster than ILP in the exact ($B=k$) case (IJCAI 2023 result); needs a new bounded-slack generalization for $B>k$ — proposed in §4 |
| **Column generation / Lagrangian relaxation** | Master problem = set cover over violating groups, pricing problem = find worst violator on demand | Standard for covering ILPs with exponentially many constraints; avoids enumerating all `minimal_violating_groups()` upfront like the one-shot solver does |
| **Greedy submodular** | Domination is set cover (submodular) → greedy gives $O(\log n)$-approx; iteratively add sensors to kill worst violation | Fast baseline / warm start for ILP, no optimality guarantee on Precision side |
| **Explicit disjunct-matrix constructions adapted to graph structure** (Kautz–Singleton, Porat–Rothschild) | In unconstrained CGT, explicit $d$-disjunct designs exist with $O(d^2\log n)$ tests | Not directly portable (test sets are graph-locked), but motivates a *design principle*: maximize pairwise symmetric-difference of closed neighborhoods when picking $D$, instead of pure ILP search |
| **Randomized construction + probabilistic method** | Sample $D$ with inclusion probability tuned to degree; prove whp Precision holds for large enough $|D|$ | Gives a fast theoretical upper bound on optimal $|D|$ before running exact ILP |

## 3. GISMO: what it actually does (not just "SAT")

`summary.md` already notes Cover-ICS sits relative to "GISMO/GICS" and the
repo has a working GISMO pipeline (`plan.md`, `parse_gismo_output.py`,
`run_task1.py` — docker-exec'd `gismo` binary on `.gcnf` files). It's worth
being precise about *why* GISMO beats plain ILP/MaxSAT, because it is not
"SAT instead of ILP" — it avoids the combinatorial constraint-enumeration
problem entirely, which is the actual bottleneck flagged in `summary.md`
(both the Apriori-style `minimal_violating_groups()` search and the
per-round CBC spawn).

Source: Ghosh, Meel et al., *"Solving the Identifying Code Set Problem with
Grouped Independent Support"*, IJCAI 2023 (arXiv:2306.15693). Tool:
[github.com/meelgroup/gismo](https://github.com/meelgroup/gismo).

**1. One polynomial-size CNF, not one constraint per fire-pair.** Instead of
generating a constraint per violating group of fire-sets (ILP/naive-SAT
approach, size blows up with $\binom{|V|}{k}^2$), GISMO builds a *single*
formula $F_k(X,Y,A)$:

- $x_v$: "node $v$ is on fire."
- $y_v$: "sensor at $v$ fires," defined once via
  $y_v \leftrightarrow \bigvee_{u \in N^+_1(v)} x_u$ — literally the same
  relation as this repo's `closed_neighborhood`/signature function, but
  written as a *propositional definition* rather than evaluated per fire-set.
- $\sum_v x_v \le k$ (cardinality constraint).
- $\mathcal G := \{G_v = \{x_v, y_v\} : v \in V\}$ — one **group** per node,
  pairing "is on fire" with "is sensed," so selecting sensor $v$ means
  keeping group $G_v$.

Size: $O(k|V|+|E|)$ clauses — polynomial in $k$, not combinatorial.

**2. Sensor selection = finding a minimal Grouped Independent Support
(GIS).** A subset $\mathcal I \subseteq \mathcal G$ is a GIS iff two
solutions of $F_k$ that agree on $\mathcal I$'s variables necessarily agree
on *all* variables — i.e. the groups in $\mathcal I$ *functionally
determine* the whole fire pattern. GISMO's Lemma 1: $D = \{v : G_v \in
\mathcal I\}$ is a GICS of $\Gamma$ iff $\mathcal I$ is a GIS of $F_k$. So
"minimum identifying code" becomes "minimum grouped independent support" —
a model-counting-adjacent problem with its own mature SAT tooling, distinct
from covering-ILP tooling.

**3. Minimality is checked via Padoa's theorem, not enumeration.** Group $G$
is redundant (droppable) iff its variables are *implicitly definable* from
the rest — classically checkable by unsatisfiability of a **twin formula**:
duplicate all non-fixed variables ($Z \to \hat Z$), conjoin
$F_k(Z,A)\wedge F_k(\hat Z,A)$, force every *other* selected group's
variables to agree between the twins (via assumption/indicator literals
$e_j$), and ask whether that also forces $G$'s variables to agree. If UNSAT
(forced to agree), $G$ is redundant. This is a **deletion-based minimal
independent-support search**: iterate over groups, try dropping each,
confirm redundancy with one *conflict-budget-limited* incremental SAT call
(anytime — trades completeness for speed, same spirit as the existing repo's
`max_frontier` guard). No fire-pair is ever explicitly listed.

**4. Payoff (from the paper).** Exact ($B=k$) case: handles graphs up to
21,363 nodes vs. 494 for ILP (~40×), up to 520× faster median solve time on
shared instances, solutions within ~10% of ILP-optimal (set-minimal, not
cardinality-minimal — same caveat as any deletion-based MUS-style method).

## 4. Adapting GISMO to Cover-ICS's B-bounded relaxation

**A first attempt at this (below, struck through in spirit) was wrong in
exactly the way `summary.md`'s Counterexample section warns about**, and is
worth keeping as a cautionary note. The first idea was: build a *twin*
formula — two copies $Z,\hat Z$ — and ask via a cardinality query whether
$|Z\cup\hat Z|$ can exceed $B$. That is a strictly **pairwise** check
($U_1,U_2$), and `summary.md` already proves pairwise checking is
insufficient: on $K_{1,m}$ with $D=\{h\}$, $k=1$, $B=2$, every *pair* of
leaves $\{l_i\},\{l_j\}$ unions to size 2 (not $>B$), so a 2-copy check
never fires — yet the true equivalence class is *all $m$ leaves at once*,
union size $m$. A twin formula is blind to this for exactly the same reason
`solve_cover_ics_naive()` is: **Precision bounds the union of the whole
signature-equivalence class, not any pairwise union within it.**

**The fix: duplicate the formula $B+1$ times, not twice.** The defining
property we actually need to check is on the *decoder*, not on a pair:

$$\varphi_D(s) = \bigcup\{W : |W|\le k,\ s_W = s\}, \qquad \text{violation} \iff |\varphi_D(s)| > B \text{ for some reachable } s$$

Crucially, $v \in \varphi_D(s)$ only requires *some* witnessing fire $W\ni v$
with $s_W = s$ to exist — different vertices in $\varphi_D(s)$ can come from
completely different, mutually inconsistent witnesses (in the star example,
each leaf is witnessed by its own disjoint singleton fire). So the query
needs $B+1$ *independent* existential witnesses sharing one signature, not
one joint fire of size $B+1$. Build $B+1$ copies of the base formula,
$X^{(1)},Y^{(1)},\dots,X^{(B+1)},Y^{(B+1)}$, each an independent solution of
$F_k$ (so each individually respects $|X^{(i)}|\le k$), chained to share one
signature on the candidate groups:

$$\Phi_B \;:=\; \bigwedge_{i=1}^{B+1} F_k\big(X^{(i)},Y^{(i)},A\big) \;\wedge\; \bigwedge_{i=2}^{B+1} \big(Y^{(i)}\!\restriction_D = Y^{(1)}\!\restriction_D\big)$$

then ask, with one small "distinct representatives" gadget (assignment
variables $sel_{i,v}$: copy $i$'s designated witness is $v$; $\sum_v
sel_{i,v}=1$ per copy; $sel_{i,v}\Rightarrow x^{(i)}_v$; $\sum_i sel_{i,v}\le
1$ per vertex to force the $B+1$ chosen vertices distinct across copies):

$$\Phi_B \wedge \bigwedge_i\Big(\sum_v sel_{i,v}=1\Big) \wedge \bigwedge_v\Big(\sum_i sel_{i,v}\le 1\Big) \;\overset{?}{\in} \text{SAT}$$

SAT ⇒ $B+1$ distinct vertices are each individually witnessable under one
shared signature ⇒ Precision violated for the current group set. UNSAT for
every signature (which the solver searches implicitly, since $Y^{(1)}|_D$ is
free, not fixed) ⇒ Precision holds. **This is the correct generalization**:
it replaces GISMO's exact-equality Padoa check ($2$-copy, forced equal) with
a bounded-witness-count check ($(B{+}1)$-copy, forced same signature, forced
distinct witnesses) — recovers exact GISMO when $B=k$ isn't quite right
either, but the point is it's the right *shape* of relaxation: from
"equality" (2 copies, all-or-nothing) to "at most $B$ distinct witnesses"
($B{+}1$ copies, counting), rather than from "equality" to "pairwise union
bound" (which is simply too weak, as the counterexample shows).

Verify against the counterexample: $K_{1,m}$, $D=\{h\}$, $k=1$, $B=2$ ⇒ this
builds $3$ copies. Assign $x^{(1)}_{l_1}=1,x^{(2)}_{l_2}=1,x^{(3)}_{l_3}=1$;
all three signatures are $\{h\}$ (equal); the three designated witnesses
$l_1,l_2,l_3$ are distinct ⇒ **SAT**, correctly flags $D=\{h\}$ as violating
Precision whenever $m\ge 3$ — exactly matching `summary.md`'s stated correct
condition, unlike the pairwise/twin version which never flags it at all.

**Cost.** Encoding size is $(B{+}1)\cdot O(k|V|+|E|)$ for the duplicated base
formula plus $O(B\cdot|D|)$ for signature-chaining plus $O(B\cdot|V|)$ for
the witness-assignment gadget — polynomial in $B,k,|V|,|E|$ for fixed $B$
(the usual regime, since $B$ is a small precision budget, not scaling with
$|V|$), so this still avoids the combinatorial-in-$\binom{|V|}{k}$ blowup
that motivated moving to GISMO's machinery in the first place. It's more
expensive than a 2-copy check by a factor of $\sim B$, which is the correct
price for correctness here — no construction that only compares $O(1)$
copies can detect a violation whose witnesses number $B{+}1$.

Wrap this oracle in the same deletion-based (drop redundant groups) or
insertion-based (grow $D$ from $\emptyset$) group search GISMO uses,
still with conflict-limited incremental SAT calls for anytime behavior; only
the inner oracle changes (2-copy Padoa/definability → $(B{+}1)$-copy
bounded-witness-count). This constructs, in effect, a
**$(k,B)$-list-disjunct-by-design** matrix (ties to §5) while keeping
GISMO's core property: still one symbolic SAT query per candidate group,
never an explicit enumeration of violating fire-set groups — but now an
actually-correct one.

**Practical next step**, given the repo already has a working GISMO
pipeline: (a) benchmark stock `gismo` as-is for the exact $B=k$ case as a
real empirical baseline (the `plan.md` Task1 checklist already sets this
up), (b) prototype the $(B{+}1)$-copy oracle above as a patch to the same
group-search loop, starting from small $B$ (2–3) where the blow-up factor is
still cheap, and cross-check against `find_violating_groups()` on the
existing small demo graphs (including the $K_{1,m}$ counterexample itself)
before trusting it on anything larger.

## 5. Approaches for (B): fast decode of K from s_U

This is the part worth prioritizing — the "efficient encode algorithm."

1. **Design D to be $(k,B)$-list-disjunct directly**, rather than only
   verifying Precision post hoc via ILP cuts. A list-disjunct matrix admits a
   closed-form decoder, the **COMP decoder** (Combinatorial Orthogonal
   Matching Pursuit):
   $$K = \bigcup \{v \in V : N^+_1(v)\cap D \subseteq s_U\}$$
   i.e. every vertex whose own signature is consistent with the observed
   signal. Runtime $O(|D|\cdot|V|)$, trivially parallel, no combinatorial
   search over fire-sets at query time — the list-disjunct property
   *guarantees* $|K|\le B$ by construction, so nothing needs to be solved
   online.
   - Indyk, Ngo, Rudra — *"Efficiently Decodable Non-adaptive Group
     Testing"*, SODA 2010.
   - Ngo, Porat, Rudra — *"Efficiently Decodable Error-Correcting List
     Disjunct Matrices and Applications"*, ICALP 2011.
   - Cheraghchi — *"Noise-Resilient Group Testing: Limitations and
     Constructions"*, arXiv:0811.2609 — defines $(d,\ell)$-list-disjunct
     matrices and proves near-linear decode time.

2. **If D is only Cover-ICS-feasible (not list-disjunct by construction)**:
   precompute a lookup table $s \mapsto \varphi_D(s)$ once, right after ILP
   solves for $D$. `find_violating_groups()` / `minimal_violating_groups()`
   already walk signature-equivalence classes to verify Precision — the
   table is close to a free byproduct of that pass. Online query becomes an
   $O(1)$ hash lookup instead of re-deriving $K$ each time.

3. **Noisy/faulty sensors**: if sensor readings can be wrong, the relevant
   generalization is **error-correcting list-disjunct matrices**
   (Cheraghchi, FCT 2009; Cheraghchi–Nakos) — decode stays sub-linear even
   with $e_0$ false positive and $e_1$ false negative sensor reports.

4. **Same meelgroup lineage, decode side: MGT** — *"A MaxSAT-Based Framework
   for Group Testing"* (Ciampiconi, Ghosh, Scutari, Meel — AAAI 2020, tool:
   [github.com/meelgroup/mgt](https://github.com/meelgroup/mgt)) solves the
   *decoding* phase of non-adaptive group testing with weighted partial
   MaxSAT: hard clauses pin each test outcome to the OR of its pool's
   defect variables, soft unit clauses penalize $x_v{=}1$ to prefer sparse
   explanations. Stock MGT returns one minimum-weight consistent defective
   set, not the full union $\varphi_D(s)$ needed here — to get the
   bounded-size superset $K$, it needs adapting into a *maximum realizable
   support* query: repeatedly re-solve while forcing previously-discovered
   $x_v{=}1$, or AllSAT-style enumerate up to $B$ distinct satisfying
   supports with blocking clauses, rather than stopping at the first
   minimum-weight model. The §4 $(B{+}1)$-copy oracle could unify both: the
   same "does a shared signature admit $\ge m$ distinct witnesses" query,
   run with $m=B+1$ at design time (does $D$ violate Precision) and with
   growing $m$ at decode time (enumerate $\varphi_D(s_U)$ itself, one
   witness at a time, stopping once no further distinct witness exists).

## 6. Recommended direction

Prioritize the **GISMO-style $(k,B)$-relaxed GIS** (§3–4) over the naive
ILP/MaxSAT encodings: the real bottleneck in both existing solvers
(`summary.md`'s Apriori-style `minimal_violating_groups()` enumeration, and
lazy ILP's per-round CBC spawn) is combinatorial constraint generation, and
that's precisely what GISMO's polynomial formula + Padoa-style definability
check is built to avoid. The naive 2-copy ("twin") extension of that check
is *wrong* for $B>k$ — it only catches pairwise violations, and
`summary.md`'s own $K_{1,m}$ counterexample shows pairwise checking misses
whole-class violations — so §4's $(B{+}1)$-copy bounded-witness-count oracle
is the one to build: it keeps GISMO's "one symbolic SAT query per candidate
group, no explicit fire-set enumeration" property while actually matching
Cover-ICS's real Precision definition. Separately, reframe the online decode
side (§5) as a **list-disjunct** design goal so the query-time algorithm is
a closed-form COMP formula (or, if reusing the GIS machinery, the same
$(B{+}1)$-copy oracle run as an enumerator, per the §5.4 note) — either way,
decode should not require re-deriving $K$ from scratch per query. The
existing `ilp_cover_ics.py` remains useful as a small-graph exact-optimum
baseline — and specifically as a correctness check for the new oracle, since
it already implements the correct (non-pairwise) condition and already has
the $K_{1,m}$ counterexample as a regression case. The repo's already-working
`gismo` docker pipeline (`plan.md` Task1) is the fastest path to an
empirical baseline for the exact $B=k$ case before building the $(k,B)$
extension.

## 7. References

- Karpovsky, Chakrabarty, Levitin (1998) — original identifying codes paper.
- Ben-Haim, Litsyn (2005) — *"On a New Class of Codes for Identifying
  Vertices in Graphs"* ($(1,\le\ell)$-identifying codes).
- Charon, Hudry, Lobstein (2003, Theoretical Computer Science) — NP-hardness
  of minimum identifying/locating-dominating codes.
- *"A comparison of approaches for finding minimum identifying codes on
  graphs"*, Quantum Information Processing, Springer 2016 — ILP vs SAT vs
  heuristics.
- Ghosh, Meel et al. — *"Solving the Identifying Code Set Problem with
  Grouped Independent Support"*, IJCAI 2023 (arXiv:2306.15693). Tool:
  [github.com/meelgroup/gismo](https://github.com/meelgroup/gismo). §3–4.
- Ciampiconi, Ghosh, Scutari, Meel — *"A MaxSAT-Based Framework for Group
  Testing"*, AAAI 2020. Tool:
  [github.com/meelgroup/mgt](https://github.com/meelgroup/mgt). §5.
- Ivrii, Malik, Meel, Vardi — *"On Computing Minimal Independent Support and
  Its Applications to Sampling and Counting"* (Constraints, 2016) /
  *"Faster Smarter Minimal Independent Support"* (IJCAI 2016) — the
  independent-support literature GISMO's group-search builds on; background
  for Padoa's-theorem-based definability checking via incremental SAT.
- Karbasi, Zadimoghaddam, Cheraghchi, Mohajer — *"Graph-Constrained Group
  Testing"*, IEEE Trans. Info. Theory 2012 (arXiv:1001.1445).
- Indyk, Ngo, Rudra — *"Efficiently Decodable Non-adaptive Group Testing"*,
  SODA 2010.
- Ngo, Porat, Rudra — *"Efficiently Decodable Error-Correcting List Disjunct
  Matrices and Applications"*, ICALP 2011.
- Cheraghchi — *"Noise-Resilient Group Testing: Limitations and
  Constructions"*, arXiv:0811.2609.
- Du, Hwang — *Combinatorial Group Testing and Its Applications* (standard
  textbook reference for disjunct/separable matrix theory).
