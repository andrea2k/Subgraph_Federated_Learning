# Evaluation rules

| Protocol | Model selection | Test labels | Historical name |
|---|---|---|---|
| AA | Available validation labels | Available labels | VV |
| AC | Available validation labels | Complete labels | VO |
| CC | Complete validation labels | Complete labels | OO |

AC is the final thesis's main comparison. AA and AC use exactly the same selected state. Test labels never select a model.

The selector is validation Macro PR-AUC, maximized. Each task uses trapezoidal precision–recall curve area, including tied-score handling; this is different from average precision. A task with no positive evaluation labels has `NaN`, not zero. The client macro averages defined tasks, and the system result weights defined clients by evaluated node count.

Federated methods select one round containing the deployable system state. Fully Local selects an epoch independently per client. Oracle physical-community federations select independently per community.

Results report the mean and sample standard deviation over seeds 0, 1, 2. Ablation differences are computed within matched seeds before summarizing. The centralized paper reference pools q within each seed before computing its final summary.

q changes positive-label availability, not graph structure. All negatives remain available. Hidden positive labels are masked out of the loss; they are not relabeled as negatives. The same allocation rule applies to train, validation, and test.

Frozen CSVs and analysis tables retain historical protocol IDs for traceability. New training outputs include an AA/AC/CC `protocol` column, and the main figures use the final thesis terminology.
