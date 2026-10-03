# Recorded result exports

`recorded/required/` contains the six main metric files, `scores.csv`, and
compact gzip-compressed prediction files. Each prediction preserves generated
text, example ID, numerical gold, extracted answer, and automatic scoring flags.

`recorded/extensions_v1/` covers LoRA capacity and generation length.
`recorded/extensions_v2/` covers calculator-annotation removal, a repeated-GSM
control, MathQA mixture, and error analysis. Their prediction files have the
same compact format. `optional_scores.csv` contains all 18 additional rows.

Repeated source prompts and worked gold solutions are omitted from bulk
exports. Metric values and generated text are retained. Numerical gold is
stored as its final `####` line, preserving exact scoring behavior. Model and
dataset revisions, input hashes, training histories, and filter counts remain
available in the logs and audit JSON files.

Historical fingerprints refer to the recorded run. The portable source notebook
uses project-relative paths and fixed upstream revisions, so it has its own
code hash. `export_provenance.json` records both hashes and the transformations.

`python scripts/verify_recorded_results.py` re-parses all generated answers,
recomputes their scores, and checks the CSV summaries and paired counts.
