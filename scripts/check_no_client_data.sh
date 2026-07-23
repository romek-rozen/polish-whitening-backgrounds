#!/usr/bin/env bash
# Refuse a commit that would publish client keyword data.
#
# This repo is public. The clustering work runs on a private keyword export
# whose phrases include brand terms. Cluster *labels* are literally those
# keywords — the most central phrase of each group — so a "harmless" example
# pasted into a docstring or a report can publish a client's brand keywords.
#
# That has already happened twice in this repo (an examples/ dump, and a
# docstring illustration), both caught by hand. Hand-checking does not scale
# across sessions and agents, so this runs as a pre-commit hook instead.
#
# It is a blunt instrument on purpose: it greps staged content for phrases seen
# in the client data. False positives are expected and are cheap — rewrite the
# example generically. Do not add narrow exceptions; widen the corpus of
# forbidden terms instead when a new dataset arrives.
#
# Usage:
#   bash scripts/check_no_client_data.sh          # check staged content
#   git commit ...                                # runs automatically via hook
#
# Install the hook:
#   ln -sf ../../scripts/check_no_client_data.sh .git/hooks/pre-commit

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phrases observed in the client export. Case-insensitive, partial matches.
# Keep this list in sync when a new private dataset is analysed.
FORBIDDEN='mbti|osobowoś|osobowosc|szkolenie celne|szkolenia celne|numer eori|\beori\b|\bcbam\b|\beudr\b|intrastat|vademecum|nęcki|nęcki|flegmatyk|melancholik|sangwinik|choleryk|konosament|isztar|asertywn|krupówk|krupowk|lazy girl job|shift shock|job crafting'

fail=0

# 1. Nothing from a work/ directory may ever be committed.
work_staged="$(git diff --cached --name-only | grep '/work/\|^work/' || true)"
if [ -n "$work_staged" ]; then
    echo "BLOCKED: files from a work/ directory are staged:" >&2
    echo "$work_staged" | sed 's/^/    /' >&2
    echo "  work/ holds the client's keyword data and must stay untracked." >&2
    fail=1
fi

# 2. No staged *content* may contain client phrases — including files that are
#    otherwise legitimate, like a README or a docstring.
hits="$(git diff --cached -U0 | grep '^+' | grep -viE '^\+\+\+' | grep -iE "$FORBIDDEN" || true)"
if [ -n "$hits" ]; then
    echo "BLOCKED: staged content contains client keyword phrases:" >&2
    echo "$hits" | head -20 | sed 's/^/    /' >&2
    echo "  Cluster labels ARE the client's keywords, brand terms included." >&2
    echo "  Rewrite the example generically, e.g. '<term> co to' instead of a real phrase." >&2
    fail=1
fi

# 3. Binary artefacts built from the data (xlsx/parquet/npy) are never source.
binaries="$(git diff --cached --name-only | grep -iE '\.(xlsx|parquet|npy|csv)$' || true)"
for path in $binaries; do
    case "$path" in
        # The repo's own product: fitted whitening matrices. Numeric, derived
        # from a public corpus, no keywords.
        backgrounds/*/*.npy) ;;
        # Aggregate sweep statistics carry no keywords; those are allowed.
        */examples/*_sweep.csv|*/examples/*sweep*.xlsx) ;;
        *) echo "BLOCKED: $path looks like a data artefact — is it keyword-free?" >&2
           echo "  If it genuinely holds only aggregate statistics, add it to the" >&2
           echo "  allow-list in scripts/check_no_client_data.sh with a reason." >&2
           fail=1 ;;
    esac
done

if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "Commit refused. Fix the above, or bypass with --no-verify ONLY if you" >&2
    echo "have personally verified the content contains no client data." >&2
    exit 1
fi

echo "check_no_client_data: OK"
exit 0
