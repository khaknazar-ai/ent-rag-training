"""Deterministic extractive highlights, not a generative model or quality proof."""

import re
from collections import Counter

from enterprise_rag.application.answer import ModelSelection
from enterprise_rag.domain.summaries import SUMMARY_BUDGET, SUMMARY_REVISION


class LocalExtractiveSummary:
    kind = "deterministic"
    revision = SUMMARY_REVISION

    def summarize(self, snippets):
        # Candidates remain exact substrings, never cut down to the quote cap.
        # Conservatively omit boundary fragments instead of splicing chunks.
        candidates = []
        for position, item in enumerate(snippets):
            for match in re.finditer(
                r"[^.!?\u3002\uff01\uff1f]+(?:[.!?\u3002\uff01\uff1f]+|$)", item.text
            ):
                quote = match.group().strip()
                if (position > 0 and match.start() == 0) or (
                    position < len(snippets) - 1
                    and match.end() == len(item.text)
                    and quote[-1:] not in ".!?\u3002\uff01\uff1f"
                ):
                    continue
                if not quote or len(quote) > SUMMARY_BUDGET.max_quote_chars:
                    continue
                words = frozenset(re.findall(r"\w+", quote.casefold()))
                if words:
                    candidates.append((item.evidence_id, quote, words))
        frequency = Counter(word for _, _, words in candidates for word in words)
        chosen: list[int] = []
        seen: set[str] = set()
        while len(chosen) < SUMMARY_BUDGET.max_points:
            remaining = [
                (i, sum(frequency[word] for word in words - seen) / len(words))
                for i, (_, quote, words) in enumerate(candidates)
                if i not in chosen and quote not in {candidates[j][1] for j in chosen}
            ]
            if not remaining:
                break
            index, _ = max(remaining, key=lambda pair: (pair[1], -pair[0]))
            chosen.append(index)
            seen.update(candidates[index][2])
        return tuple(
            ModelSelection(False, candidates[i][0], candidates[i][1])
            for i in sorted(chosen)
        )
