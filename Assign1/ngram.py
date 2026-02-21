import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


UNK = "<UNK>"
BOS = "<s>"


class NGramModel:
    def __init__(self, n: int, alpha: float = 0.1) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        self.n = n
        self.alpha = alpha
        self.context_counts: Counter[Tuple[str, ...]] = Counter()
        self.ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()
        self._argmax_cache: Dict[Tuple[str, ...], Tuple[str, float]] = {}

    def _pad_context(self, prev_tokens: Sequence[str]) -> Tuple[str, ...]:
        need = self.n - 1
        if len(prev_tokens) >= need:
            return tuple(prev_tokens[-need:])
        return tuple([BOS] * (need - len(prev_tokens)) + list(prev_tokens))

    def fit(self, methods: List[List[str]], vocab: set[str]) -> None:
        self.vocab = set(vocab)
        self.vocab.add(UNK)

        for method in methods:
            mapped = [tok if tok in self.vocab else UNK for tok in method]
            for i, token in enumerate(mapped):
                context = self._pad_context(mapped[:i])
                self.context_counts[context] += 1
                self.ngram_counts[context + (token,)] += 1

    def probability(self, context: Sequence[str], token: str) -> float:
        if token not in self.vocab:
            token = UNK
        ctx = self._pad_context(context)
        count_ctx = self.context_counts[ctx]
        count_ng = self.ngram_counts[ctx + (token,)]
        v = len(self.vocab)
        return (count_ng + self.alpha) / (count_ctx + self.alpha * v)

    def predict_next(self, context: Sequence[str]) -> Tuple[str, float]:
        ctx = self._pad_context(context)
        if ctx in self._argmax_cache:
            return self._argmax_cache[ctx]

        best_token = UNK
        best_prob = -1.0
        for tok in self.vocab:
            p = self.probability(ctx, tok)
            if p > best_prob:
                best_prob = p
                best_token = tok

        self._argmax_cache[ctx] = (best_token, best_prob)
        return best_token, best_prob



def read_tokenized_methods(path: Path, min_tokens: int = 10, dedup: bool = True) -> List[List[str]]:
    seen = set()
    methods: List[List[str]] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ascii_line = raw.encode("ascii", errors="ignore").decode("ascii").strip()
        if not ascii_line:
            continue
        toks = ascii_line.split()
        if len(toks) < min_tokens:
            continue

        key = " ".join(toks)
        if dedup and key in seen:
            continue
        seen.add(key)
        methods.append(toks)

    return methods



def split_dataset(
    methods: List[List[str]],
    val_size: int = 1000,
    test_size: int = 1000,
    t1_cap: int = 15000,
    t2_cap: int = 25000,
    t3_cap: int = 35000,
    seed: int = 42,
) -> Dict[str, List[List[str]]]:
    if len(methods) < (val_size + test_size + 100):
        raise ValueError(
            f"Need more data. Found {len(methods)} methods after filtering; "
            f"need at least {val_size + test_size + 100}."
        )

    rng = random.Random(seed)
    pool = list(methods)
    rng.shuffle(pool)

    val = pool[:val_size]
    test = pool[val_size : val_size + test_size]
    train_pool = pool[val_size + test_size :]

    t1 = train_pool[: min(t1_cap, len(train_pool))]
    t2 = train_pool[: min(t2_cap, len(train_pool))]
    t3 = train_pool[: min(t3_cap, len(train_pool))]

    return {"T1": t1, "T2": t2, "T3": t3, "V": val, "Te": test}



def build_vocab(methods: List[List[str]]) -> set[str]:
    vocab = set()
    for m in methods:
        vocab.update(m)
    return vocab



def map_oov(methods: List[List[str]], vocab: set[str]) -> List[List[str]]:
    mapped = []
    for m in methods:
        mapped.append([tok if tok in vocab else UNK for tok in m])
    return mapped



def perplexity(model: NGramModel, methods: List[List[str]]) -> float:
    total_log_prob = 0.0
    total_tokens = 0

    for method in methods:
        for i, gt in enumerate(method):
            context = method[:i]
            p = model.probability(context, gt)
            total_log_prob += math.log(p)
            total_tokens += 1

    if total_tokens == 0:
        raise ValueError("Cannot compute perplexity on empty token stream")

    return math.exp(-total_log_prob / total_tokens)



def build_predictions_for_method(model: NGramModel, method_tokens: List[str]) -> List[dict]:
    preds = []
    ctx_size = model.n - 1
    for i, gt in enumerate(method_tokens):
        padded = [BOS] * max(0, ctx_size - i) + method_tokens[max(0, i - ctx_size) : i]
        pred_token, pred_prob = model.predict_next(padded)
        preds.append(
            {
                "context": padded,
                "predToken": pred_token,
                "predProbability": round(float(pred_prob), 6),
                "groundTruth": gt,
            }
        )
    return preds



def write_results_json(
    out_path: Path,
    test_set_name: str,
    model: NGramModel,
    methods: List[List[str]],
) -> None:
    ppl = perplexity(model, methods)
    data = []

    for idx, method in enumerate(methods, start=1):
        data.append(
            {
                "index": f"ID{idx}",
                "tokenizedCode": " ".join(method),
                "contextWindow": model.n,
                "predictions": build_predictions_for_method(model, method),
            }
        )

    payload = {"testSet": test_set_name, "perplexity": round(float(ppl), 6), "data": data}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def train_and_select(
    splits: Dict[str, List[List[str]]],
    n_values: Iterable[int],
    alpha: float,
) -> Tuple[NGramModel, dict]:
    best_model = None
    best_meta = None

    for train_name in ["T1", "T2", "T3"]:
        train_set = splits[train_name]
        vocab = build_vocab(train_set)

        val_set = map_oov(splits["V"], vocab)
        train_mapped = map_oov(train_set, vocab)

        for n in n_values:
            model = NGramModel(n=n, alpha=alpha)
            model.fit(train_mapped, vocab)
            val_ppl = perplexity(model, val_set)

            meta = {
                "train_set": train_name,
                "n": n,
                "alpha": alpha,
                "vocab_size": len(vocab),
                "val_perplexity": val_ppl,
            }

            if best_meta is None or val_ppl < best_meta["val_perplexity"]:
                best_meta = meta
                best_model = model

            print(
                f"train={train_name} n={n} alpha={alpha} "
                f"vocab={len(vocab)} val_ppl={val_ppl:.6f}"
            )

    if best_model is None or best_meta is None:
        raise RuntimeError("No model trained")

    return best_model, best_meta



def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram code token model for Java methods")
    parser.add_argument("--methods", type=Path, default=Path("methods.txt"), help="Tokenized methods file")
    parser.add_argument("--provided-test", type=Path, default=None, help="Optional provided test .txt")
    parser.add_argument("--self-test", type=Path, default=None, help="Optional self-created test .txt")
    parser.add_argument("--min-tokens", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-values", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    methods = read_tokenized_methods(args.methods, min_tokens=args.min_tokens, dedup=True)
    print(f"Loaded {len(methods)} cleaned methods from {args.methods}")

    splits = split_dataset(
        methods,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    best_model, best_meta = train_and_select(splits=splits, n_values=args.n_values, alpha=args.alpha)
    print("Best configuration:", json.dumps(best_meta, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_train_vocab = build_vocab(splits[best_meta["train_set"]])
    split_test = map_oov(splits["Te"], best_train_vocab)
    write_results_json(
        out_path=args.out_dir / "results-split-test.json",
        test_set_name="split_test",
        model=best_model,
        methods=split_test,
    )
    print(f"Wrote {args.out_dir / 'results-split-test.json'}")

    if args.provided_test is not None:
        provided = read_tokenized_methods(args.provided_test, min_tokens=args.min_tokens, dedup=False)
        provided = map_oov(provided, best_train_vocab)
        write_results_json(
            out_path=args.out_dir / "results-provided.json",
            test_set_name=str(args.provided_test.name),
            model=best_model,
            methods=provided,
        )
        print(f"Wrote {args.out_dir / 'results-provided.json'}")

    if args.self_test is not None:
        self_test = read_tokenized_methods(args.self_test, min_tokens=args.min_tokens, dedup=False)
        self_test = map_oov(self_test, best_train_vocab)
        write_results_json(
            out_path=args.out_dir / "results-self.json",
            test_set_name=str(args.self_test.name),
            model=best_model,
            methods=self_test,
        )
        print(f"Wrote {args.out_dir / 'results-self.json'}")


if __name__ == "__main__":
    main()
