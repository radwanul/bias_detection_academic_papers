import argparse
from dataset_loader import load_and_standardize
from preprocessing import get_tokenizer, tokenize_dataset, save_processed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="HF dataset name, e.g., allenai/real-toxicity-prompts")
    ap.add_argument("--task", default="binary", choices=["binary", "regression", "multilabel"], help="Task type")
    ap.add_argument("--score_key", default=None, help="Column used for binary/regression when needed (e.g., toxicity)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for binary label")
    ap.add_argument("--model", default="bert-base-uncased", help="Tokenizer model name")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--out", default="data/processed/out")
    args = ap.parse_args()


ds, info = load_and_standardize(name=args.name, task=args.task, score_key=args.score_key, threshold=args.thr)
print("INFO:", info)


tok = get_tokenizer(args.model)
tokenized = tokenize_dataset(ds, tok, max_length=args.max_length)
path = save_processed(tokenized, args.out)
print("Saved to:", path)