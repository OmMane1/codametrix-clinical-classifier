import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path("runs/biomedbert_mtsamples_eval/checkpoint-171")
DEFAULT_TOKENIZER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

ID_TO_CATEGORY = {
    0: "Cardiology",
    1: "Neurology",
    2: "Orthopedics",
    3: "Gastroenterology",
    4: "Other",
}


class TestCaseDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int,
        truncation_strategy: str = "head",
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self) -> int:
        return len(self.texts)

    def encode_head_tail(self, text: str) -> dict[str, torch.Tensor]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        token_budget = self.max_length - special_tokens

        if len(token_ids) > token_budget:
            head_budget = token_budget // 2
            tail_budget = token_budget - head_budget
            token_ids = token_ids[:head_budget] + token_ids[-tail_budget:]

        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        attention_mask = [1] * len(token_ids)

        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        item = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if "token_type_ids" in self.tokenizer.model_input_names:
            item["token_type_ids"] = torch.zeros(self.max_length, dtype=torch.long)

        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.texts[index]

        if self.truncation_strategy == "head_tail":
            return self.encode_head_tail(text)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def parse_test_cases(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?ms)^Case\s+(\d+)\s*\n(.*?)(?=^Case\s+\d+\s*\n|\Z)"
    )

    rows = []
    for match in pattern.finditer(text):
        case_number = int(match.group(1))
        case_text = re.sub(r"\s+", " ", match.group(2)).strip()
        if case_text:
            rows.append({"case_number": case_number, "text": case_text})

    if not rows:
        raise ValueError(f"No cases found in {path}")

    return pd.DataFrame(rows).sort_values("case_number").reset_index(drop=True)


def predict(
    df: pd.DataFrame,
    model_dir: Path,
    tokenizer_name_or_path: str,
    max_length: int,
    batch_size: int,
    truncation_strategy: str,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    dataset = TestCaseDataset(
        df["text"].tolist(),
        tokenizer,
        max_length,
        truncation_strategy,
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    output = pd.DataFrame(
        {
            "case_number": df["case_number"],
            "category": [ID_TO_CATEGORY[label] for label in predictions],
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate example-submission formatted predictions for test cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("medical-transcript-test-cases.txt"),
        help="Text file containing Case N blocks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path with case_number,category.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Fine-tuned model checkpoint or best_model directory.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name/path. Defaults to the base BiomedBERT tokenizer.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--truncation-strategy",
        choices=["head", "head_tail"],
        default="head",
    )

    args = parser.parse_args()

    df = parse_test_cases(args.input)
    print(f"Parsed {len(df)} test cases from {args.input}")

    output = predict(
        df,
        args.model_dir,
        args.tokenizer,
        args.max_length,
        args.batch_size,
        args.truncation_strategy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")
    print(output["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
