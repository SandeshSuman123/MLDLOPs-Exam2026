import re
from sacrebleu.metrics import BLEU

def strip_rtf(text):
    text = re.sub(r'\{\\colortbl[^}]*\}', '', text)
    text = re.sub(r'\{\\fonttbl[^}]*\}', '', text)
    text = re.sub(r'\{\\\*[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+[-]?\d*\s?', ' ', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\', '', text)
    lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 2]
    return lines

def load_file(path):
    """Load either .rtf or .txt file and return lines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    # If RTF file, strip formatting
    if path.endswith(".rtf") or raw.startswith("{\\rtf"):
        lines = strip_rtf(raw)
    else:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines

def compute_bleu(hypothesis_path, reference_path):
    hypotheses = load_file(hypothesis_path)
    references = load_file(reference_path)

    print(f"Hypothesis lines : {len(hypotheses)}")
    print(f"Reference lines  : {len(references)}")

    # Make sure both have same number of lines
    min_len = min(len(hypotheses), len(references))
    hypotheses = hypotheses[:min_len]
    references = references[:min_len]

    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [references])

    print(f"\nBLEU Score: {result}")
    print(f"BLEU Score (numeric): {result.score:.2f}")
    return result.score

if __name__ == "__main__":
    
    compute_bleu("output.txt", "output.rtf")
