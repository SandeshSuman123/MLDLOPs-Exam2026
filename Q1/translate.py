from transformers import MarianMTModel, MarianTokenizer

def translate_file(input_path, output_path):
    model_name = "Helsinki-NLP/opus-mt-bn-en"
    print(f"Loading model: {model_name}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} lines to translate...")
    print("\nFirst 3 input lines:")
    for l in lines[:3]:
        print(" ", l)
    print()

    translated_lines = []
    batch_size = 16

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_lines.extend(decoded)
        print(f"  Translated {min(i+batch_size, len(lines))}/{len(lines)} lines")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

    print(f"\nTranslation complete! Output saved to: {output_path}")
    print(f"\nFirst translated line:\n{translated_lines[0]}")
    return translated_lines

if __name__ == "__main__":
    translate_file("input.txt", "output.txt")
