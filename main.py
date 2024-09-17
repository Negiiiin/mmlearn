from mmlearn.modules.decoders.preprocessor import UnifiedIOPreprocessor


preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="/h/negin/llama/tokenizer.model")
print(f"done!")