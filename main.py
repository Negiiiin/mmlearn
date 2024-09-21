# from mmlearn.modules.decoders.preprocessor import UnifiedIOPreprocessor


# preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="/h/negin/llama/tokenizer.model")
# print(f"done!")

from mmlearn.modules.decoders.LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
# from mmlearn.modules.decoders.LLaVA.llava import LlavaModel, LlavaConfig

# Load the LLaVA model
model_name = "/model-weights/llava-1.5-13b-hf"
model = LlavaConfig.from_pretrained(model_name)

# print(f" model.instances {model.instances}")

print(f"DONE!")