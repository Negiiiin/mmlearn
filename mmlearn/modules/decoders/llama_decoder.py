import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from mmlearn.modules.decoders.config import T5Config

embed_size = T5Config.emb_dim
vocab_size = T5Config.vocab_size

class CustomLLaMAModel(nn.Module):
    def __init__(self, pretrained_llama, embed_size):
        super(CustomLLaMAModel, self).__init__()
        self.llama = pretrained_llama
        class CustomEmbeddingLayer(nn.Module):
            def __init__(self, vocab_size, embed_size):
                super(CustomEmbeddingLayer, self).__init__()
                self.embeddings = nn.Embedding(vocab_size, embed_size)

            def forward(self, input_ids):
                return self.embeddings(input_ids)
        custom_embedding_layer = CustomEmbeddingLayer(vocab_size=vocab_size, embed_size=embed_size)
        self.llama.transformer.wte = custom_embedding_layer

    def forward(self, text_features, image_features, labels=None):
        # Assuming text_features and image_features are already combined
        combined_features = torch.cat((text_features, image_features), dim=-1)

        # Pass combined features directly to transformer layers
        outputs = self.llama(inputs_embeds=combined_features, labels=labels)
        return outputs


    
    
# Load LLaMA model without the embedding layer
# pretrained_llama = LlamaForCausalLM.from_pretrained("llama-7b")
# llama_model = CustomLLaMAModel(pretrained_llama, embed_size=embed_size)



