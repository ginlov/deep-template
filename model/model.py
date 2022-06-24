from base import BaseModel
from transformers import AlbertConfig, AlbertModel
from torch.nn import Sequential

import torch


class SpellTransformer(BaseModel):
    def __init__(self, word_transformer_config, char_transformer_config):
        self.word_albert_config = AlbertConfig(**word_transformer_config)
        self.char_albert_config = AlbertConfig(**char_transformer_config)

        self.word_albert = AlbertModel(self.word_albert_config)
        self.char_albert = AlbertModel(self.char_albert_config)

        self.detectorNet = Sequential()
        self.correctorNet = Sequential()

    def forward(self, input_char, input_word):
        batch, length_sen, length_word = input_char.shape
        output_char = self.char_albert(input_char.view(batch, length_sen*length_word))
        char_embedding = output_char.last_hidden_state.view(batch, length_sen, length_word,
                                                            self.char_albert_config.hidden_size)
        char_word_embedding = char_embedding.mean(dim=2)
        input_word_albert = torch.cat([input_word, char_word_embedding], dim=-1)
        output_word = self.word_albert(input_word_albert).last_hidden_state
        detector_output = self.detectorNet(output_word)
        corrector_output = self.correctorNet(output_word)

        return detector_output, corrector_output
