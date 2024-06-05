import tritonclient.grpc
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np
import concurrent.futures
from src.settings.settings import (
    TranslationSettings,
    TranslationTokenizerSettings,
    TranslationTritonServerSettings
)
from src.components.translate.base import BaseTranslation


translate_triton_server_settings = TranslationTritonServerSettings

class TritonServerTranslation(BaseTranslation):
    """
    CTranslate2 backend deployed on Triton server for inference.
    """

    triton_server_endpoint = translate_triton_server_settings.triton_server_endpoint
    triton_model_name = translate_triton_server_settings.triton_model_name

    def __init__(
        self,
        translate_settings: TranslationSettings,
        translate_tokenizer_settings: TranslationTokenizerSettings,
    ):
        super().__init__(translate_settings, translate_tokenizer_settings)

    @property
    def _translate_settings_kwargs(self) -> dict[str, any]:
        base_translate_settings = {
            "model": self.translate_settings.model,
            "src_lang": self.translate_settings.src_lang,
            "tag_lang": self.translate_settings.tag_lang,
        }
        return base_translate_settings

    @property
    def _translate_tokenizer_settings_kwargs(self) -> dict[str, any]:
        base_translate_tokenizer_settings = {
            "pretrained_model_name_or_path": self.translate_tokenizer_settings.pretrained_model_name_or_path,
            "truncation": self.translate_tokenizer_settings.truncation,
            "max_length": self.translate_tokenizer_settings.max_length,
            "padding": self.translate_tokenizer_settings.padding,
        }
        return base_translate_tokenizer_settings

    def get_tokenizer(self, forward):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._translate_tokenizer_settings_kwargs['pretrained_model_name_or_path'],
            src_lang=self._translate_settings_kwargs["src_lang"] if forward else self._translate_settings_kwargs["tag_lang"],
        )
        return tokenizer

    def get_input_ids(self, tokenizer, sentences) -> any:
        input_ids = tokenizer(
            sentences,
            return_attention_mask=False,
            return_tensors="np",
            padding=self._translate_tokenizer_settings_kwargs["padding"],
            max_length=self._translate_tokenizer_settings_kwargs["max_length"],
            truncation=self._translate_tokenizer_settings_kwargs["truncation"],
        ).input_ids.astype(np.int32)
        return input_ids

    def get_prefix_ids(self, tokenizer, sentences, forward) -> any:
        prefix_ids = np.expand_dims(
            np.array([tokenizer.lang_code_to_id[self._translate_settings_kwargs['tag_lang'] if forward else self._translate_settings_kwargs["src_lang"]]], dtype=np.int32), axis=0
        )
        prefix_ids = np.repeat(prefix_ids, len(sentences), 0)
        return prefix_ids

    def infer(self, input_ids, prefix_ids, sentences):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(sentences)):
                futures.append(
                    executor.submit(
                        TritonServerTranslation.triton_infer,
                        input_ids[i : i + 1, :],
                        prefix_ids[i : i + 1, :],
                    )
                )
            outputs = [future.result().as_numpy("OUTPUT_IDS") for future in futures]

        return outputs

    def decode(self, tokenizer, outputs):
        return [tokenizer.batch_decode(out_tokens)[0] for out_tokens in outputs]

    @classmethod
    def triton_infer(cls, input_ids, prefix_ids):
        client = tritonclient.grpc.InferenceServerClient(cls.triton_server_endpoint)
        inputs = [
            tritonclient.grpc.InferInput(
                "INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
            ),
        ]
        inputs.append(
            tritonclient.grpc.InferInput(
                "TARGET_PREFIX", prefix_ids.shape, np_to_triton_dtype(prefix_ids.dtype)
            )
        )

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(prefix_ids)

        outputs = [tritonclient.grpc.InferRequestedOutput("OUTPUT_IDS")]

        res = client.infer(
            model_name=cls.triton_model_name,
            inputs=inputs,
            outputs=outputs,
        )

        return res