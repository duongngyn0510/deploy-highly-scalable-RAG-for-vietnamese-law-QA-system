import tritonclient.grpc
import numpy as np
import concurrent.futures
from src.settings.settings import (
    TranslationSettings,
    TranslationTokenizerSettings,
    NllbTranslationTritonServerSettings,
)
from tritonclient.utils import np_to_triton_dtype
from src.components.translation.base import BaseTranslation


class NllbTritonServerTranslation(BaseTranslation):
    """
    Ctranslate2 backend for Nllb model that deployed on Triton server for inference.
    """

    def __init__(
        self,
        translation_tokenizer_settings: TranslationTokenizerSettings,
        nllb_translation_triton_server_settings: NllbTranslationTritonServerSettings,
    ):
        super().__init__(translation_tokenizer_settings)
        self.nllb_translation_triton_server_settings = (
            nllb_translation_triton_server_settings
        )

    @property
    def _translation_tokenizer_settings_kwargs(self) -> dict[str, any]:
        base_translation_tokenizer_settings = {
            "pretrained_model_name_or_path": self.translation_tokenizer_settings.pretrained_model_name_or_path,
            "truncation": self.translation_tokenizer_settings.truncation,
            "max_length": self.translation_tokenizer_settings.max_length,
            "padding": self.translation_tokenizer_settings.padding,
        }
        return base_translation_tokenizer_settings

    @property
    def _nllb_translation_triton_server_settings_kwargs(self) -> dict[str, any]:
        base_nllb_translation_triton_server_settings = {
            "triton_server_endpoint": self.nllb_translation_triton_server_settings.triton_server_endpoint,
            "triton_model_name": self.nllb_translation_triton_server_settings.triton_model_name,
            "src_lang": self.nllb_translation_triton_server_settings.src_lang,
            "tag_lang": self.nllb_translation_triton_server_settings.tag_lang,
        }
        return base_nllb_translation_triton_server_settings
    
    def preprocess_message_for_translation(self, message: str) -> tuple[list[str], list[int]]:
        message_list = []
        len_list = []
        len_list.append(0)
        for mess in message.split("\n"):
            mess = mess.strip()
            mess_list = mess.split(";")
            mess_list = [mess for mess in mess_list if len(mess.strip().split()) > 3]
            message_list.extend(mess_list)
            len_list.append(len_list[-1] + len(mess_list))
        return message_list, len_list

    def postprocess_message_for_translation(
        self, translated_list: list[str], len_list: list[int], forward: bool
    ) -> str:
        tag_lang = (
            self._nllb_translation_triton_server_settings_kwargs['tag_lang']
            if forward
            else self._nllb_translation_triton_server_settings_kwargs['src_lang']
        )
        res = ""
        pre = 0
        for id in len_list[1:]:
            for idx in range(pre, id):
                res += translated_list[idx].split(tag_lang)[-1].strip() + ";"
            res = res.strip().strip(";") + "\n"
            pre = id
        return res
    
    def get_input_ids(self, tokenizer, sentences: list[str]) -> list[np.array]:
        input_ids_list = []
        for sentence in sentences:
            input_ids = tokenizer(
                sentence,
                return_attention_mask=False,
                return_tensors="np",
            ).input_ids.astype(np.int32)
            input_ids_list.append(input_ids)
        return input_ids_list

    def get_prefix_ids(self, tokenizer, sentences: list[str], forward: bool) -> np.array:
        prefix_ids = np.expand_dims(
            np.array(
                [
                    tokenizer.lang_code_to_id[
                        self._nllb_translation_triton_server_settings_kwargs["tag_lang"]
                        if forward
                        else self._nllb_translation_triton_server_settings_kwargs["src_lang"]
                    ]
                ],
                dtype=np.int32,
            ),
            axis=0,
        )
        prefix_ids = np.repeat(prefix_ids, len(sentences), 0)
        return prefix_ids

    def infer(self, input_ids_list: list[np.array], prefix_ids: np.array, sentences):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(sentences)):
                futures.append(
                    executor.submit(
                        self.triton_infer,
                        input_ids_list[i],
                        prefix_ids[i : i + 1, :],
                    )
                )
            outputs = [future.result().as_numpy("OUTPUT_IDS") for future in futures]
        return outputs

    def decode(self, tokenizer, outputs: list) -> list[str]:
        return [tokenizer.batch_decode(out_tokens)[0] for out_tokens in outputs]

    def triton_infer(self, input_ids: np.array, prefix_ids: np.array):
        client = tritonclient.grpc.InferenceServerClient(
            url=self._nllb_translation_triton_server_settings_kwargs[
                "triton_server_endpoint"
            ]
        )

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
            model_name=self._nllb_translation_triton_server_settings_kwargs[
                "triton_model_name"
            ],
            inputs=inputs,
            outputs=outputs,
        )

        return res
