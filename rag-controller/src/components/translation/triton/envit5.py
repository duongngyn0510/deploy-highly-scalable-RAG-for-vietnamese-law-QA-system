import tritonclient.grpc
import numpy as np
import concurrent.futures
from src.settings.settings import (
    TranslationSettings,
    TranslationTokenizerSettings,
    Envit5TranslationTritonServerSettings,
)
from tritonclient.utils import np_to_triton_dtype
from src.components.translation.base import BaseTranslation


class Envit5TritonServerTranslation(BaseTranslation):
    """
    Ctranslate2 backend for Envit5 model that deployed on Triton server for inference.
    """

    def __init__(
        self,
        translation_tokenizer_settings: TranslationTokenizerSettings,
        nllb_translation_triton_server_settings: Envit5TranslationTritonServerSettings,
    ):
        super().__init__(translation_tokenizer_settings)
        self.nllb_translation_triton_server_settings = (
            nllb_translation_triton_server_settings
        )

    @property
    def _translation_tokenizer_settings_kwargs(self) -> dict[str, any]:
        base_translation_tokenizer_settings = {
            "pretrained_model_name_or_path": self.translation_tokenizer_settings.pretrained_model_name_or_path,
            "padding": self.translation_tokenizer_settings.padding,
        }
        return base_translation_tokenizer_settings

    @property
    def _envit5_translation_triton_server_settings_kwargs(self) -> dict[str, any]:
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
            mess_list = [mess for mess in mess_list if len(mess.strip().split()) > 10]
            message_list.extend(mess_list)
            len_list.append(len_list[-1] + len(mess_list))
        
        message_list = [
            self._envit5_translation_triton_server_settings_kwargs['src_lang'] + ': ' + message
            for message in message_list
        ]
        print('message_list', message_list)
        return message_list, len_list

    def postprocess_message_for_translation(
        self, translated_list: list[str], len_list: list[int], forward: bool
    ) -> str:
        res = ""
        if forward:
            pre = 0
            for id in len_list[1:]:
                for idx in range(pre, id):
                    res += translated_list[idx].split(
                        self._envit5_translation_triton_server_settings_kwargs['tag_lang'] + ':'
                    )[-1].strip() + ";"
                res = res.strip().strip(";") + "\n"
                pre = id
        else:
            res = translated_list[0].split(
                self._envit5_translation_triton_server_settings_kwargs['src_lang'] + ':'
            )[-1].strip()
        return res
    
    # def get_input_ids(self, tokenizer, sentences: list[str]) -> np.array:
    #     input_ids = tokenizer(
    #         sentences, 
    #         return_attention_mask=False, 
    #         return_tensors="np", 
    #         padding=self._translation_tokenizer_settings_kwargs['padding'],
    #         ).input_ids.astype(np.int32)
    #     return input_ids

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

    def get_prefix_ids(self):
        return 

    def infer(self, input_ids: np.array, sentences: list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(len(sentences)):
                futures.append(
                    executor.submit(
                        self.triton_infer,
                        input_ids[i],
                    )
                )
            outputs = [future.result().as_numpy("OUTPUT_IDS") for future in futures]
        return outputs

    def decode(self, tokenizer, outputs: list):
        return [tokenizer.batch_decode(out_tokens)[0] for out_tokens in outputs]

    def translation_pipeline(
        self, 
        input_message: str, 
        context_str: str, 
        forward: bool,
        tokenizer,
    ):
        # Envit5 only has one tokenizer for both English and Vietnamese
        # Translate from Vietnamese to Englist
        if forward:
            input_message_list, len_input_message_list = self.preprocess_message_for_translation(
                message=input_message
            )
            context_str_list, len_context_str_list = self.preprocess_message_for_translation(
                message=context_str
            )
            print('len(context_str_list)', len(context_str_list))
            sentences_list = input_message_list + context_str_list

            input_ids = self.get_input_ids(
                tokenizer=tokenizer, 
                sentences=sentences_list
            )
            # Envit5 model dont need PREFIX_IDS, ignore get_prefix_ids function
            outputs = self.infer(
                input_ids=input_ids,
                sentences=sentences_list
            )

            translated_list = self.decode(tokenizer, outputs)
            translated_input_message_list = translated_list[: len(input_message_list)]
            translated_context_str_list = translated_list[len(input_message_list) :]

            translated_input_message = self.postprocess_message_for_translation(
                translated_input_message_list, len_input_message_list, forward
            )
            translated_context_str = self.postprocess_message_for_translation(
                translated_context_str_list, len_context_str_list, forward
            )
            print('len(translated_context_str_list)', len(translated_context_str_list))
            return translated_input_message, translated_context_str

        # Translate from English back to vietnamese, so dont have input message
        # LLM response is a clear sentence, so dont need preprocess
        else:
            input_ids = self.get_input_ids(
                tokenizer=tokenizer, 
                sentences=[context_str]
            )
            outputs = self.infer(
                input_ids=input_ids,
                sentences=[context_str]
            )
            translated_list = self.decode(tokenizer, outputs)
            translated_context_str = self.postprocess_message_for_translation(
                translated_list, len_list=None, forward=forward
            )
            return None, translated_context_str

    def triton_infer(self, input_ids: np.array):
        client = tritonclient.grpc.InferenceServerClient(
            url=self._envit5_translation_triton_server_settings_kwargs[
                "triton_server_endpoint"
            ]
        )

        inputs = [
            tritonclient.grpc.InferInput(
                "INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input_ids)

        outputs = [tritonclient.grpc.InferRequestedOutput("OUTPUT_IDS")]

        res = client.infer(
            model_name=self._envit5_translation_triton_server_settings_kwargs[
                "triton_model_name"
            ],
            inputs=inputs,
            outputs=outputs,
        )

        return res
