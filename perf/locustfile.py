from locust import HttpUser, task, between
from loguru import logger
import os


class ModelUser(HttpUser):
    wait_time = between(0, 1)

    @task
    def predict(self):
        logger.info("Sending POST requests!")
        with open("questions.txt", "r") as file:
            questions = file.readlines()

        for text in questions:
            data = {
                "messages": [
                    {
                        "content": "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.",
                        "role": "system",
                    },
                    {
                        "content": text,
                        "role": "user",
                    },
                ],
                "stream": False,
                "use_context": True,
            }
            self.client.post("/v1/chat/completions", json=data)
            # data = {
            #     'model': 'mistralai/Mistral-7B-Instruct-v0.2',
            #     'temperature': 1.0,
            #     'n': 1,
            #     'best_of': None,
            #     'presence_penalty': 0.0,
            #     'frequency_penalty': 0.0,
            #     'repetition_penalty': 1.0,
            #     'top_p': 1.0,
            #     'top_k': -1,
            #     'min_p': 0,
            #     'use_beam_search': False,
            #     'length_penalty': 1.0,
            #     'early_stopping': False,
            #     'stop': None,
            #     'ignore_eos': False,
            #     'max_tokens': 1024,
            #     'logprobs': None,
            #     'prompt_logprobs': None,
            #     'skip_special_tokens': True,
            #     'spaces_between_special_tokens': True,
            #     'prompt': text
            # }

            # self.client.post("/v1/completions", json=data)