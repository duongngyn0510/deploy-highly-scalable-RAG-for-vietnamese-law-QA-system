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
                "text": text
            }
            self.client.post("/v1/retrieve", json=data)