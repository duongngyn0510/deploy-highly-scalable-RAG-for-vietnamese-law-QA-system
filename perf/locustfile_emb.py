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
                "inputs": text
            }
            self.client.post("/embed", json=data)