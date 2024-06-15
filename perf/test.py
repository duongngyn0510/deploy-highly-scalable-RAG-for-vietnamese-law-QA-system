import requests

# Replace with the actual URL of your FastAPI endpoint
url = "http://localhost:8089/v1/chat/completions"

# Define the input data as a dictionary
data = {
    "messages": [
        {
            "content": "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.",
            "role": "system",
        },
        {
            "content": "Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu tiền ?",
            "role": "user",
        },
    ],
    "stream": False,
    "use_context": True,
}

# Set the request headers to indicate JSON content type
headers = {"Content-Type": "application/json"}

import time

st = time.time()
# Send the POST request using requests
response = requests.post(url, json=data, headers=headers)
print("time chat:", time.time() - st)
# Check for successful response status code (typically 200 or similar)
if response.status_code == 200:
    print("Request successful!")
    # Process the response data (assuming it's JSON)
    response_data = response.json()
    print(response_data)  # Example: {"message": "Here's your response..."}
else:
    print(f"Error: Request failed with status code {response.status_code}")
    print(response.text)  # Print the error message from the response
