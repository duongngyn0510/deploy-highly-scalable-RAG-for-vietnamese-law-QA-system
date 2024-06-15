from llama_index.llms.vllm import VllmServer
llm = VllmServer(api_url='http://103.145.79.20:6017/generate', max_new_tokens=20)

llm.stream_complete("việt nam là quốc gia ở đâu ?")