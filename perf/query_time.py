def calculate_mean(file_name, query):
    total = 0
    count = 0
    with open(file_name, "r") as file:
        for line in file:
            if query in line:
                try:
                    value = float(line.split(" ")[2].strip())
                    total += value
                    count += 1
                except:
                    pass
    return total / count


file_name = "logs_ccu1.txt"
vector_query = "vector_nodes time"
bm25_query = "bm25_nodes time"
print(f"vector :", calculate_mean(file_name, vector_query))
print(f"bm25 :", calculate_mean(file_name, bm25_query))
