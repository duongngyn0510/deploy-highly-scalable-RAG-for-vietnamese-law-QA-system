

### Phiên bản của các gói được sử dụng để cấu hình và triển khai

+ [kubectl](https://kubernetes.io/vi/docs/tasks/tools/) v1.28.2
+ [kubens, kubectx](https://github.com/ahmetb/kubectx) v0.9.5
+ [helm](https://helm.sh/) v3.15.3
+ [terraform](https://www.terraform.io/) v1.8.2
+ [docker-compose](https://docs.docker.com/compose) v2.21.0

## Thành phần Embedding

+ Tạo và kết nối với cụm GKE theo các bước hướng dẫn tại phần 1 `Create GKE Cluster` trong `README.md` tại [Repo](https://github.com/duongngyn0510/continuous-deployment-to-gke-cluster)
```bash
cd embedding/terraform
terraform init
terraform plan 
terraform apply 
```

+ Triển khai

    + Tạo namespace `emb`
    ```bash
    kubectl create ns emb
    kubens emb
    ```
    
    + Triển khai model lên cụm GKE (Chi tiết về các giá trị khi triển khai hãy xem trong `embedding/helm-charts/emb/values.yaml`)
    ```bash
    helm upgrade --install emb embedding/helm-charts/emb --namespace emb
    ```

    Cụm GKE Embedding sử dụng docker image với backend là [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) sẽ thực hiện pull model trên Huggingface về dựa trên model-id, nếu muốn thay đổi model hãy cập nhật giá trị ở `deployment.container.args`



## Thành phần Weaviate

+ Tạo và kết nối với cụm tương tự như Embedding
```bash
cd weaviate/terraform
terraform init
terraform plan 
terraform apply 
```
+ Triển khai

    + Tạo namespace `weaviate`
    ```bash
    kubectl create ns weaviate
    ```
    
    + Triển khai (Chi tiết về các giá trị khi triển khai hãy xem trong `weaviate/helm-charts/weaviate/values.yaml`)
    ```bash
    helm upgrade --install weaviate weaviate/helm-charts/weaviate --namespace weaviate
    ```

## Thành phần Reranker và LLM (Chạy trên máy chủ GPU)

+ Thay thể model Reranker và LLM trong file `docker-compose.yml`

+ Triển khai
```bash
cd reranker-llm
docker compose up --build
```
## Thành phần điều phối RAG
+ Tạo và kết nối với cụm tương tự như Embedding
```bash
cd rag-controller/terraform
terraform init
terraform plan 
terraform apply
```

+ Triển khai

    + Tạo namespace `rag`
    ```bash
    kubectl create ns rag
    ```

    + Triển khai nginx ingress
    ```bash
	kubectl create namespace ingress-nginx
	kubens ingress-nginx
	helm upgrade --install ingress-nginx rag-controller/helm-charts/ingress-nginx \
	--namespace ingress-nginx \
	--set controller.metrics.enabled=true \
	--set-string controller.podAnnotations."prometheus\.io/scrape"="true" \
	--set-string controller.podAnnotations."prometheus\.io/port"="10254" \
    ```
    
    + Triển khai điều phối RAG (Chi tiết về các giá trị khi triển khai hãy xem trong `rag-controller/helm-charts/rag/values.yaml`)

    Tạo secret cho biến môi trường `NVIDIA_NIM_API` trong trường hợp sử dụng [NVIDIA_NIM](https://build.nvidia.com/explore/discover) 

    ```bash
    kubectl create secret generic rag-secret --from-literal=NVIDIA_NIM_API=[YOUR_API_KEY]
    ```

    
    ```bash
    helm upgrade --install rag rag-controller/helm-charts/rag --set deployment.image.name=duong05102002/rag-controller \
	--set deployment.image.version=v0.0.21 --namespace rag
    ```

    Truy cập vào địa chỉ trong `HOSTS` là đầu ra của command mở giao diện
    ```bash
    kubectl get ing
    ```
    
