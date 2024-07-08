i=12

# docker build -t legal-rag:v1.$i .
# docker tag legal-rag:v1.$i asia.gcr.io/legal-rag-427109/legal-rag:v1.$i
# docker push asia.gcr.io/legal-rag-427109/legal-rag:v1.$i
kubens rag
helm upgrade --install rag helm-charts/rag --set deployment.image.name=asia.gcr.io/orbital-age-427114-u7/legal-rag \
--set deployment.image.version=v1.$i