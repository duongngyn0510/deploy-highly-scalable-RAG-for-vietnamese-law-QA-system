variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "legal-rag-427114"
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-east1-a"
}

variable "k8s" {
  description = "GKE for legal-rag"
  default     = "embedding"
}

variable "machine_type" {
  description = "VM type" 
  default     = "c2d-highcpu-4" 
}