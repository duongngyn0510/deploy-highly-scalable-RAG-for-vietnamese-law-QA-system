variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "orbital-age-427114-u7"
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-east1-a"
}

variable "k8s" {
  description = "GKE for legal-rag"
  default     = "rag-controller"
}

variable "machine_type" {
  description = "VM type" 
  default     = "c2d-standard-4" 
}