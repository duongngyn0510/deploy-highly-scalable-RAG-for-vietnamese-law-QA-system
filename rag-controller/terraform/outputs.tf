output "project_id" {
  value       = var.project_id
  description = "Project ID"
}

output "kubernetes_cluster_name" {
  value       = google_container_cluster.cluster.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = google_container_cluster.cluster.endpoint
  description = "GKE Cluster Host"
}

output "region" {
  value       = var.region
  description = "GKE region"
}

output "machine_type" {
  value       = var.machine_type
  description = "VM type"
}