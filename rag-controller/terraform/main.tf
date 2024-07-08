terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.80.0" // Provider version
    }
  }
  required_version = "1.8.2" // Terraform version
}

provider "google" {
  credentials = "./secret/legal-rag-3cd976d963d1.json"
  project     = var.project_id
  region      = var.region
}

// Google Kubernetes Engine
resource "google_container_cluster" "cluster" {
  name     = "${var.k8s}-gke"
  location = var.region
  remove_default_node_pool = true
  initial_node_count       = 1

  node_config {
    disk_size_gb = 10
  }
}

resource "google_container_node_pool" "node_pool" {
  name       = "my-node-pool"
  location   = google_container_cluster.cluster.location
  cluster    = google_container_cluster.cluster.name
  initial_node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 2
  }

  node_config {
    disk_size_gb = 30
    machine_type = var.machine_type
  }
}
