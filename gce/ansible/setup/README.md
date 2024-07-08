### Create GCE
```bash
export $(grep -v '^#' .env | xargs)
ansible-playbook create_compute_instance.yaml -i ../inventory
```