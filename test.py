from prometheus_api_client import PrometheusConnect

def get_prometheus_metrics(prometheus_url, query):
    # Connect to Prometheus
    prom = PrometheusConnect(url="http://prometheus-server.monitoring.svc.cluster.local:80", disable_ssl=True)
    
    # Execute the query
    result = prom.custom_query(query='sum(increase(nginx_ingress_controller_requests{}[2h]))')
    
    # Return the result
    return result

# Example usage
prometheus_url = 'http://35.185.131.27'  # Replace with your Prometheus server URL
query = 'sum(increase(nginx_ingress_controller_requests{}[2h]))'  # Replace with your PromQL query
metrics = get_prometheus_metrics(prometheus_url, query)
print(float(metrics[0]['value'][1]))
# Print the metrics
# for metric in metrics:
#     print(f"Metric: {metric['metric']}, Value: {metric['value']}")
