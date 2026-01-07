# Seldon Core Local Setup Guide

This guide walks you through deploying your fraud detection model using Seldon Core on a local Kubernetes cluster, **alongside** the existing FastAPI POC.

## Overview

**Why Add Seldon Core?**
- **Production parity:** Same stack as production deployment
- **Built-in features:** Auto-scaling, A/B testing, canary deployments
- **Multi-model serving:** Serve multiple models with routing
- **Kubernetes native:** Integrates with K8s ecosystem
- **Enterprise ready:** Monitoring, explainability, outlier detection

**Architecture:**
```
┌────────────────────────────────────────────────────────────┐
│                    LOCAL ENVIRONMENT                       │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────┐              ┌──────────────┐            │
│  │   FastAPI    │              │ Seldon Core  │            │
│  │  (Port 8000) │              │ (Port 8001)  │            │
│  │              │              │              │            │
│  │  • Simple    │              │  • K8s-based │            │
│  │  • Direct    │              │  • Scalable  │            │
│  │  • Fast dev  │              │  • Prod-like │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                             │                    │
│         └─────────────┬───────────────┘                    │
│                       ▼                                    │
│              ┌─────────────────┐                           │
│              │  MLflow Registry│                           │
│              │  Cassandra      │                           │
│              └─────────────────┘                           │
└────────────────────────────────────────────────────────────┘
```

---

## Quick Start (15 minutes)

### Step 1: Set up Kubernetes Cluster

**Option A: Docker Desktop (Recommended)**
1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Enable Kubernetes
4. Click Apply & Restart
5. Wait 2-3 minutes for cluster to start

**Option B: Minikube**
```bash
brew install minikube
minikube start --cpus 4 --memory 8192 --driver=docker
```

**Option C: Kind**
```bash
brew install kind
kind create cluster --name ml-platform --config kubernetes/kind-config.yaml
```

### Step 2: Verify Cluster

```bash
# Make scripts executable
chmod +x kubernetes/*.sh

# Run setup script
./kubernetes/setup-local-k8s.sh
```

**Expected output:**
```
✓ Kubernetes platform detected: docker-desktop
✓ Cluster is accessible
✓ Namespaces created: seldon-system, ml-models
✓ Kubernetes cluster is ready!
```

### Step 3: Install Seldon Core

```bash
./kubernetes/install-seldon.sh
```

**Expected output:**
```
✓ Seldon Core operator YAML applied
✓ Seldon operator is ready
✓ Seldon Core installed successfully!
```

**Verify installation:**
```bash
kubectl get pods -n seldon-system

# Should show:
# NAME                                          READY   STATUS    RESTARTS
# seldon-controller-manager-xxx                 1/1     Running   0
```

### Step 4: Deploy Fraud Detection Model

```bash
# Ensure MLflow and model are running
docker-compose up -d mlflow
python poc/train_fraud_model.py  # If not already trained

# Deploy to Seldon
kubectl apply -f kubernetes/fraud-detector-deployment.yaml
```

**Check deployment status:**
```bash
# Watch deployment (Ctrl+C to exit)
kubectl get sdep fraud-detector -n ml-models -w

# Should eventually show:
# NAME             STATE     READY   AGE
# fraud-detector   Available   1/1     2m
```

### Step 5: Access the Model

**Port forward to local machine:**
```bash
kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models
```

Keep this terminal open. The model is now accessible at `http://localhost:8001`

### Step 6: Test Predictions

In a new terminal:

```bash
# Make test script executable
chmod +x kubernetes/test-seldon.py

# Run tests
python kubernetes/test-seldon.py
```

---

## Detailed Setup

### Understanding the Deployment

**File:** `kubernetes/fraud-detector-deployment.yaml`

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: fraud-detector
  namespace: ml-models
spec:
  predictors:
  - name: default
    replicas: 1  # Can scale to multiple replicas
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: file:///mnt/models
```

**Key components:**

1. **Init Container:** Downloads model from MLflow on startup
2. **Model Server:** Serves predictions using sklearn server
3. **Service:** Exposes model via Kubernetes service
4. **Auto-scaling:** Can add HPA (Horizontal Pod Autoscaler)

### Accessing the Model

**Method 1: Port Forwarding (Development)**
```bash
kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models
```
Access at: `http://localhost:8001`

**Method 2: NodePort (Local testing)**
- Docker Desktop / Kind: `http://localhost:30000`
- Minikube: `http://$(minikube ip):30000`

**Method 3: Ingress (Advanced)**
```bash
# Install NGINX ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Apply ingress rules (create separately)
kubectl apply -f kubernetes/fraud-detector-ingress.yaml
```

### API Format Differences

**FastAPI Request:**
```json
{
  "transaction_id": "123",
  "amount": 500.0,
  "transaction_hour": 23,
  "merchant_category": "Electronics"
}
```

**Seldon Request:**
```json
{
  "data": {
    "ndarray": [[500.0, 23, 0, 1, 1, 25.0, 8, 22]]
  }
}
```

**Seldon Response:**
```json
{
  "data": {
    "ndarray": [[0.13, 0.87]]  // [legit_prob, fraud_prob]
  },
  "meta": {}
}
```

---

## Comparison: FastAPI vs Seldon Core

### Run Side-by-Side Comparison

```bash
# Terminal 1: Run FastAPI
python poc/serve_model.py

# Terminal 2: Run Seldon port-forward
kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models

# Terminal 3: Run comparison
python kubernetes/compare-fastapi-seldon.py
```

**Expected Results:**

| Metric | FastAPI | Seldon Core |
|--------|---------|-------------|
| **Setup Time** | 30 seconds | 5-10 minutes |
| **Average Latency** | 2-3ms | 3-5ms |
| **P99 Latency** | 4-5ms | 6-8ms |
| **Memory Usage** | ~200MB | ~500MB |
| **Auto-scaling** | Manual | Built-in |
| **A/B Testing** | Custom code | Native |
| **Production Ready** | No | Yes |

---

## Advanced Features

### 1. A/B Testing (Traffic Splitting)

Update `fraud-detector-deployment.yaml`:

```yaml
spec:
  predictors:
  - name: model-v1
    replicas: 2
    traffic: 80  # 80% traffic
    graph:
      modelUri: models:/fraud_detector/1
      
  - name: model-v2
    replicas: 1
    traffic: 20  # 20% traffic
    graph:
      modelUri: models:/fraud_detector/2
```

### 2. Canary Deployment

```yaml
spec:
  predictors:
  - name: stable
    replicas: 3
    traffic: 95
    
  - name: canary
    replicas: 1
    traffic: 5  # Test new version with 5% traffic
```

### 3. Auto-scaling

```bash
# Add Horizontal Pod Autoscaler
kubectl autoscale deployment fraud-detector-default-0-classifier \
  --cpu-percent=70 \
  --min=1 \
  --max=10 \
  -n ml-models
```

### 4. Custom Metrics

Add to deployment:
```yaml
componentSpecs:
- spec:
    containers:
    - name: classifier
      env:
      - name: SELDON_ENABLE_METRICS
        value: "true"
```

Metrics available at: `/prometheus`

---

## Monitoring

### View Logs

```bash
# Get pod name
kubectl get pods -n ml-models

# View logs
kubectl logs <pod-name> -n ml-models

# Stream logs
kubectl logs -f <pod-name> -n ml-models

# View init container logs (model download)
kubectl logs <pod-name> -n ml-models -c model-loader
```

### Check Resource Usage

```bash
# CPU and memory
kubectl top pods -n ml-models

# Detailed pod info
kubectl describe pod <pod-name> -n ml-models
```

### Prometheus Metrics

```bash
# Port forward Prometheus (if installed)
kubectl port-forward svc/seldon-core-analytics-prometheus 9090:80 -n seldon-system

# Access at: http://localhost:9090
```

**Key metrics:**
- `seldon_api_executor_server_requests_seconds`
- `seldon_api_executor_server_requests_total`

---

## Troubleshooting

### Issue: Pod not starting

```bash
# Check pod status
kubectl get pods -n ml-models

# View events
kubectl describe pod <pod-name> -n ml-models

# Common causes:
# 1. Model not in MLflow registry
# 2. MLflow not accessible (use host.docker.internal)
# 3. Insufficient resources
```

**Solution:**
```bash
# Check MLflow is accessible from K8s
kubectl run test --rm -it --image=curlimages/curl -- \
  curl http://host.docker.internal:5000/health
```

### Issue: Model loading fails

```bash
# Check init container logs
kubectl logs <pod-name> -n ml-models -c model-loader

# Common causes:
# 1. Model not registered in MLflow
# 2. Model version doesn't exist
# 3. Network issues
```

**Solution:**
```bash
# Verify model exists
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=fraud_detector
```

### Issue: High latency

**Causes:**
1. Model loading on first request
2. Resource constraints
3. Network overhead

**Solutions:**
```bash
# Increase resources
kubectl edit sdep fraud-detector -n ml-models

# Change:
resources:
  requests:
    cpu: "1"
    memory: "1Gi"
```

### Issue: Can't connect to Cassandra

The deployment uses `host.docker.internal` to access Cassandra running in Docker Compose.

**For Minikube:**
```bash
# Get host IP
minikube ssh "grep host.docker.internal /etc/hosts || echo '192.168.65.2 host.docker.internal' | sudo tee -a /etc/hosts"
```

**For Kind:**
```bash
# Kind requires extra network config
# Use service-based connection instead
```

---

## Production Migration Path

### Phase 1: POC (Current)
- Local K8s (Docker Desktop / Minikube)
- Single replica
- Manual deployment
- Port forwarding for access

### Phase 2: Staging
- Cloud K8s (EKS / GKE / AKS)
- 3-5 replicas
- CI/CD with GitLab/GitHub Actions
- Load balancer + ingress

### Phase 3: Production
- Multi-region K8s clusters
- 100+ replicas with auto-scaling
- A/B testing, canary deployments
- Full observability stack

**Migration steps:**
1. Export deployment: `kubectl get sdep fraud-detector -n ml-models -o yaml > prod-deployment.yaml`
2. Update container registry (use ECR/GCR instead of local images)
3. Add production resource limits
4. Configure ingress/load balancer
5. Set up monitoring and alerting

---

## Cleanup

### Remove Seldon Deployment

```bash
kubectl delete -f kubernetes/fraud-detector-deployment.yaml
```

### Uninstall Seldon Core

```bash
kubectl delete namespace seldon-system
kubectl delete namespace ml-models
```

### Delete K8s Cluster

**Docker Desktop:**
Settings → Kubernetes → Disable Kubernetes

**Minikube:**
```bash
minikube delete
```

**Kind:**
```bash
kind delete cluster --name ml-platform
```

---

## Next Steps

1. ✅ **Test both FastAPI and Seldon** side-by-side
2. ✅ **Compare performance** using comparison script
3. ✅ **Experiment with scaling** (increase replicas)
4. 📝 **Document findings** for production planning
5. 🚀 **Plan cloud deployment** using learnings

---

## Resources

- **Seldon Core Docs:** https://docs.seldon.io/
- **Kubernetes Docs:** https://kubernetes.io/docs/
- **MLflow Model Serving:** https://mlflow.org/docs/latest/models.html
- **Comparison Script:** `kubernetes/compare-fastapi-seldon.py`

---

## Summary

**What You've Built:**
- ✅ Local Kubernetes cluster
- ✅ Seldon Core operator installed
- ✅ Fraud detection model deployed via Seldon
- ✅ Side-by-side comparison with FastAPI
- ✅ Production-ready deployment pattern validated

**Key Learnings:**
- Seldon adds ~1-3ms latency vs FastAPI (still <5ms)
- Kubernetes overhead is ~500MB RAM
- Production features (scaling, A/B testing) are built-in
- Same patterns scale from local to production

**Ready for Production:** The deployment patterns you're using locally translate directly to production - just different scale!
