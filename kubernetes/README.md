# Kubernetes & Seldon Core Setup

This directory contains everything needed to deploy the fraud detection model using Seldon Core on a local Kubernetes cluster.

## 📁 Files Overview

### Setup Scripts
- **`setup-local-k8s.sh`** - Detects and configures local K8s cluster
- **`install-seldon.sh`** - Installs Seldon Core operator
- **`kind-config.yaml`** - Configuration for Kind cluster (optional)

### Deployment
- **`fraud-detector-deployment.yaml`** - Seldon deployment manifest
- **`test-seldon.py`** - Test Seldon deployment
- **`compare-fastapi-seldon.py`** - Compare FastAPI vs Seldon performance

## 🚀 Quick Start

```bash
# 1. Setup Kubernetes (choose one)
# Option A: Use Docker Desktop (Settings → Kubernetes → Enable)
# Option B: Use Minikube
minikube start --cpus 4 --memory 8192
# Option C: Use Kind
kind create cluster --name ml-platform --config kubernetes/kind-config.yaml

# 2. Verify and setup namespaces
chmod +x kubernetes/*.sh
./kubernetes/setup-local-k8s.sh

# 3. Install Seldon Core
./kubernetes/install-seldon.sh

# 4. Deploy fraud detection model
kubectl apply -f kubernetes/fraud-detector-deployment.yaml

# 5. Access the model
kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models

# 6. Test predictions
python kubernetes/test-seldon.py
```

## 📊 Compare with FastAPI

Run both serving platforms side-by-side:

```bash
# Terminal 1: FastAPI
python poc/serve_model.py

# Terminal 2: Seldon (keep running)
kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models

# Terminal 3: Run comparison
python kubernetes/compare-fastapi-seldon.py
```

## 🔍 Monitoring

```bash
# View pods
kubectl get pods -n ml-models

# View logs
kubectl logs -f <pod-name> -n ml-models

# View deployment status
kubectl get sdep fraud-detector -n ml-models

# View services
kubectl get svc -n ml-models
```

## 🧹 Cleanup

```bash
# Delete deployment
kubectl delete -f kubernetes/fraud-detector-deployment.yaml

# Uninstall Seldon
kubectl delete namespace seldon-system
kubectl delete namespace ml-models

# Delete cluster (if using Kind/Minikube)
kind delete cluster --name ml-platform
# or
minikube delete
```

## 📚 Full Documentation

See `../SELDON_SETUP.md` for detailed instructions, troubleshooting, and advanced features.

## ⚡ Quick Reference

| Command | Description |
|---------|-------------|
| `kubectl get sdep -n ml-models` | Check deployment status |
| `kubectl describe sdep fraud-detector -n ml-models` | Detailed deployment info |
| `kubectl logs <pod> -n ml-models` | View pod logs |
| `kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models` | Access locally |
| `kubectl delete sdep fraud-detector -n ml-models` | Delete deployment |
| `kubectl top pods -n ml-models` | Resource usage |

## 🎯 Expected Results

- **Setup Time:** 10-15 minutes (first time)
- **Model Latency:** 3-5ms (P99 < 8ms)
- **Memory Usage:** ~500MB per replica
- **CPU Usage:** ~200m per replica (idle)
- **Success Rate:** >99%

## 🆚 When to Use

**Use FastAPI when:**
- Quick prototyping
- Simple single-model serving
- Local development
- Learning ML serving basics

**Use Seldon Core when:**
- Testing production deployment patterns
- Need auto-scaling
- Multi-model serving
- A/B testing / canary deployments
- Learning Kubernetes + ML

## 🐛 Troubleshooting

**Pod not starting?**
```bash
kubectl describe pod <pod-name> -n ml-models
kubectl logs <pod-name> -n ml-models -c model-loader
```

**Can't connect to MLflow?**
- Ensure `host.docker.internal` resolves correctly
- For Minikube: May need to configure host access

**High latency?**
- Increase resources in deployment yaml
- Use ONNX model format
- Enable model caching

See `../SELDON_SETUP.md` for detailed troubleshooting.
