#!/bin/bash

set -e

echo "=============================================="
echo "Installing Seldon Core on Local Cluster"
echo "=============================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "✗ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "✗ helm not found. Please install helm first."
    echo "  Install: brew install helm (macOS)"
    echo "  Or see: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "✗ Cannot access Kubernetes cluster"
    echo "  Run: ./kubernetes/setup-local-k8s.sh"
    exit 1
fi

SELDON_VERSION="1.17.1"

echo ""
echo "[1/6] Adding Seldon Helm repository..."
echo "======================================"

helm repo add seldonio https://storage.googleapis.com/seldon-charts
helm repo update

echo "✓ Helm repository added"

echo ""
echo "[2/6] Creating seldon-system namespace..."
echo "=========================================="

kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -

echo "✓ Namespace created"

echo ""
echo "[3/6] Installing Seldon Core Operator..."
echo "========================================="

# Install Seldon Core with Helm
helm upgrade --install seldon-core seldonio/seldon-core-operator \
    --namespace seldon-system \
    --version ${SELDON_VERSION} \
    --set usageMetrics.enabled=false \
    --set istio.enabled=false

echo "✓ Seldon Core operator installed"

echo ""
echo "[4/6] Waiting for Seldon operator to be ready..."
echo "================================================="

# Wait for operator deployment
kubectl wait --for=condition=available --timeout=300s \
    deployment/seldon-controller-manager -n seldon-system || {
    echo "⚠ Timeout waiting for operator. Checking status..."
    kubectl get pods -n seldon-system
    exit 1
}

echo "✓ Seldon operator is ready"

echo ""
echo "[5/6] Creating ml-models namespace and service account..."
echo "=========================================================="

kubectl create namespace ml-models --dry-run=client -o yaml | kubectl apply -f -
kubectl create serviceaccount seldon-model-sa -n ml-models --dry-run=client -o yaml | kubectl apply -f -

echo "✓ Namespace and service account created"

echo ""
echo "[6/6] Verifying installation..."
echo "==============================="

echo ""
echo "Seldon Core Pods:"
kubectl get pods -n seldon-system

echo ""
echo "Seldon Core Version:"
kubectl get crd seldondeployments.machinelearning.seldon.io -o jsonpath='{.spec.versions[0].name}' && echo ""

echo ""
echo "=============================================="
echo "✓ Seldon Core installed successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Deploy fraud detection model:"
echo "   kubectl apply -f kubernetes/fraud-detector-deployment.yaml"
echo ""
echo "2. Check deployment status:"
echo "   kubectl get sdep -n ml-models"
echo ""
echo "3. Access the model:"
echo "   kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models"
echo ""
echo "4. Test prediction:"
echo "   python kubernetes/test-seldon.py"
echo "=============================================="
