#!/bin/bash

set -e

echo "=============================================="
echo "Local Kubernetes Setup for ML Platform POC"
echo "=============================================="

# Detect which K8s platform to use
detect_platform() {
    if docker info 2>/dev/null | grep -q "Kubernetes"; then
        echo "docker-desktop"
    elif command -v minikube &> /dev/null && minikube status &> /dev/null; then
        echo "minikube"
    elif command -v kind &> /dev/null && kind get clusters 2>/dev/null | grep -q "ml-platform"; then
        echo "kind"
    else
        echo "none"
    fi
}

PLATFORM=$(detect_platform)

echo ""
echo "[1/4] Checking Kubernetes platform..."
echo "======================================"

if [ "$PLATFORM" = "none" ]; then
    echo "No Kubernetes cluster detected. Please choose an option:"
    echo ""
    echo "Option 1: Docker Desktop (Recommended - Easiest)"
    echo "  - Open Docker Desktop → Settings → Kubernetes"
    echo "  - Enable Kubernetes and click Apply"
    echo ""
    echo "Option 2: Install Minikube"
    echo "  brew install minikube"
    echo "  minikube start --cpus 4 --memory 8192 --driver=docker"
    echo ""
    echo "Option 3: Install Kind"
    echo "  brew install kind"
    echo "  kind create cluster --name ml-platform --config kubernetes/kind-config.yaml"
    echo ""
    echo "Run this script again after setting up Kubernetes."
    exit 1
fi

echo "✓ Kubernetes platform detected: $PLATFORM"

# Set context based on platform
case $PLATFORM in
    "docker-desktop")
        kubectl config use-context docker-desktop
        ;;
    "minikube")
        kubectl config use-context minikube
        # Enable ingress addon
        minikube addons enable ingress
        ;;
    "kind")
        kubectl config use-context kind-ml-platform
        ;;
esac

echo ""
echo "[2/4] Verifying cluster access..."
echo "=================================="
if kubectl cluster-info &> /dev/null; then
    echo "✓ Cluster is accessible"
    kubectl cluster-info | head -n 2
else
    echo "✗ Cannot access cluster"
    exit 1
fi

echo ""
echo "[3/4] Creating namespaces..."
echo "============================"
kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace ml-models --dry-run=client -o yaml | kubectl apply -f -
echo "✓ Namespaces created: seldon-system, ml-models"

echo ""
echo "[4/4] Cluster information..."
echo "============================"
echo "Nodes:"
kubectl get nodes
echo ""
echo "Contexts:"
kubectl config get-contexts
echo ""

echo "=============================================="
echo "✓ Kubernetes cluster is ready!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Install Seldon Core: ./kubernetes/install-seldon.sh"
echo "2. Deploy model: kubectl apply -f kubernetes/fraud-detector-deployment.yaml"
echo ""
echo "Useful commands:"
echo "  kubectl get pods -A                  # View all pods"
echo "  kubectl get svc -n ml-models         # View services"
echo "  kubectl logs <pod-name> -n ml-models # View logs"
echo "=============================================="
