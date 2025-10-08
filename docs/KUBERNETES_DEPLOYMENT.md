# Kubernetes Deployment Guide

This guide covers deploying the Gator AI Influencer Platform to Kubernetes clusters.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl CLI installed and configured
- Kustomize (built into kubectl 1.14+)
- 100GB+ storage available
- Optional: Helm 3.x for alternative deployment

## Quick Start

### 1. Deploy to Development

```bash
# Apply all resources for development environment
kubectl apply -k kubernetes/overlays/development/

# Verify deployment
kubectl get pods -n gator-dev
kubectl get svc -n gator-dev
```

### 2. Deploy to Staging

```bash
# Create namespace
kubectl create namespace gator-staging

# Update secrets (IMPORTANT - do this first!)
kubectl create secret generic gator-secrets \
  --from-literal=database-url='postgresql://gator:CHANGE_ME@postgres:5432/gator' \
  --from-literal=secret-key='CHANGE_ME_TO_RANDOM_STRING' \
  --from-literal=openai-api-key='sk-...' \
  -n gator-staging

# Deploy
kubectl apply -k kubernetes/overlays/staging/

# Verify
kubectl get all -n gator-staging
```

### 3. Deploy to Production

```bash
# Create namespace
kubectl create namespace gator-prod

# Update secrets (CRITICAL - use secure values!)
kubectl create secret generic gator-secrets \
  --from-literal=database-url='postgresql://gator:SECURE_PASSWORD@postgres:5432/gator' \
  --from-literal=secret-key='SECURE_RANDOM_STRING_64_CHARS_MIN' \
  --from-literal=jwt-secret='ANOTHER_SECURE_RANDOM_STRING' \
  --from-literal=openai-api-key='sk-...' \
  --from-literal=anthropic-api-key='sk-ant-...' \
  --from-literal=elevenlabs-api-key='...' \
  -n gator-prod

# Deploy
kubectl apply -k kubernetes/overlays/production/

# Verify
kubectl get all -n gator-prod
```

## Architecture Overview

The Kubernetes deployment consists of:

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│              (Ingress Controller + TLS)                  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Gator API Pods (3-20 replicas)             │
│            Auto-scaled based on CPU/Memory               │
└───────────────┬───────────────────┬─────────────────────┘
                │                   │
        ┌───────┴────────┐    ┌────┴──────────┐
        ↓                ↓    ↓               ↓
┌──────────────┐  ┌──────────┐  ┌──────────────┐
│  PostgreSQL  │  │  Redis   │  │ Shared       │
│  StatefulSet │  │  Cache   │  │ Storage      │
│              │  │          │  │ (PVC)        │
└──────────────┘  └──────────┘  └──────────────┘
```

### Components

1. **API Deployment**: FastAPI application (3-20 replicas)
2. **PostgreSQL StatefulSet**: Primary database (1 replica)
3. **Redis Deployment**: Caching and sessions (1 replica)
4. **Ingress**: SSL/TLS termination and routing
5. **HPA**: Horizontal Pod Autoscaler for API pods
6. **PVCs**: Persistent storage for content and models

## Configuration

### Environment-Specific Settings

#### Development
- **Replicas**: 1
- **Memory**: 1-2Gi per pod
- **Storage**: 10Gi content, 50Gi models
- **Log Level**: DEBUG
- **Features**: All enabled for testing

#### Staging
- **Replicas**: 2
- **Memory**: 2-4Gi per pod
- **Storage**: 50Gi content, 250Gi models
- **Log Level**: INFO
- **Features**: Production-like configuration

#### Production
- **Replicas**: 5-20 (auto-scaled)
- **Memory**: 4-8Gi per pod
- **Storage**: 100Gi content, 500Gi models
- **Log Level**: WARNING
- **Features**: Optimized for performance

### Secrets Management

**NEVER** commit actual secrets to version control. Create secrets using:

```bash
# Method 1: From literal values
kubectl create secret generic gator-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=secret-key='...' \
  -n <namespace>

# Method 2: From .env file
kubectl create secret generic gator-secrets \
  --from-env-file=.env.production \
  -n <namespace>

# Method 3: From individual files
kubectl create secret generic gator-secrets \
  --from-file=database-url=./secrets/db-url.txt \
  --from-file=secret-key=./secrets/secret-key.txt \
  -n <namespace>
```

### ConfigMap Updates

To update configuration without secrets:

```bash
# Edit the configmap
kubectl edit configmap gator-config -n <namespace>

# Or apply from file
kubectl apply -f kubernetes/base/configmap.yaml -n <namespace>

# Restart pods to pick up changes
kubectl rollout restart deployment/gator-api -n <namespace>
```

## Storage

### Persistent Volume Claims

Three PVCs are created:

1. **gator-content-pvc**: Generated content (images, videos, etc.)
2. **gator-models-pvc**: AI model files
3. **postgres-pvc**: Database storage

### Storage Classes

Default uses `standard` storage class. Modify for your cluster:

```yaml
# For AWS EBS
storageClassName: gp3

# For GCP Persistent Disk
storageClassName: pd-ssd

# For Azure Disk
storageClassName: managed-premium
```

### Backup Strategy

```bash
# Backup content PVC
kubectl exec -n <namespace> <pod-name> -- tar czf - /app/generated_content | \
  gzip > gator-content-backup-$(date +%Y%m%d).tar.gz

# Backup database
kubectl exec -n <namespace> postgres-0 -- pg_dump -U gator gator | \
  gzip > gator-db-backup-$(date +%Y%m%d).sql.gz
```

## Scaling

### Manual Scaling

```bash
# Scale API pods
kubectl scale deployment gator-api --replicas=10 -n <namespace>

# Scale down
kubectl scale deployment gator-api --replicas=3 -n <namespace>
```

### Auto-Scaling

Horizontal Pod Autoscaler (HPA) is configured by default:

```yaml
minReplicas: 3
maxReplicas: 10
metrics:
  - cpu: 70%
  - memory: 80%
```

Modify HPA:

```bash
# Edit HPA
kubectl edit hpa gator-api-hpa -n <namespace>

# View HPA status
kubectl get hpa -n <namespace>
kubectl describe hpa gator-api-hpa -n <namespace>
```

## Monitoring

### Health Checks

```bash
# Check pod health
kubectl get pods -n <namespace>

# View pod logs
kubectl logs -f deployment/gator-api -n <namespace>

# Check health endpoint
kubectl exec -it <pod-name> -n <namespace> -- curl http://localhost:8000/health
```

### Resource Usage

```bash
# View resource usage
kubectl top pods -n <namespace>
kubectl top nodes

# View detailed metrics
kubectl describe pod <pod-name> -n <namespace>
```

### Events

```bash
# View recent events
kubectl get events -n <namespace> --sort-by='.lastTimestamp'

# Watch events in real-time
kubectl get events -n <namespace> --watch
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n <namespace>

# Check logs
kubectl logs <pod-name> -n <namespace>

# Check events
kubectl get events -n <namespace>
```

Common causes:
- Insufficient resources (CPU/memory)
- Missing secrets
- Image pull errors
- Volume mount issues

#### 2. Database Connection Issues

```bash
# Check PostgreSQL pod
kubectl logs postgres-0 -n <namespace>

# Test connection from API pod
kubectl exec -it <api-pod> -n <namespace> -- \
  python -c "import asyncpg; print('OK')"

# Verify secret
kubectl get secret gator-secrets -n <namespace> -o yaml
```

#### 3. Ingress Not Working

```bash
# Check ingress status
kubectl describe ingress gator-ingress -n <namespace>

# Verify ingress controller
kubectl get pods -n ingress-nginx

# Test service directly
kubectl port-forward svc/gator-api 8000:8000 -n <namespace>
```

#### 4. Storage Issues

```bash
# Check PVC status
kubectl get pvc -n <namespace>

# Describe PVC
kubectl describe pvc gator-content-pvc -n <namespace>

# Check available storage
kubectl exec -it <pod-name> -n <namespace> -- df -h
```

### Debug Commands

```bash
# Interactive shell in pod
kubectl exec -it <pod-name> -n <namespace> -- /bin/bash

# Port forward for local testing
kubectl port-forward <pod-name> 8000:8000 -n <namespace>

# Copy files from pod
kubectl cp <namespace>/<pod-name>:/path/to/file ./local-file

# View all resources
kubectl get all -n <namespace>
```

## Updates and Rollouts

### Rolling Update

```bash
# Update image version
kubectl set image deployment/gator-api \
  gator-api=gator/api:v1.1.0 \
  -n <namespace>

# Check rollout status
kubectl rollout status deployment/gator-api -n <namespace>

# View rollout history
kubectl rollout history deployment/gator-api -n <namespace>
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/gator-api -n <namespace>

# Rollback to specific revision
kubectl rollout undo deployment/gator-api --to-revision=2 -n <namespace>
```

### Zero-Downtime Deployment

The deployment is configured for zero-downtime updates:
- Rolling update strategy
- Health checks (liveness and readiness probes)
- Grace period for pod termination
- HPA maintains minimum replicas

## Security

### Network Policies

Create network policies to restrict traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gator-api-policy
  namespace: gator-prod
spec:
  podSelector:
    matchLabels:
      app: gator
      component: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          component: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          component: cache
    ports:
    - protocol: TCP
      port: 6379
```

### RBAC

Create service account with limited permissions:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gator-api-sa
  namespace: gator-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gator-api-role
  namespace: gator-prod
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gator-api-rolebinding
  namespace: gator-prod
subjects:
- kind: ServiceAccount
  name: gator-api-sa
roleRef:
  kind: Role
  name: gator-api-role
  apiGroup: rbac.authorization.k8s.io
```

### Pod Security

Add security context to pods:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
```

## Cost Optimization

### Resource Limits

Set appropriate resource limits to avoid over-provisioning:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Cluster Autoscaler

Enable cluster autoscaler for your cloud provider:

```bash
# AWS
aws autoscaling set-desired-capacity --auto-scaling-group-name <asg-name> --desired-capacity 5

# GCP
gcloud container clusters update <cluster-name> --enable-autoscaling \
  --min-nodes 3 --max-nodes 10

# Azure
az aks update --resource-group <rg> --name <cluster-name> \
  --enable-cluster-autoscaler --min-count 3 --max-count 10
```

### Spot/Preemptible Instances

Use spot instances for non-critical workloads:

```yaml
nodeSelector:
  node.kubernetes.io/instance-type: spot
tolerations:
- key: "spot"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

## Production Checklist

Before deploying to production:

- [ ] Update all secrets with secure random values
- [ ] Configure proper ingress domain and TLS
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set resource limits and requests
- [ ] Enable HPA with appropriate thresholds
- [ ] Configure backup strategy
- [ ] Set up alerts for critical metrics
- [ ] Test disaster recovery procedures
- [ ] Document runbooks for common issues
- [ ] Set up CI/CD pipeline
- [ ] Perform load testing
- [ ] Review security policies
- [ ] Configure network policies
- [ ] Set up RBAC with least privilege

## Next Steps

1. Review [Cloud Deployment Guide](CLOUD_DEPLOYMENT.md) for provider-specific instructions
2. Set up monitoring with [Monitoring Guide](MONITORING.md)
3. Configure CI/CD with [CI/CD Guide](CICD.md)
4. Review [Security Best Practices](../SECURITY_ETHICS.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/terminills/gator/issues
- Documentation: https://github.com/terminills/gator/tree/main/docs
- Community: Discord server (link in README)
