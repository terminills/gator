# Cloud Deployment Guide

Deploy Gator AI Influencer Platform to major cloud providers with optimized configurations.

## Overview

This guide covers deployment to:
- Amazon Web Services (AWS)
- Google Cloud Platform (GCP)
- Microsoft Azure

Each section includes one-click deployment templates and manual setup instructions.

---

## AWS Deployment

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                 Route 53 (DNS)                        │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│         CloudFront CDN + WAF                          │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│     Application Load Balancer (ALB)                   │
└──────┬─────────────────────────────────┬─────────────┘
       │                                 │
┌──────┴──────────────┐        ┌────────┴──────────────┐
│   ECS Fargate       │        │   EKS Cluster         │
│   (Serverless)      │   OR   │   (Kubernetes)        │
│   3-20 tasks        │        │   3-20 pods           │
└──────┬──────────────┘        └────────┬──────────────┘
       │                                │
┌──────┴─────────────────┬──────────────┴──────────────┐
│                        │                              │
│  RDS PostgreSQL   ┌────┴─────┐   ┌────────────────┐  │
│  Multi-AZ         │ElastiCache│   │ S3 + EFS      │  │
│                   │  Redis    │   │ Storage       │  │
└───────────────────┴──────────┴───┴────────────────┘  │
                                                        │
└───────────────────────────────────────────────────────┘
```

### Option 1: ECS Fargate (Recommended for Quick Start)

#### One-Click Deploy with CloudFormation

```bash
# Download template
curl -o gator-ecs.yaml https://raw.githubusercontent.com/terminills/gator/main/cloud/aws/ecs-fargate.yaml

# Deploy stack
aws cloudformation create-stack \
  --stack-name gator-production \
  --template-body file://gator-ecs.yaml \
  --parameters \
    ParameterKey=DomainName,ParameterValue=api.yourdomain.com \
    ParameterKey=DatabasePassword,ParameterValue=SecurePassword123! \
    ParameterKey=OpenAIApiKey,ParameterValue=sk-... \
  --capabilities CAPABILITY_IAM

# Monitor deployment
aws cloudformation describe-stacks \
  --stack-name gator-production \
  --query 'Stacks[0].StackStatus'
```

#### Manual ECS Setup

1. **Create VPC and Networking**
```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=gator-vpc}]'

# Create subnets (2 public, 2 private across AZs)
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.1.0/24 --availability-zone us-east-1a
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.2.0/24 --availability-zone us-east-1b
```

2. **Create RDS Database**
```bash
aws rds create-db-instance \
  --db-instance-identifier gator-db \
  --db-instance-class db.t3.large \
  --engine postgres \
  --engine-version 15.3 \
  --master-username gator \
  --master-user-password SecurePassword123! \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --vpc-security-group-ids <sg-id> \
  --db-subnet-group-name <subnet-group>
```

3. **Create ECS Cluster**
```bash
aws ecs create-cluster --cluster-name gator-cluster
```

4. **Create Task Definition**
```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

5. **Create Service**
```bash
aws ecs create-service \
  --cluster gator-cluster \
  --service-name gator-api \
  --task-definition gator-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=gator-api,containerPort=8000"
```

### Option 2: EKS (Kubernetes)

#### One-Click Deploy with eksctl

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name gator-cluster \
  --version 1.28 \
  --region us-east-1 \
  --nodegroup-name gator-nodes \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Deploy application
kubectl apply -k kubernetes/overlays/production/
```

### Storage Configuration

#### S3 for Generated Content
```bash
# Create S3 bucket
aws s3 mb s3://gator-content-production

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket gator-content-production \
  --versioning-configuration Status=Enabled

# Configure lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket gator-content-production \
  --lifecycle-configuration file://s3-lifecycle.json
```

#### EFS for Model Storage
```bash
# Create EFS filesystem
aws efs create-file-system \
  --creation-token gator-models \
  --performance-mode maxIO \
  --throughput-mode provisioned \
  --provisioned-throughput-in-mibps 100 \
  --encrypted
```

### Cost Estimates (Monthly)

**Small Deployment** (10k requests/day):
- ECS Fargate (3 tasks): $150
- RDS db.t3.large: $120
- ElastiCache t3.micro: $15
- S3 Storage (100GB): $3
- Data Transfer: $50
- **Total: ~$340/month**

**Medium Deployment** (100k requests/day):
- ECS Fargate (8 tasks): $400
- RDS db.r6g.xlarge: $350
- ElastiCache r6g.large: $150
- S3 Storage (1TB): $24
- CloudFront: $100
- Data Transfer: $200
- **Total: ~$1,224/month**

**Large Deployment** (1M requests/day):
- EKS Cluster + EC2: $1,200
- RDS db.r6g.4xlarge Multi-AZ: $1,600
- ElastiCache cluster: $400
- S3 Storage (10TB): $240
- CloudFront: $500
- Data Transfer: $800
- **Total: ~$4,740/month**

---

## Google Cloud Platform Deployment

### Architecture

```
┌──────────────────────────────────────────────────────┐
│            Cloud DNS + Cloud CDN                      │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│      Cloud Load Balancing + Cloud Armor              │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│         GKE Cluster (Kubernetes)                      │
│         3-20 pods with auto-scaling                   │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴─────────────────┬─────────────────────────────┐
│                        │                             │
│  Cloud SQL         ┌───┴──────┐  ┌────────────────┐ │
│  PostgreSQL        │ Memorystore│ │ Cloud Storage  │ │
│  HA Config         │  Redis    │  │ + Filestore    │ │
└────────────────────┴───────────┴──┴────────────────┘ │
                                                        │
└───────────────────────────────────────────────────────┘
```

### Deploy with GKE

#### One-Click Deploy

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable redis.googleapis.com

# Create cluster
gcloud container clusters create gator-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials gator-cluster --zone us-central1-a

# Deploy application
kubectl apply -k kubernetes/overlays/production/
```

#### Create Cloud SQL Instance

```bash
gcloud sql instances create gator-db \
  --database-version=POSTGRES_15 \
  --tier=db-custom-4-16384 \
  --region=us-central1 \
  --availability-type=REGIONAL \
  --storage-size=100GB \
  --storage-type=SSD \
  --storage-auto-increase
```

#### Create Memorystore (Redis)

```bash
gcloud redis instances create gator-cache \
  --size=5 \
  --region=us-central1 \
  --redis-version=redis_6_x \
  --tier=standard
```

### Storage Configuration

#### Cloud Storage for Content
```bash
# Create bucket
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l US gs://gator-content-production/

# Enable versioning
gsutil versioning set on gs://gator-content-production/

# Set lifecycle policy
gsutil lifecycle set storage-lifecycle.json gs://gator-content-production/
```

#### Filestore for Models
```bash
gcloud filestore instances create gator-models \
  --zone=us-central1-a \
  --tier=STANDARD \
  --file-share=name="models",capacity=1TB \
  --network=name="default"
```

### Cost Estimates (Monthly)

**Small Deployment**:
- GKE Cluster: $150
- Cloud SQL (db-custom-2-8): $180
- Memorystore (5GB): $35
- Cloud Storage (100GB): $3
- **Total: ~$368/month**

**Medium Deployment**:
- GKE Cluster: $500
- Cloud SQL (db-custom-8-32): $650
- Memorystore (20GB): $140
- Cloud Storage (1TB): $24
- Cloud CDN: $80
- **Total: ~$1,394/month**

**Large Deployment**:
- GKE Cluster: $1,500
- Cloud SQL HA (db-custom-32-128): $2,600
- Memorystore cluster: $500
- Cloud Storage (10TB): $240
- Cloud CDN: $400
- **Total: ~$5,240/month**

---

## Microsoft Azure Deployment

### Architecture

```
┌──────────────────────────────────────────────────────┐
│      Azure DNS + Azure Front Door                     │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│   Application Gateway + Web Application Firewall      │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────┐
│       AKS Cluster (Kubernetes)                        │
│       3-20 pods with VMSS auto-scaling                │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴─────────────────┬─────────────────────────────┐
│                        │                             │
│  Azure Database    ┌───┴──────┐  ┌────────────────┐ │
│  for PostgreSQL    │Azure Cache│  │ Blob Storage + │ │
│  Flexible Server   │for Redis  │  │ Azure Files    │ │
└────────────────────┴───────────┴──┴────────────────┘ │
                                                        │
└───────────────────────────────────────────────────────┘
```

### Deploy with AKS

#### One-Click Deploy with ARM Template

```bash
# Download template
curl -o gator-azure.json https://raw.githubusercontent.com/terminills/gator/main/cloud/azure/aks-deployment.json

# Create resource group
az group create --name gator-production --location eastus

# Deploy template
az deployment group create \
  --resource-group gator-production \
  --template-file gator-azure.json \
  --parameters \
    clusterName=gator-cluster \
    databasePassword=SecurePassword123! \
    openaiApiKey=sk-...
```

#### Manual AKS Setup

1. **Create AKS Cluster**
```bash
az aks create \
  --resource-group gator-production \
  --name gator-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

2. **Get Credentials**
```bash
az aks get-credentials \
  --resource-group gator-production \
  --name gator-cluster
```

3. **Create PostgreSQL Database**
```bash
az postgres flexible-server create \
  --resource-group gator-production \
  --name gator-db \
  --location eastus \
  --admin-user gator \
  --admin-password SecurePassword123! \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --storage-size 128 \
  --version 15
```

4. **Create Azure Cache for Redis**
```bash
az redis create \
  --resource-group gator-production \
  --name gator-cache \
  --location eastus \
  --sku Standard \
  --vm-size c3
```

5. **Deploy Application**
```bash
kubectl apply -k kubernetes/overlays/production/
```

### Storage Configuration

#### Blob Storage for Content
```bash
# Create storage account
az storage account create \
  --name gatorstorage \
  --resource-group gator-production \
  --location eastus \
  --sku Standard_LRS

# Create container
az storage container create \
  --name content \
  --account-name gatorstorage
```

#### Azure Files for Models
```bash
az storage share create \
  --name models \
  --account-name gatorstorage \
  --quota 1024
```

### Cost Estimates (Monthly)

**Small Deployment**:
- AKS Cluster: $140
- PostgreSQL Flexible (D2s_v3): $200
- Azure Cache (C1): $80
- Blob Storage (100GB): $3
- **Total: ~$423/month**

**Medium Deployment**:
- AKS Cluster: $500
- PostgreSQL Flexible (D8s_v3): $700
- Azure Cache (C3): $250
- Blob Storage (1TB): $24
- Front Door: $100
- **Total: ~$1,574/month**

**Large Deployment**:
- AKS Cluster: $1,400
- PostgreSQL HA (D32s_v3): $2,800
- Azure Cache Premium: $650
- Blob Storage (10TB): $240
- Front Door + WAF: $400
- **Total: ~$5,490/month**

---

## Multi-Cloud Deployment

### Using Terraform

Deploy to multiple clouds with a single configuration:

```hcl
# main.tf
module "gator_aws" {
  source = "./modules/aws"
  enabled = var.deploy_to_aws
}

module "gator_gcp" {
  source = "./modules/gcp"
  enabled = var.deploy_to_gcp
}

module "gator_azure" {
  source = "./modules/azure"
  enabled = var.deploy_to_azure
}
```

```bash
# Initialize
terraform init

# Plan deployment
terraform plan

# Apply configuration
terraform apply
```

---

## Performance Optimization

### CDN Configuration

#### AWS CloudFront
```bash
aws cloudfront create-distribution \
  --origin-domain-name gator-api.example.com \
  --default-root-object index.html
```

#### GCP Cloud CDN
```bash
gcloud compute backend-services update gator-backend \
  --enable-cdn \
  --cache-mode CACHE_ALL_STATIC
```

#### Azure Front Door
```bash
az afd profile create \
  --profile-name gator-cdn \
  --resource-group gator-production \
  --sku Premium_AzureFrontDoor
```

### Database Optimization

#### Connection Pooling
```python
# config/database.py
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 40
DATABASE_POOL_TIMEOUT = 30
DATABASE_POOL_RECYCLE = 3600
```

#### Read Replicas
- AWS: Create Read Replica in RDS
- GCP: Enable read replicas in Cloud SQL
- Azure: Configure read scale-out

### Caching Strategy

```python
# Redis caching configuration
CACHE_TTL_SHORT = 300      # 5 minutes
CACHE_TTL_MEDIUM = 3600    # 1 hour
CACHE_TTL_LONG = 86400     # 24 hours

# Cache keys
CACHE_KEY_PERSONA = "persona:{id}"
CACHE_KEY_CONTENT = "content:{id}"
CACHE_KEY_ANALYTICS = "analytics:{persona_id}:{date}"
```

---

## Security Best Practices

### SSL/TLS Configuration

#### AWS Certificate Manager
```bash
aws acm request-certificate \
  --domain-name api.yourdomain.com \
  --validation-method DNS
```

#### GCP Managed Certificates
```bash
gcloud compute ssl-certificates create gator-cert \
  --domains api.yourdomain.com
```

#### Azure Key Vault
```bash
az keyvault certificate create \
  --vault-name gator-keyvault \
  --name gator-cert \
  --policy "$(az keyvault certificate get-default-policy)"
```

### Secrets Management

#### AWS Secrets Manager
```bash
aws secretsmanager create-secret \
  --name gator/api-keys \
  --secret-string file://secrets.json
```

#### GCP Secret Manager
```bash
gcloud secrets create gator-api-keys \
  --data-file=secrets.json
```

#### Azure Key Vault
```bash
az keyvault secret set \
  --vault-name gator-keyvault \
  --name api-keys \
  --file secrets.json
```

---

## Monitoring and Alerting

### AWS CloudWatch

```bash
# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name GatorMetrics \
  --dashboard-body file://cloudwatch-dashboard.json

# Create alarms
aws cloudwatch put-metric-alarm \
  --alarm-name high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --threshold 80
```

### GCP Cloud Monitoring

```bash
# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High CPU Alert" \
  --condition-display-name="CPU above 80%"
```

### Azure Monitor

```bash
# Create metric alert
az monitor metrics alert create \
  --name high-cpu \
  --resource-group gator-production \
  --scopes /subscriptions/.../resourceGroups/gator-production \
  --condition "avg Percentage CPU > 80"
```

---

## Backup and Disaster Recovery

### Automated Backups

#### AWS
```bash
# Enable RDS automated backups
aws rds modify-db-instance \
  --db-instance-identifier gator-db \
  --backup-retention-period 30 \
  --preferred-backup-window "03:00-04:00"
```

#### GCP
```bash
# Configure Cloud SQL backups
gcloud sql instances patch gator-db \
  --backup-start-time 03:00 \
  --retained-backups-count 30
```

#### Azure
```bash
# Configure PostgreSQL backups
az postgres flexible-server update \
  --resource-group gator-production \
  --name gator-db \
  --backup-retention 30
```

### Disaster Recovery Testing

```bash
# Simulate failure and test recovery
# 1. Take snapshot
# 2. Delete resources
# 3. Restore from snapshot
# 4. Verify data integrity
# 5. Measure Recovery Time Objective (RTO)
```

---

## Migration Between Clouds

### Database Migration

```bash
# Export from source
pg_dump -h source-db.rds.amazonaws.com -U gator gator > gator-backup.sql

# Import to destination
psql -h destination-db.cloudsql.com -U gator gator < gator-backup.sql
```

### Storage Migration

```bash
# AWS S3 to GCP Cloud Storage
gsutil -m rsync -r s3://gator-content gs://gator-content

# GCP to Azure
azcopy sync "https://storage.googleapis.com/gator-content" \
  "https://gatorstorage.blob.core.windows.net/content"
```

---

## Support and Resources

### Cloud Provider Support
- **AWS**: AWS Support Plans
- **GCP**: Google Cloud Support
- **Azure**: Azure Support Plans

### Documentation
- [AWS Documentation](https://docs.aws.amazon.com/)
- [GCP Documentation](https://cloud.google.com/docs)
- [Azure Documentation](https://docs.microsoft.com/azure)

### Cost Calculators
- [AWS Pricing Calculator](https://calculator.aws/)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)

---

## Conclusion

Choose the cloud provider that best fits your:
- Budget requirements
- Geographic distribution needs
- Existing infrastructure
- Team expertise
- Compliance requirements

All three major cloud providers can successfully host Gator with similar performance and reliability.

For detailed Kubernetes configuration, see [Kubernetes Deployment Guide](KUBERNETES_DEPLOYMENT.md).
