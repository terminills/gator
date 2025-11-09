# 🦎 Gator AI Influencer Platform

**Gator don't play no shit**

Gator is a comprehensive AI-powered platform that enables you to create, manage, and deploy AI influencers with complete control over their persona, content generation, and social media presence. Built for privacy, customization, and professional-grade content creation.

## 🚀 Quick Start

### 1. Automated Server Setup (Ubuntu/Debian)
```bash
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash
```

### 2. Manual Setup
```bash
# Clone and setup
git clone https://github.com/terminills/gator.git
cd gator
python setup_db.py  # Initialize database
python demo.py      # Run demo to verify setup

# Start the platform
cd src && python -m backend.api.main
# Visit http://localhost:8000 for the dashboard
```

### 3. Updating an Existing Installation
```bash
# Update dependencies and migrate database
./update.sh

# For verbose output
./update.sh --verbose

# Skip verification step
./update.sh --skip-verification
```

## ✨ Current Features

### 🎭 AI Persona Management
- **Complete Persona System**: Create and manage AI influencer personas with customizable appearance, personality, and content themes
- **Style Preferences**: Define visual aesthetics, content tone, and generation parameters
- **Multi-Persona Support**: Manage multiple AI influencers from a single dashboard
- **Persona Templates**: Pre-built templates for different influencer types (tech, lifestyle, creative, etc.)

### 🎨 Content Generation Pipeline  
- **Local AI Image Generation**: Privacy-focused image generation using Stable Diffusion (no API costs!)
- **API-Based Image Generation**: Integrated support for DALL-E 3 and other cloud services
- **Text Content Generation**: LLM integration for captions, posts, and engagement content
- **Video Generation Support**: Framework ready for video content creation
- **Style Consistency**: Maintains visual and tonal consistency across all generated content
- **Content Quality Control**: Multiple quality levels (draft, standard, high) with post-processing

### 📱 Social Media Integration
- **Multi-Platform Publishing**: Support for Instagram, Facebook, Twitter, TikTok, LinkedIn
- **Automated Scheduling**: Smart scheduling based on engagement patterns
- **Content Adaptation**: Automatically formats content for different platform requirements
- **Engagement Tracking**: Real-time analytics and performance metrics
- **Cross-Platform Sync**: Coordinate campaigns across multiple social networks

### 📊 RSS Feed & Trend Analysis
- **RSS Feed Ingestion**: Monitor news feeds and trending topics for content inspiration
- **Topic Extraction**: AI-powered analysis of current events and trends
- **Content Triggers**: Automatically generate relevant content based on trending topics
- **Knowledge Base**: Searchable database of ingested content for persona context

### 🎯 Administrative Dashboard
- **Web-Based Control Panel**: Modern, responsive interface for complete platform management
- **Real-Time Analytics**: Monitor content generation, engagement rates, and system performance
- **API Management**: Configure external services and API keys
- **User Management**: Multi-user support with role-based access control
- **Domain Management**: Built-in GoDaddy DNS integration for automated domain setup
- **Database Management**: One-click database backups and schema synchronization

### 🏗️ Technical Architecture
- **FastAPI Backend**: High-performance async Python backend with comprehensive API
- **SQLAlchemy ORM**: Robust database layer with async operations and migrations  
- **Modular Design**: Clean separation of concerns with pluggable service architecture
- **Production Ready**: Structured logging, monitoring, and error handling
- **Security First**: Authentication, input validation, and security middleware
- **Scalable**: Designed for horizontal scaling and high availability

### 🔒 Privacy & Security
- **Self-Hosted**: Complete data sovereignty - runs entirely on your infrastructure
- **No External Dependencies**: All AI processing happens on your hardware
- **Encrypted Storage**: Secure storage of persona data and generated content  
- **Access Control**: Role-based permissions and API key management
- **Audit Logging**: Comprehensive logging for security and compliance

### 📊 Analytics & Monitoring
- **Real-Time Metrics**: Live dashboards for content generation and engagement
- **Performance Tracking**: Monitor system resource usage and API response times
- **Content Analytics**: Track engagement rates, click-through rates, and audience metrics
- **Usage Reports**: Detailed reporting for content generation and publishing statistics

## 🔮 Planned Features

### 🎬 Advanced Content Generation
- **Video Generation**: Full video creation pipeline with AI-powered editing
- **Voice Synthesis**: Custom voice generation for video content and podcasts
- **Interactive Content**: Polls, stories, and engagement-driven content formats
- **3D Avatar Support**: Integration with 3D character models and animation

### 🤖 Enhanced AI Capabilities  
- **Conversation AI**: Real-time chat and comment response automation
- **Sentiment Analysis**: Advanced mood and trend analysis for content optimization
- **Personalized Content**: Audience-specific content generation based on engagement data
- **Multi-Modal AI**: Combined text, image, and video generation in single workflows

### 🌐 Platform Expansion
- **Mobile App**: Native mobile applications for iOS and Android
- **API Marketplace**: Plugin system for third-party integrations
- **White Label**: Complete white-label solution for agencies and enterprises
- **Multi-Tenancy**: Support for multiple organizations within single installation

### ☁️ Cloud & Enterprise Features
- **Kubernetes Support**: Container orchestration for enterprise deployments
- **Cloud Deployment**: One-click deployment to AWS, Google Cloud, and Azure
- **Load Balancing**: Built-in load balancing and auto-scaling capabilities
- **Backup & Recovery**: Automated backup systems and disaster recovery

## 💻 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ or Debian 11+
- **CPU**: 8+ cores (Intel/AMD x64)
- **RAM**: 16GB DDR4
- **Storage**: 100GB SSD
- **GPU**: NVIDIA GTX 1060+ (6GB VRAM)
- **Network**: 1Gbps connection

### Recommended Configuration  
- **OS**: Ubuntu 22.04 LTS
- **CPU**: AMD EPYC 7502 (32+ cores) or Intel Xeon equivalent
- **RAM**: 64GB+ DDR4
- **Storage**: 1TB NVMe SSD + 10TB traditional storage
- **GPU**: 5x AMD MI25 or NVIDIA RTX 4090
- **Network**: 5Gbps+ fiber connection

## 🛠️ Installation & Setup

### Automated Installation (Recommended)
```bash
# Download and run automated setup script
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash -s -- --domain your-domain.com --email admin@your-domain.com

# With AMD GPU support (auto-detects MI25)
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash -s -- --rocm --domain your-domain.com --email admin@your-domain.com

# With NVIDIA GPU support
curl -sSL https://raw.githubusercontent.com/terminills/gator/main/server-setup.sh | sudo bash -s -- --nvidia --domain your-domain.com --email admin@your-domain.com
```

**Note for AMD MI25 Users**: The setup script automatically detects MI25 GPUs and installs ROCm 5.7.1 with gfx900-optimized configuration. See [MI25 Compatibility Guide](docs/MI25_COMPATIBILITY.md) for details.

### Manual Installation
```bash
# 1. Clone repository
git clone https://github.com/terminills/gator.git
cd gator

# 2. Install dependencies
pip install -e .

# 3. Setup environment
cp .env.template .env
# Edit .env with your configuration

# 4. Initialize database
python setup_db.py

# 5. Run demo to verify installation
python demo.py

# 6. Start the platform
cd src && python -m backend.api.main
```

### Docker Installation
```bash
# Quick start with Docker Compose
git clone https://github.com/terminills/gator.git
cd gator
docker-compose up -d
```

## 🎮 GPU Support

### AMD GPUs (ROCm)

Gator supports AMD GPUs through ROCm with automatic version detection and optimization:

#### Next-Gen Support (ROCm 7.0+) - **PyTorch 2.10** 🆕
- **Full PyTorch 2.10 nightly support** with automatic dependency compatibility checking
- **ROCm 7.0+ detection**: Automatically detects and installs correct PyTorch versions
- **Intelligent dependency management**: Compatible versions of transformers, diffusers, and accelerate
- See [PyTorch Version Compatibility Guide](PYTORCH_VERSION_COMPATIBILITY.md) for details

#### Modern GPUs (ROCm 6.5+) - **Recommended**
- **Radeon Pro V620 (RDNA2/gfx1030)**: Fully supported with ROCm 6.5+ ✨
  - Multi-GPU: 2-8 cards (96GB-256GB VRAM)
  - Standard PyTorch wheels available
  - Nightly builds supported
- **MI210/MI250 (CDNA2/gfx90a)**: ROCm 6.5+
- **RX 7900 Series (RDNA3/gfx1100)**: ROCm 6.5+
- **RX 6000 Series (RDNA2/gfx1030)**: ROCm 6.5+

#### Legacy GPUs (ROCm 5.7)
- **MI25 (Vega 10/gfx900)**: Fully supported with ROCm 5.7.1
  - See [MI25 Compatibility Guide](docs/MI25_COMPATIBILITY.md)

#### Multi-GPU Support 🚀
Gator automatically detects and optimizes for multi-GPU configurations:
- **2 GPUs**: Parallel inference and load balancing
- **3 GPUs**: Specialized tasks (LLM + Image + Video)
- **4+ GPUs**: Enterprise multi-tenant deployment

See [Multi-GPU Setup Guide](docs/MULTI_GPU_SETUP.md) for V620 and other modern GPU configurations.

Installation:
```bash
# Automatic detection and installation
sudo bash server-setup.sh --rocm

# Check PyTorch version and compatible dependencies
python demo_pytorch_version_check.py

# Install PyTorch 2.10 nightly with ROCm 7.0
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

**PyTorch Version Compatibility**: The system automatically detects your PyTorch version and ensures compatible versions of ML libraries are installed. See [PyTorch Version Compatibility Guide](PYTORCH_VERSION_COMPATIBILITY.md) for details.

### NVIDIA GPUs (CUDA)

- **RTX 40 Series**: Full support with CUDA 12.0+
- **RTX 30 Series**: Full support with CUDA 11.x+
- **GTX 16/10 Series**: Supported with CUDA 11.x+

Installation:
```bash
# Automatic detection and installation
sudo bash server-setup.sh --nvidia
```

### CPU-Only Mode

Gator can run without GPU using CPU-based inference engines, though performance will be reduced:
```bash
# Install without GPU support
sudo bash server-setup.sh
```

## 🌍 Domain & DNS Management

Gator includes built-in GoDaddy DNS integration for automated domain management:

1. **Purchase Domain**: Buy domain through GoDaddy or transfer existing domain
2. **Configure API**: Add GoDaddy API credentials in admin panel
3. **Automatic Setup**: Platform automatically configures DNS records, SSL certificates, and subdomains
4. **Monitoring**: Real-time DNS status monitoring and automatic renewal

### Supported DNS Providers
- GoDaddy (built-in)
- Cloudflare (planned)
- Route53 (planned) 
- Custom DNS (manual configuration)

## 🚀 API Documentation

### REST API Endpoints
- **Personas**: `/api/v1/personas/` - Manage AI personas
- **Content**: `/api/v1/content/` - Generated content management
- **Social**: `/api/v1/social/` - Social media integration
- **Analytics**: `/api/v1/analytics/` - Platform metrics and reporting
- **Feeds**: `/api/v1/feeds/` - RSS feed management

Interactive API documentation available at `http://your-domain.com:8000/docs`

### WebSocket API
- **Real-time Updates**: Live content generation progress
- **System Monitoring**: Real-time system metrics
- **Chat Interface**: Live persona interaction testing

## 📱 Dashboard Features

### Main Dashboard
- **System Overview**: Platform health, resource usage, and active processes
- **Quick Actions**: Generate content, schedule posts, manage personas
- **Recent Activity**: Latest generated content and system events
- **Performance Metrics**: Real-time analytics and engagement statistics

### Persona Management
- **Visual Editor**: Drag-and-drop persona configuration interface
- **Template Library**: Pre-built persona templates for different niches
- **Style Gallery**: Visual style selection and customization tools
- **Preview System**: Real-time preview of persona-generated content

### Content Library
- **Generated Content**: Browse, edit, and manage all AI-generated content
- **Batch Operations**: Bulk content generation and management
- **Quality Control**: Content approval workflows and moderation tools
- **Publishing Queue**: Scheduled content with publishing calendar

### Analytics Dashboard
- **Engagement Metrics**: Detailed analytics for all social media platforms
- **Audience Insights**: Demographic and behavioral analysis
- **Content Performance**: Track which content types perform best
- **ROI Tracking**: Revenue and conversion tracking for monetized content

## 🔧 Configuration

### Environment Variables
```bash
# Core Settings
GATOR_ENV=production
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost/gator

# AI Model Configuration  
AI_MODEL_PATH=/path/to/models
HUGGING_FACE_TOKEN=your-token
OPENAI_API_KEY=your-key

# Social Media APIs
FACEBOOK_API_KEY=your-key
INSTAGRAM_API_KEY=your-key
TWITTER_API_KEY=your-key

# DNS Management
GODADDY_API_KEY=your-key
GODADDY_API_SECRET=your-secret
DEFAULT_DOMAIN=your-domain.com

# Storage & CDN
STORAGE_BACKEND=local  # local, s3, gcs
CDN_ENDPOINT=https://cdn.your-domain.com
```

### Database Configuration
Gator supports multiple database backends:
- **PostgreSQL** (recommended for production)
- **SQLite** (development only)
- **MySQL/MariaDB** (alternative option)

## 🔐 Security & Compliance

### Security Features
- **OAuth 2.0**: Industry-standard authentication
- **API Rate Limiting**: Protect against abuse and overuse
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Parameterized queries and ORM security
- **XSS Prevention**: Content sanitization and CSP headers
- **HTTPS Enforcement**: TLS encryption for all communications

### Compliance Support
- **GDPR Ready**: Data privacy controls and user consent management
- **CCPA Compliance**: California privacy law compliance tools
- **Content Labeling**: AI-generated content disclosure and watermarking
- **Age Verification**: Built-in age verification for adult content platforms
- **Audit Trails**: Comprehensive logging for regulatory compliance

## 🔧 Maintenance & Updates

### Updating Your Installation
Keep your Gator installation up-to-date with the latest features, bug fixes, and security patches:

```bash
# Standard update (recommended)
./update.sh

# Update with detailed output
./update.sh --verbose

# Quick update without verification
./update.sh --skip-verification

# Update without running migrations (if already applied)
./update.sh --skip-migrations
```

The update script automatically:
- ✅ Checks Python version and prerequisites
- ✅ Updates pip to the latest version
- ✅ Updates all Python dependencies
- ✅ Runs database migration scripts
- ✅ Updates database schema to latest version
- ✅ Verifies installation integrity

### Manual Database Migrations
If you need to run migrations manually:

```bash
# Run a specific migration
python migrate_add_base_image_status.py
python migrate_add_appearance_locking.py

# Reinitialize database (caution: may reset data)
python setup_db.py
```

### Backup Before Updates
Always backup your database before updating:

```bash
# SQLite backup
cp gator.db gator.db.backup.$(date +%Y%m%d)

# PostgreSQL backup
pg_dump -U gator_user gator_production > backup_$(date +%Y%m%d).sql
```

## 📞 Support & Community

### Documentation
- **API Reference**: Complete API documentation with examples
- **Developer Guide**: In-depth development and customization guide  
- **Deployment Guide**: Production deployment best practices
- **[Local Image Generation Guide](LOCAL_IMAGE_GENERATION.md)**: Setup and usage for local AI image generation
- **Troubleshooting**: Common issues and solutions

### Community Resources
- **GitHub Discussions**: Community Q&A and feature requests
- **Discord Server**: Real-time community chat and support
- **Video Tutorials**: Step-by-step setup and usage guides
- **Blog**: Updates, tutorials, and industry insights

### Professional Support
- **Priority Support**: Dedicated support for enterprise customers
- **Custom Development**: Tailored features and integrations
- **Training**: On-site training and onboarding assistance
- **SLA Options**: Service level agreements for mission-critical deployments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code of Conduct
- Development Workflow  
- Pull Request Process
- Issue Reporting

## ⭐ Star History

If you find Gator helpful, please consider giving it a star on GitHub to help others discover the project!
