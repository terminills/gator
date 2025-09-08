"""
DNS Management API Routes

Provides endpoints for domain and DNS management through the admin panel.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel, Field

from backend.services.dns_service import (
    DNSService, DNSRecord, DomainInfo, DNSProvider, RecordType,
    create_dns_service
)
from backend.database.connection import get_db_session
from backend.config.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(prefix="/dns", tags=["dns"])


class CreateDNSRecordRequest(BaseModel):
    """Request to create a DNS record."""
    name: str = Field(..., description="Record name (subdomain)")
    type: RecordType = Field(..., description="Record type")
    data: str = Field(..., description="Record value/data")
    ttl: int = Field(default=3600, ge=300, le=86400, description="Time to live in seconds")
    priority: Optional[int] = Field(None, description="Priority (for MX records)")


class UpdateDNSRecordRequest(BaseModel):
    """Request to update a DNS record."""
    data: str = Field(..., description="New record value/data")
    ttl: int = Field(default=3600, ge=300, le=86400, description="Time to live in seconds")
    priority: Optional[int] = Field(None, description="Priority (for MX records)")


class SetupPlatformRequest(BaseModel):
    """Request to setup platform DNS."""
    domain: str = Field(..., description="Domain name")
    server_ip: str = Field(..., description="Server IP address")
    provider: DNSProvider = Field(default=DNSProvider.GODADDY, description="DNS provider")


class DNSProviderConfig(BaseModel):
    """DNS provider configuration."""
    provider: DNSProvider = Field(..., description="Provider type")
    api_key: str = Field(..., description="API key")
    api_secret: str = Field(..., description="API secret")
    environment: str = Field(default="production", description="Environment (production/ote)")


class DNSRecordResponse(BaseModel):
    """DNS record response."""
    name: str
    type: str
    data: str
    ttl: int
    priority: Optional[int] = None


class DomainInfoResponse(BaseModel):
    """Domain information response."""
    domain: str
    provider: str
    status: str
    expires: Optional[str] = None
    nameservers: List[str]
    records: List[DNSRecordResponse]


async def get_dns_service(session = Depends(get_db_session)):
    """Dependency to get DNS service."""
    settings = get_settings()
    
    # Get GoDaddy credentials from settings
    godaddy_key = getattr(settings, 'GODADDY_API_KEY', None)
    godaddy_secret = getattr(settings, 'GODADDY_API_SECRET', None)
    
    service = await create_dns_service(
        db_session=session,
        godaddy_api_key=godaddy_key,
        godaddy_api_secret=godaddy_secret
    )
    
    try:
        yield service
    finally:
        await service.close()


@router.get("/domain/{domain}", response_model=DomainInfoResponse)
async def get_domain_info(
    domain: str,
    provider: DNSProvider = Query(default=DNSProvider.GODADDY),
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Get domain information and DNS records.
    
    Args:
        domain: Domain name to query
        provider: DNS provider to use
        
    Returns:
        Domain information including DNS records
    """
    try:
        domain_info = await dns_service.get_domain_info(domain, provider)
        
        if not domain_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Domain {domain} not found or inaccessible"
            )
        
        # Convert to response format
        records = [
            DNSRecordResponse(
                name=record.name,
                type=record.type,
                data=record.data,
                ttl=record.ttl,
                priority=record.priority
            )
            for record in domain_info.records
        ]
        
        return DomainInfoResponse(
            domain=domain_info.domain,
            provider=domain_info.provider,
            status=domain_info.status,
            expires=domain_info.expires.isoformat() if domain_info.expires else None,
            nameservers=domain_info.nameservers,
            records=records
        )
        
    except Exception as e:
        logger.error(f"Failed to get domain info for {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve domain information: {str(e)}"
        )


@router.post("/domain/{domain}/records")
async def create_dns_record(
    domain: str,
    record_request: CreateDNSRecordRequest,
    provider: DNSProvider = Query(default=DNSProvider.GODADDY),
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Create a new DNS record.
    
    Args:
        domain: Domain name
        record_request: DNS record details
        provider: DNS provider to use
        
    Returns:
        Success status
    """
    try:
        record = DNSRecord(
            name=record_request.name,
            type=record_request.type,
            data=record_request.data,
            ttl=record_request.ttl,
            priority=record_request.priority
        )
        
        success = await dns_service.create_dns_record(domain, record, provider)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create DNS record"
            )
        
        logger.info(f"Created DNS record: {record.name}.{domain} -> {record.data}")
        
        return {
            "success": True,
            "message": f"DNS record created successfully",
            "record": {
                "name": record.name,
                "type": record.type,
                "data": record.data,
                "ttl": record.ttl
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create DNS record for {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create DNS record: {str(e)}"
        )


@router.put("/domain/{domain}/records/{record_type}/{record_name}")
async def update_dns_record(
    domain: str,
    record_type: RecordType,
    record_name: str,
    record_request: UpdateDNSRecordRequest,
    provider: DNSProvider = Query(default=DNSProvider.GODADDY),
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Update an existing DNS record.
    
    Args:
        domain: Domain name
        record_type: Record type
        record_name: Record name
        record_request: Updated record details
        provider: DNS provider to use
        
    Returns:
        Success status
    """
    try:
        record = DNSRecord(
            name=record_name,
            type=record_type,
            data=record_request.data,
            ttl=record_request.ttl,
            priority=record_request.priority
        )
        
        success = await dns_service.update_dns_record(domain, record, provider)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update DNS record"
            )
        
        logger.info(f"Updated DNS record: {record.name}.{domain} -> {record.data}")
        
        return {
            "success": True,
            "message": "DNS record updated successfully",
            "record": {
                "name": record.name,
                "type": record.type,
                "data": record.data,
                "ttl": record.ttl
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update DNS record for {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update DNS record: {str(e)}"
        )


@router.delete("/domain/{domain}/records/{record_type}/{record_name}")
async def delete_dns_record(
    domain: str,
    record_type: str,
    record_name: str,
    provider: DNSProvider = Query(default=DNSProvider.GODADDY),
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Delete a DNS record.
    
    Args:
        domain: Domain name
        record_type: Record type
        record_name: Record name
        provider: DNS provider to use
        
    Returns:
        Success status
    """
    try:
        success = await dns_service.delete_dns_record(domain, record_type, record_name, provider)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete DNS record"
            )
        
        logger.info(f"Deleted DNS record: {record_name}.{domain} ({record_type})")
        
        return {
            "success": True,
            "message": "DNS record deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete DNS record for {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete DNS record: {str(e)}"
        )


@router.post("/setup-platform")
async def setup_platform_dns(
    setup_request: SetupPlatformRequest,
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Setup DNS records for Gator platform.
    
    Creates standard DNS records needed for the platform including:
    - Main domain (A record)
    - www subdomain
    - api subdomain  
    - admin subdomain
    - cdn subdomain
    
    Args:
        setup_request: Platform setup details
        
    Returns:
        Setup status and created records
    """
    try:
        success = await dns_service.setup_gator_platform(
            domain=setup_request.domain,
            server_ip=setup_request.server_ip,
            provider=setup_request.provider
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to setup platform DNS records"
            )
        
        logger.info(f"Setup Gator platform DNS for {setup_request.domain}")
        
        return {
            "success": True,
            "message": "Platform DNS setup completed successfully",
            "domain": setup_request.domain,
            "records_created": [
                f"{setup_request.domain}",
                f"www.{setup_request.domain}",
                f"api.{setup_request.domain}",
                f"admin.{setup_request.domain}",
                f"cdn.{setup_request.domain}"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to setup platform DNS: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup platform DNS: {str(e)}"
        )


@router.post("/validate-propagation/{domain}")
async def validate_dns_propagation(
    domain: str,
    expected_ip: str = Query(..., description="Expected IP address"),
    timeout: int = Query(default=300, ge=60, le=600, description="Timeout in seconds"),
    dns_service: DNSService = Depends(get_dns_service)
):
    """
    Validate DNS propagation by checking if domain resolves to expected IP.
    
    Args:
        domain: Domain to check
        expected_ip: Expected IP address
        timeout: Maximum wait time in seconds
        
    Returns:
        Propagation status
    """
    try:
        logger.info(f"Validating DNS propagation for {domain} -> {expected_ip}")
        
        success = await dns_service.validate_dns_propagation(
            domain=domain,
            expected_ip=expected_ip,
            timeout=timeout
        )
        
        return {
            "success": success,
            "domain": domain,
            "expected_ip": expected_ip,
            "message": (
                "DNS propagation successful" if success 
                else "DNS propagation timeout or failed"
            ),
            "propagated": success
        }
        
    except Exception as e:
        logger.error(f"Failed to validate DNS propagation for {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate DNS propagation: {str(e)}"
        )


@router.get("/providers")
async def list_dns_providers():
    """
    List available DNS providers and their status.
    
    Returns:
        List of DNS providers and configuration status
    """
    settings = get_settings()
    
    providers = []
    
    # Check GoDaddy configuration
    godaddy_configured = bool(
        getattr(settings, 'GODADDY_API_KEY', None) and 
        getattr(settings, 'GODADDY_API_SECRET', None)
    )
    
    providers.append({
        "name": "GoDaddy",
        "provider": "godaddy",
        "configured": godaddy_configured,
        "status": "active" if godaddy_configured else "not_configured",
        "supported_features": [
            "create_records",
            "update_records", 
            "delete_records",
            "domain_info",
            "platform_setup"
        ]
    })
    
    # Future providers
    providers.extend([
        {
            "name": "Cloudflare",
            "provider": "cloudflare",
            "configured": False,
            "status": "planned",
            "supported_features": []
        },
        {
            "name": "AWS Route53",
            "provider": "route53", 
            "configured": False,
            "status": "planned",
            "supported_features": []
        }
    ])
    
    return {
        "providers": providers,
        "default_provider": "godaddy"
    }