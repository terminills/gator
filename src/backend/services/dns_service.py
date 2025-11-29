"""
DNS Management Service

Handles domain name management through various DNS providers,
with primary support for GoDaddy DNS API integration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.logging import get_logger

logger = get_logger(__name__)


class DNSProvider(str, Enum):
    """Supported DNS providers."""

    GODADDY = "godaddy"
    CLOUDFLARE = "cloudflare"  # Future implementation
    ROUTE53 = "route53"  # Future implementation
    CUSTOM = "custom"  # Manual configuration


class RecordType(str, Enum):
    """DNS record types."""

    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    TXT = "TXT"
    NS = "NS"
    SOA = "SOA"


class DNSRecord(BaseModel):
    """DNS record model."""

    name: str = Field(..., description="Record name (subdomain)")
    type: RecordType = Field(..., description="Record type")
    data: str = Field(..., description="Record value/data")
    ttl: int = Field(default=3600, description="Time to live in seconds")
    priority: Optional[int] = Field(None, description="Priority (for MX records)")


class DomainInfo(BaseModel):
    """Domain information model."""

    domain: str = Field(..., description="Domain name")
    provider: DNSProvider = Field(..., description="DNS provider")
    status: str = Field(..., description="Domain status")
    expires: Optional[datetime] = Field(None, description="Domain expiration date")
    nameservers: List[str] = Field(
        default_factory=list, description="Domain nameservers"
    )
    records: List[DNSRecord] = Field(default_factory=list, description="DNS records")


class GoDaddyDNSClient:
    """GoDaddy DNS API client."""

    def __init__(self, api_key: str, api_secret: str, environment: str = "production"):
        """
        Initialize GoDaddy DNS client.

        Args:
            api_key: GoDaddy API key
            api_secret: GoDaddy API secret
            environment: API environment ('production' or 'ote' for testing)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (
            "https://api.godaddy.com"
            if environment == "production"
            else "https://api.ote-godaddy.com"
        )
        self.headers = {
            "Authorization": f"sso-key {api_key}:{api_secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """
        Get domain information from GoDaddy.

        Args:
            domain: Domain name

        Returns:
            Domain information
        """
        try:
            response = await self.client.get(f"{self.base_url}/v1/domains/{domain}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get domain info for {domain}: {e}")
            raise

    async def get_dns_records(
        self, domain: str, record_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get DNS records for a domain.

        Args:
            domain: Domain name
            record_type: Optional record type filter

        Returns:
            List of DNS records
        """
        try:
            url = f"{self.base_url}/v1/domains/{domain}/records"
            if record_type:
                url += f"/{record_type}"

            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get DNS records for {domain}: {e}")
            raise

    async def create_dns_record(self, domain: str, record: DNSRecord) -> bool:
        """
        Create a new DNS record.

        Args:
            domain: Domain name
            record: DNS record to create

        Returns:
            True if successful
        """
        try:
            data = [
                {
                    "name": record.name,
                    "type": record.type,
                    "data": record.data,
                    "ttl": record.ttl,
                }
            ]

            # Add priority for MX records
            if record.type == RecordType.MX and record.priority:
                data[0]["priority"] = record.priority

            response = await self.client.patch(
                f"{self.base_url}/v1/domains/{domain}/records/{record.type}/{record.name}",
                json=data,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to create DNS record for {domain}: {e}")
            return False

    async def update_dns_record(self, domain: str, record: DNSRecord) -> bool:
        """
        Update an existing DNS record.

        Args:
            domain: Domain name
            record: DNS record to update

        Returns:
            True if successful
        """
        try:
            data = [
                {
                    "name": record.name,
                    "type": record.type,
                    "data": record.data,
                    "ttl": record.ttl,
                }
            ]

            # Add priority for MX records
            if record.type == RecordType.MX and record.priority:
                data[0]["priority"] = record.priority

            response = await self.client.put(
                f"{self.base_url}/v1/domains/{domain}/records/{record.type}/{record.name}",
                json=data,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to update DNS record for {domain}: {e}")
            return False

    async def delete_dns_record(self, domain: str, record_type: str, name: str) -> bool:
        """
        Delete a DNS record.

        Args:
            domain: Domain name
            record_type: Record type
            name: Record name

        Returns:
            True if successful
        """
        try:
            response = await self.client.delete(
                f"{self.base_url}/v1/domains/{domain}/records/{record_type}/{name}"
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to delete DNS record for {domain}: {e}")
            return False

    async def setup_gator_dns(self, domain: str, server_ip: str) -> bool:
        """
        Setup default DNS records for Gator platform.

        Args:
            domain: Domain name
            server_ip: Server IP address

        Returns:
            True if successful
        """
        try:
            # Default DNS records for Gator platform
            records = [
                DNSRecord(name="@", type=RecordType.A, data=server_ip, ttl=600),
                DNSRecord(name="www", type=RecordType.A, data=server_ip, ttl=600),
                DNSRecord(name="api", type=RecordType.A, data=server_ip, ttl=600),
                DNSRecord(name="admin", type=RecordType.A, data=server_ip, ttl=600),
                DNSRecord(name="cdn", type=RecordType.A, data=server_ip, ttl=600),
            ]

            success_count = 0
            for record in records:
                if await self.create_dns_record(domain, record):
                    success_count += 1
                    logger.info(
                        f"Created DNS record: {record.name}.{domain} -> {record.data}"
                    )
                else:
                    logger.error(f"Failed to create DNS record: {record.name}.{domain}")

            return success_count == len(records)

        except Exception as e:
            logger.error(f"Failed to setup Gator DNS for {domain}: {e}")
            return False


class DNSService:
    """
    DNS Management Service.

    Provides unified interface for managing DNS records across multiple providers.
    """

    def __init__(self, db_session: AsyncSession):
        """
        Initialize DNS service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.providers = {}

    def add_provider(self, name: str, provider: Union[GoDaddyDNSClient]):
        """
        Add a DNS provider client.

        Args:
            name: Provider identifier
            provider: DNS provider client
        """
        self.providers[name] = provider

    async def get_domain_info(
        self, domain: str, provider: DNSProvider = DNSProvider.GODADDY
    ) -> Optional[DomainInfo]:
        """
        Get domain information.

        Args:
            domain: Domain name
            provider: DNS provider

        Returns:
            Domain information or None if not found
        """
        try:
            if provider == DNSProvider.GODADDY and "godaddy" in self.providers:
                client = self.providers["godaddy"]

                # Get domain details
                domain_data = await client.get_domain_info(domain)

                # Get DNS records
                records_data = await client.get_dns_records(domain)

                # Convert to DNSRecord objects
                records = []
                for record_data in records_data:
                    records.append(
                        DNSRecord(
                            name=record_data.get("name", "@"),
                            type=RecordType(record_data["type"]),
                            data=record_data["data"],
                            ttl=record_data.get("ttl", 3600),
                            priority=record_data.get("priority"),
                        )
                    )

                return DomainInfo(
                    domain=domain,
                    provider=provider,
                    status=domain_data.get("status", "unknown"),
                    expires=(
                        datetime.fromisoformat(
                            domain_data["expires"].replace("Z", "+00:00")
                        )
                        if "expires" in domain_data
                        else None
                    ),
                    nameservers=domain_data.get("nameServers", []),
                    records=records,
                )

            else:
                logger.error(f"Unsupported DNS provider: {provider}")
                return None

        except Exception as e:
            logger.error(f"Failed to get domain info for {domain}: {e}")
            return None

    async def create_dns_record(
        self,
        domain: str,
        record: DNSRecord,
        provider: DNSProvider = DNSProvider.GODADDY,
    ) -> bool:
        """
        Create a DNS record.

        Args:
            domain: Domain name
            record: DNS record to create
            provider: DNS provider

        Returns:
            True if successful
        """
        try:
            if provider == DNSProvider.GODADDY and "godaddy" in self.providers:
                client = self.providers["godaddy"]
                return await client.create_dns_record(domain, record)
            else:
                logger.error(f"Unsupported DNS provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to create DNS record for {domain}: {e}")
            return False

    async def update_dns_record(
        self,
        domain: str,
        record: DNSRecord,
        provider: DNSProvider = DNSProvider.GODADDY,
    ) -> bool:
        """
        Update a DNS record.

        Args:
            domain: Domain name
            record: DNS record to update
            provider: DNS provider

        Returns:
            True if successful
        """
        try:
            if provider == DNSProvider.GODADDY and "godaddy" in self.providers:
                client = self.providers["godaddy"]
                return await client.update_dns_record(domain, record)
            else:
                logger.error(f"Unsupported DNS provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to update DNS record for {domain}: {e}")
            return False

    async def delete_dns_record(
        self,
        domain: str,
        record_type: str,
        name: str,
        provider: DNSProvider = DNSProvider.GODADDY,
    ) -> bool:
        """
        Delete a DNS record.

        Args:
            domain: Domain name
            record_type: Record type
            name: Record name
            provider: DNS provider

        Returns:
            True if successful
        """
        try:
            if provider == DNSProvider.GODADDY and "godaddy" in self.providers:
                client = self.providers["godaddy"]
                return await client.delete_dns_record(domain, record_type, name)
            else:
                logger.error(f"Unsupported DNS provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete DNS record for {domain}: {e}")
            return False

    async def setup_gator_platform(
        self, domain: str, server_ip: str, provider: DNSProvider = DNSProvider.GODADDY
    ) -> bool:
        """
        Setup DNS records for Gator platform.

        Args:
            domain: Domain name
            server_ip: Server IP address
            provider: DNS provider

        Returns:
            True if successful
        """
        try:
            if provider == DNSProvider.GODADDY and "godaddy" in self.providers:
                client = self.providers["godaddy"]
                return await client.setup_gator_dns(domain, server_ip)
            else:
                logger.error(f"Unsupported DNS provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to setup Gator platform DNS for {domain}: {e}")
            return False

    async def validate_dns_propagation(
        self, domain: str, expected_ip: str, timeout: int = 300
    ) -> bool:
        """
        Validate DNS propagation by checking if domain resolves to expected IP.

        Args:
            domain: Domain name to check
            expected_ip: Expected IP address
            timeout: Maximum time to wait in seconds

        Returns:
            True if DNS has propagated correctly
        """
        import asyncio
        import socket

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                resolved_ip = socket.gethostbyname(domain)
                if resolved_ip == expected_ip:
                    logger.info(
                        f"DNS propagation successful for {domain} -> {expected_ip}"
                    )
                    return True
                else:
                    logger.info(
                        f"DNS not yet propagated for {domain}: {resolved_ip} != {expected_ip}"
                    )

            except socket.gaierror:
                logger.info(f"DNS resolution failed for {domain}, waiting...")

            await asyncio.sleep(10)  # Check every 10 seconds

        logger.warning(f"DNS propagation timeout for {domain}")
        return False

    async def close(self):
        """Close all provider clients."""
        for provider in self.providers.values():
            if hasattr(provider, "close"):
                await provider.close()


# Factory function to create configured DNS service
async def create_dns_service(
    db_session: AsyncSession,
    godaddy_api_key: Optional[str] = None,
    godaddy_api_secret: Optional[str] = None,
) -> DNSService:
    """
    Create a configured DNS service with available providers.

    Args:
        db_session: Database session
        godaddy_api_key: GoDaddy API key
        godaddy_api_secret: GoDaddy API secret

    Returns:
        Configured DNS service
    """
    service = DNSService(db_session)

    # Add GoDaddy provider if credentials are available
    if godaddy_api_key and godaddy_api_secret:
        godaddy_client = GoDaddyDNSClient(godaddy_api_key, godaddy_api_secret)
        service.add_provider("godaddy", godaddy_client)
        logger.info("GoDaddy DNS provider initialized")

    return service
