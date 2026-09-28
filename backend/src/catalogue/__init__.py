"""Catalogue management package."""
from src.catalogue.models import Camera
from src.catalogue.parser import CatalogueParser
from src.catalogue.validator import CatalogueValidator
from src.catalogue.client import CatalogueClient, AuthenticationRequiredError

__all__ = [
    "Camera",
    "CatalogueParser",
    "CatalogueValidator",
    "CatalogueClient",
    "AuthenticationRequiredError",
]
