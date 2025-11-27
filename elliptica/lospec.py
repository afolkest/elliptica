"""Lospec palette API integration.

Provides utilities for fetching palettes from Lospec (https://lospec.com).
Can be reused in other projects that need color palette data.
"""

from __future__ import annotations
from typing import Optional
import requests


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to RGB tuple (0-1 range).

    Args:
        hex_color: Hex color string (with or without '#' prefix)

    Returns:
        Tuple of (r, g, b) in [0, 1] range

    Example:
        >>> hex_to_rgb("ff5733")
        (1.0, 0.3411764705882353, 0.2)
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def fetch_random_palette(timeout: float = 10.0) -> dict:
    """Fetch a random palette from Lospec.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary with keys:
            - name (str): Palette name
            - author (str): Creator name
            - slug (str): URL slug
            - colors (list[tuple]): RGB colors as (r, g, b) tuples in [0, 1]
            - hex_colors (list[str]): Original hex color strings

    Raises:
        requests.RequestException: If network request fails
        ValueError: If response format is invalid
    """
    # Step 1: Get random palette URL via redirect
    response = requests.get(
        'https://lospec.com/palette-list/random',
        allow_redirects=True,
        timeout=timeout
    )
    response.raise_for_status()

    # Step 2: Extract slug from final URL
    slug = response.url.split('/palette-list/')[-1]

    # Step 3: Fetch palette data via API
    api_response = requests.get(
        f'https://lospec.com/palette-list/{slug}.json',
        timeout=timeout
    )
    api_response.raise_for_status()

    data = api_response.json()

    # Step 4: Convert hex colors to RGB tuples
    hex_colors = data['colors']
    rgb_colors = [hex_to_rgb(c) for c in hex_colors]

    return {
        'name': data['name'],
        'author': data['author'],
        'slug': slug,
        'colors': rgb_colors,
        'hex_colors': hex_colors,
    }


def fetch_palette_by_slug(slug: str, timeout: float = 10.0) -> dict:
    """Fetch a specific palette by its Lospec slug.

    Args:
        slug: Palette slug (e.g., "apollo")
        timeout: Request timeout in seconds

    Returns:
        Dictionary with keys:
            - name (str): Palette name
            - author (str): Creator name
            - slug (str): URL slug
            - colors (list[tuple]): RGB colors as (r, g, b) tuples in [0, 1]
            - hex_colors (list[str]): Original hex color strings

    Raises:
        requests.RequestException: If network request fails
        ValueError: If response format is invalid
    """
    response = requests.get(
        f'https://lospec.com/palette-list/{slug}.json',
        timeout=timeout
    )
    response.raise_for_status()

    data = response.json()

    # Convert hex colors to RGB tuples
    hex_colors = data['colors']
    rgb_colors = [hex_to_rgb(c) for c in hex_colors]

    return {
        'name': data['name'],
        'author': data['author'],
        'slug': slug,
        'colors': rgb_colors,
        'hex_colors': hex_colors,
    }


def fetch_palette_by_url(url: str, timeout: float = 10.0) -> dict:
    """Fetch a palette from a Lospec URL.

    Args:
        url: Full Lospec palette URL (e.g., "https://lospec.com/palette-list/apollo")
        timeout: Request timeout in seconds

    Returns:
        Dictionary with palette data (same as fetch_palette_by_slug)

    Raises:
        ValueError: If URL format is invalid
        requests.RequestException: If network request fails
    """
    if '/palette-list/' not in url:
        raise ValueError(f"Invalid Lospec palette URL: {url}")

    slug = url.split('/palette-list/')[-1]
    # Remove any trailing query params or fragments
    slug = slug.split('?')[0].split('#')[0]

    return fetch_palette_by_slug(slug, timeout=timeout)
