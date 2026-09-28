"""Parser for raw camera catalogue payloads into normalized Camera models."""
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from src.catalogue.models import Camera


class CatalogueParser:
    """Parses arbitrary camera catalogue JSON into normalized Camera models."""

    def __init__(
        self,
        gateway_rtsp_template: Optional[str] = None,
        gateway_hls_template: Optional[str] = None,
        gateway_webrtc_template: Optional[str] = None,
    ):
        self.gateway_rtsp_template = gateway_rtsp_template or os.getenv(
            "SENTINEL_RTSP_TEMPLATE", "rtsp://103.250.160.189:8554/stream/{camera_id}"
        )
        self.gateway_hls_template = gateway_hls_template or os.getenv(
            "SENTINEL_HLS_TEMPLATE", "https://cctv.corp8.cloud/{camera_id}/index.m3u8"
        )
        self.gateway_webrtc_template = gateway_webrtc_template or os.getenv(
            "SENTINEL_WEBRTC_TEMPLATE", "http://103.250.160.189:8889/stream/{camera_id}/whep"
        )

    @staticmethod
    def _normalize_id(raw: Dict[str, Any]) -> Optional[str]:
        """Extract and normalize camera ID."""
        for key in ["camera_id", "id", "cam_id", "cameraId", "camera_number", "number"]:
            if key in raw and raw[key] is not None:
                val = str(raw[key]).strip()
                if val:
                    return val
        return None

    @staticmethod
    def _normalize_codec(raw_codec: Optional[str]) -> str:
        """Normalize codec representation to H264, H265, or UNKNOWN."""
        if not raw_codec:
            return "UNKNOWN"
        c = str(raw_codec).strip().lower()
        if c in ["h264", "h.264", "avc", "avc1"]:
            return "H264"
        if c in ["h265", "h.265", "hevc", "hev1"]:
            return "H265"
        return raw_codec.strip().upper()

    @staticmethod
    def _normalize_status(raw_status: Optional[str]) -> str:
        """Normalize operational status."""
        if not raw_status:
            return "unknown"
        s = str(raw_status).strip().lower()
        if s in ["active", "online", "running", "live"]:
            return "active"
        if s in ["inactive", "offline", "stopped", "disabled"]:
            return "inactive"
        if s in ["maintenance", "service"]:
            return "maintenance"
        return s

    @staticmethod
    def _extract_resolution(raw: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract width and height from explicit fields or composite strings."""
        width = None
        height = None

        # Check explicit integer/string fields
        for w_key in ["width", "w", "resolution_width"]:
            if w_key in raw and raw[w_key] is not None:
                try:
                    width = int(raw[w_key])
                    break
                except (ValueError, TypeError):
                    pass

        for h_key in ["height", "h", "resolution_height"]:
            if h_key in raw and raw[h_key] is not None:
                try:
                    height = int(raw[h_key])
                    break
                except (ValueError, TypeError):
                    pass

        # If still missing, check composite resolution string like "1920x1080"
        if width is None or height is None:
            for res_key in ["resolution", "res", "dimensions"]:
                if res_key in raw and isinstance(raw[res_key], str):
                    m = re.search(r"(\d+)\s*[xX*]\s*(\d+)", raw[res_key])
                    if m:
                        width = int(m.group(1))
                        height = int(m.group(2))
                        break

        return width, height

    @staticmethod
    def _extract_coordinates(raw: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Extract latitude and longitude without inventing coordinates."""
        latitude = None
        longitude = None

        # Check direct fields
        for lat_key in ["latitude", "lat"]:
            if lat_key in raw and raw[lat_key] is not None:
                try:
                    latitude = float(raw[lat_key])
                    break
                except (ValueError, TypeError):
                    pass

        for lng_key in ["longitude", "lng", "lon", "long"]:
            if lng_key in raw and raw[lng_key] is not None:
                try:
                    longitude = float(raw[lng_key])
                    break
                except (ValueError, TypeError):
                    pass

        # Check nested location or coordinates dict/list
        if (latitude is None or longitude is None) and "coordinates" in raw:
            coords = raw["coordinates"]
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    # GeoJSON convention: [longitude, latitude]
                    longitude = float(coords[0])
                    latitude = float(coords[1])
                except (ValueError, TypeError):
                    pass
            elif isinstance(coords, dict):
                try:
                    lat_cand = coords.get("lat") or coords.get("latitude")
                    lng_cand = coords.get("lng") or coords.get("lon") or coords.get("longitude")
                    if lat_cand is not None:
                        latitude = float(lat_cand)
                    if lng_cand is not None:
                        longitude = float(lng_cand)
                except (ValueError, TypeError):
                    pass

        return latitude, longitude

    def parse_camera_dict(self, raw: Dict[str, Any]) -> Camera:
        """Parse a single camera dictionary into a normalized Camera model."""
        camera_id = self._normalize_id(raw) or ""
        name = raw.get("name") or raw.get("label") or raw.get("title")
        if name is not None:
            name = str(name)

        location = raw.get("location") or raw.get("address") or raw.get("site")
        if isinstance(location, dict):
            location = location.get("name") or location.get("address") or str(location)
        elif location is not None:
            location = str(location)

        latitude, longitude = self._extract_coordinates(raw)
        status = self._normalize_status(raw.get("status") or raw.get("state"))
        codec = self._normalize_codec(raw.get("codec") or raw.get("video_codec"))
        width, height = self._extract_resolution(raw)

        fps = None
        for fps_key in ["fps", "frame_rate", "framerate"]:
            if fps_key in raw and raw[fps_key] is not None:
                try:
                    fps = float(raw[fps_key])
                    break
                except (ValueError, TypeError):
                    pass

        bitrate = None
        for br_key in ["bitrate", "bit_rate"]:
            if br_key in raw and raw[br_key] is not None:
                try:
                    bitrate = int(raw[br_key])
                    break
                except (ValueError, TypeError):
                    pass

        # Stream URLs: preserve direct fields if provided; otherwise fallback to gateway template if configured
        rtsp_url = raw.get("rtsp_url") or raw.get("rtsp") or raw.get("stream_url")
        if rtsp_url is not None:
            rtsp_url = str(rtsp_url).strip()
        elif self.gateway_rtsp_template and camera_id:
            rtsp_url = self.gateway_rtsp_template.format(id=camera_id, camera_id=camera_id)

        hls_url = raw.get("hls_url") or raw.get("hls")
        if hls_url is not None:
            hls_url = str(hls_url).strip()
        elif self.gateway_hls_template and camera_id:
            hls_url = self.gateway_hls_template.format(id=camera_id, camera_id=camera_id)

        webrtc_url = raw.get("webrtc_url") or raw.get("webrtc") or raw.get("whep_url") or raw.get("whep")
        if webrtc_url is not None:
            webrtc_url = str(webrtc_url).strip()
        elif self.gateway_webrtc_template and camera_id:
            webrtc_url = self.gateway_webrtc_template.format(id=camera_id, camera_id=camera_id)

        timezone = raw.get("timezone") or raw.get("tz")
        if timezone is not None:
            timezone = str(timezone).strip()

        # Preserve unknown fields
        handled_keys = {
            "camera_id", "id", "cam_id", "cameraId", "camera_number", "number",
            "name", "label", "title", "location", "address", "site",
            "latitude", "lat", "longitude", "lng", "lon", "long", "coordinates",
            "status", "state", "codec", "video_codec",
            "width", "w", "resolution_width", "height", "h", "resolution_height",
            "resolution", "res", "dimensions",
            "fps", "frame_rate", "framerate", "bitrate", "bit_rate",
            "rtsp_url", "rtsp", "stream_url", "hls_url", "hls",
            "webrtc_url", "webrtc", "whep_url", "whep", "timezone", "tz",
            "extra"
        }
        extra = {}
        if isinstance(raw.get("extra"), dict):
            extra.update(raw["extra"])
        for k, v in raw.items():
            if k not in handled_keys:
                extra[k] = v

        return Camera(
            camera_id=camera_id,
            name=name,
            location=location,
            latitude=latitude,
            longitude=longitude,
            status=status,
            codec=codec,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            rtsp_url=rtsp_url,
            hls_url=hls_url,
            webrtc_url=webrtc_url,
            timezone=timezone,
            extra=extra,
        )

    def parse_payload(self, payload: Union[List[Any], Dict[str, Any]]) -> List[Camera]:
        """Parse complete catalogue JSON payload into a list of Camera models."""
        camera_list = []

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            # Try finding cameras list inside a key
            if "cameras" in payload and isinstance(payload["cameras"], list):
                items = payload["cameras"]
            elif "data" in payload and isinstance(payload["data"], list):
                items = payload["data"]
            elif "items" in payload and isinstance(payload["items"], list):
                items = payload["items"]
            else:
                # Could be a dictionary of camera_id -> camera_data
                items = []
                for k, v in payload.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        if "id" not in item and "camera_id" not in item:
                            item["camera_id"] = k
                        items.append(item)
                    else:
                        items.append(payload)
                        break
        else:
            return []

        for item in items:
            if isinstance(item, dict):
                camera_list.append(self.parse_camera_dict(item))

        return camera_list
