-- Gujarat Police Unified CCTV Intelligence Platform
-- Step 13: PostgreSQL / PostGIS Spatial Database Schema

CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Cameras with Geographic Coordinates
CREATE TABLE IF NOT EXISTS cameras (
    camera_id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom GEOMETRY(Point, 4326),
    status VARCHAR(32) DEFAULT "active",
    codec VARCHAR(16) DEFAULT "H264",
    rtsp_url TEXT,
    hls_url TEXT,
    webrtc_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cameras_geom ON cameras USING GIST(geom);

-- 2. Unique Observed Vehicles
CREATE TABLE IF NOT EXISTS observed_vehicles (
    vehicle_id VARCHAR(64) PRIMARY KEY,
    registration_number VARCHAR(32) NOT NULL,
    normalized_registration VARCHAR(32) NOT NULL UNIQUE,
    camera_count INT DEFAULT 1,
    observation_count INT DEFAULT 1,
    track_count INT DEFAULT 1,
    best_consensus_score DOUBLE PRECISION,
    average_consensus_score DOUBLE PRECISION,
    status VARCHAR(32) DEFAULT "OBSERVED",
    first_seen_camera VARCHAR(32) REFERENCES cameras(camera_id),
    first_seen_pts_ms DOUBLE PRECISION,
    last_seen_camera VARCHAR(32) REFERENCES cameras(camera_id),
    last_seen_pts_ms DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_observed_norm_reg ON observed_vehicles(normalized_registration);

-- 3. Vehicle Observations (ANPR Sightings with Spatial Anchor)
CREATE TABLE IF NOT EXISTS vehicle_observations (
    observation_id VARCHAR(64) PRIMARY KEY,
    vehicle_id VARCHAR(64) REFERENCES observed_vehicles(vehicle_id),
    camera_id VARCHAR(32) REFERENCES cameras(camera_id),
    track_id VARCHAR(32) NOT NULL,
    registration_number VARCHAR(32) NOT NULL,
    normalized_registration VARCHAR(32) NOT NULL,
    consensus_score DOUBLE PRECISION,
    ocr_confidence DOUBLE PRECISION,
    recognition_status VARCHAR(32) NOT NULL,
    first_seen_pts_ms DOUBLE PRECISION,
    recognition_pts_ms DOUBLE PRECISION,
    last_seen_pts_ms DOUBLE PRECISION,
    source_time TIMESTAMPTZ,
    source_time_status VARCHAR(32) DEFAULT "NOT_RESOLVED",
    geom GEOMETRY(Point, 4326),
    evidence_image_path TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_obs_camera ON vehicle_observations(camera_id);
CREATE INDEX IF NOT EXISTS idx_obs_norm_reg ON vehicle_observations(normalized_registration);
CREATE INDEX IF NOT EXISTS idx_obs_geom ON vehicle_observations USING GIST(geom);

-- 4. Vehicle Tracks
CREATE TABLE IF NOT EXISTS vehicle_tracks (
    track_id VARCHAR(64) PRIMARY KEY,
    camera_id VARCHAR(32) REFERENCES cameras(camera_id),
    first_seen_pts_ms DOUBLE PRECISION,
    last_seen_pts_ms DOUBLE PRECISION,
    frame_count INT DEFAULT 1,
    best_frame_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Synthetic Watchlist
CREATE TABLE IF NOT EXISTS watchlist_vehicles (
    watchlist_id VARCHAR(64) PRIMARY KEY,
    registration_number VARCHAR(32) NOT NULL,
    normalized_registration VARCHAR(32) NOT NULL UNIQUE,
    category VARCHAR(64) NOT NULL,
    priority VARCHAR(16) NOT NULL,
    reason TEXT,
    status VARCHAR(16) DEFAULT "ACTIVE",
    synthetic BOOLEAN DEFAULT TRUE,
    source VARCHAR(32) DEFAULT "SYNTHETIC_DEMO",
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_watchlist_norm_reg ON watchlist_vehicles(normalized_registration);

-- 6. Watchlist Match Alerts
CREATE TABLE IF NOT EXISTS watchlist_matches (
    match_id VARCHAR(64) PRIMARY KEY,
    observation_id VARCHAR(64) REFERENCES vehicle_observations(observation_id),
    watchlist_id VARCHAR(64) REFERENCES watchlist_vehicles(watchlist_id),
    camera_id VARCHAR(32) REFERENCES cameras(camera_id),
    registration_number VARCHAR(32) NOT NULL,
    match_type VARCHAR(32) NOT NULL,
    priority VARCHAR(16) NOT NULL,
    alert_ready BOOLEAN DEFAULT TRUE,
    matched_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_matches_camera ON watchlist_matches(camera_id);

-- 7. Vehicle Journeys (Reconstructed Spatial Trajectories)
CREATE TABLE IF NOT EXISTS vehicle_journeys (
    journey_id VARCHAR(64) PRIMARY KEY,
    vehicle_id VARCHAR(64) REFERENCES observed_vehicles(vehicle_id),
    normalized_registration VARCHAR(32) NOT NULL,
    segment_count INT DEFAULT 1,
    camera_count INT DEFAULT 1,
    trajectory GEOMETRY(LineString, 4326),
    ordering_mode VARCHAR(32) DEFAULT "CAMERA_LOCAL",
    confidence_score DOUBLE PRECISION,
    report_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_journeys_geom ON vehicle_journeys USING GIST(trajectory);
