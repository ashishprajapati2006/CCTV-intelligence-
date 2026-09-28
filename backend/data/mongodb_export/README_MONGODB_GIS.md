# CCTV Vehicle Intelligence — MongoDB & GIS Import Guide

Generated dummy multi-camera vehicle tracking dataset ready for MongoDB & GIS visualization.

## Files Generated:
- `mongodb_cameras.json`: CCTV Cameras catalogue with GeoJSON Point coordinates `[longitude, latitude]`.
- `mongodb_vehicle_sightings.json`: Vehicle camera sightings with timestamps and speeds.
- `mongodb_vehicle_journeys.json`: Reconstructed multi-camera vehicle routes with GeoJSON LineString trajectories.

---

## 1. Quick Import via `mongoimport`

Run these commands in your terminal:

```bash
# 1. Import Cameras
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=cameras --file=mongodb_cameras.json --jsonArray

# 2. Import Sightings
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=vehicle_sightings --file=mongodb_vehicle_sightings.json --jsonArray

# 3. Import Journeys
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=vehicle_journeys --file=mongodb_vehicle_journeys.json --jsonArray
```

---

## 2. Essential MongoDB Geospatial Indexes (2dsphere)

Open `mongosh` and create 2dsphere spatial indexes:

```javascript
use cctv_gis;

// Index on camera locations
db.cameras.createIndex({ "location": "2dsphere" });

// Index on vehicle sightings
db.vehicle_sightings.createIndex({ "location": "2dsphere" });
db.vehicle_sightings.createIndex({ "registration_number": 1, "timestamp": 1 });

// Index on multi-camera journey paths
db.vehicle_journeys.createIndex({ "trajectory": "2dsphere" });
db.vehicle_journeys.createIndex({ "registration_number": 1 });
```

---

## 3. Useful GIS Spatial Queries in MongoDB

### Find all cameras within 2 km of a coordinate:
```javascript
db.cameras.find({
  location: {
    $near: {
      $geometry: { type: "Point", coordinates: [72.5620, 23.0286] },
      $maxDistance: 2000 // meters
    }
  }
});
```

### Trace complete journey of a vehicle by registration number:
```javascript
db.vehicle_journeys.findOne({ registration_number: "GJ01AB1234" });
```
