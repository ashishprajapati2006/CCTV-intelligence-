# Synthetic Demonstration Watchlist Metadata

## Safety & Data Integrity Disclaimer

> [!IMPORTANT]
> **ALL DATA IN THIS DIRECTORY IS SYNTHETIC DEMONSTRATION DATA.**
>
> 1. None of the registration numbers, descriptions, or watchlist entries represent actual police lookout notices, stolen vehicles, wanted persons, or ongoing criminal investigations.
> 2. This dataset was constructed strictly for testing and demonstrating automated ANPR watchlist matching and alerting logic in Step 9 of the Gujarat Police Innovation Hackathon 2026 platform.
> 3. No connection to or data extraction from government databases (VAHAN, SARTHI, eGujCop, AFIS, NAFIS) was performed or attempted.
> 4. All categories carry the explicit prefix `DEMO_` (e.g. `DEMO_STOLEN_VEHICLE`) to ensure complete transparency during public evaluations.

---

## Watchlist Categories

| Category | Description |
|---|---|
| `DEMO_STOLEN_VEHICLE` | Simulated vehicle reported stolen for testing immediate high-priority ANPR hit alerting. |
| `DEMO_BLACKLISTED_VEHICLE` | Simulated vehicle banned from specific urban zones or security perimeters. |
| `DEMO_VEHICLE_OF_INTEREST` | Simulated vehicle linked to an ongoing investigation requiring silent monitoring and route logging. |
| `DEMO_MISSING_VEHICLE` | Simulated missing or abandoned vehicle lookup. |

---

## Priority Levels

- **`CRITICAL`**: Immediate dispatcher alert triggering audible/visual emergency banners.
- **`HIGH`**: Real-time notification dispatched to patrol monitoring dashboards.
- **`MEDIUM`**: Logged with high priority for supervisor review.
- **`LOW`**: Routine monitoring log.

---

## Lifecycle Statuses

- **`ACTIVE`**: Eligible for real-time ANPR matching.
- **`INACTIVE`**: Retained in database for audit, but ignored by matching engine.
- **`EXPIRED`**: Expired lookout notice, not matched against live streams.
