# NG-SIP
Natural Gas Scheduling &amp; Imbalance Intelligence Platform
# NG-SIP: Natural Gas Scheduling & Imbalance Intelligence Platform

Natural gas desks live at the intersection of **physical constraints**, **market volatility**, and **regulatory scrutiny**. This project is an end‑to‑end, research‑backed decision‑support platform that mirrors the daily workflow of a physical natural gas trader responsible for:

- Procuring gas for power generation  
- Scheduling on interstate pipelines  
- Managing imbalances within contractual and regulatory limits  

The goal is simple: **turn noisy market, pipeline, and operational data into actionable, risk‑aware decisions**.

---

## Project overview

NG‑SIP is built as a modular platform with four core capabilities:

1. **Real‑time imbalance monitoring**  
   - Tracks scheduled vs. actual flows and storage positions  
   - Estimates imbalance exposure and simulates pipeline penalties  
   - Surfaces pipeline notices (OFOs, critical constraints, maintenance)

2. **Intraday procurement & scheduling optimizer**  
   - Ingests cash market quotes, generation forecasts, and pipeline constraints  
   - Recommends procurement and balancing actions under risk limits and authority matrices  
   - Supports “what‑if” scenarios for outages, price shocks, and constraint changes

3. **Gas burn forecasting and gas–power coupling**  
   - Forecasts gas demand from gas‑fired generation using weather, LMPs, and historical burns  
   - Quantifies uncertainty (confidence intervals) to avoid overconfidence in point forecasts  
   - Explicitly models the link between gas and power markets, as real desks must

4. **Market intelligence and counterparty analytics**  
   - Maintains structured views of counterparties, quotes, and historical behavior  
   - Generates a daily **Morning Gas Brief** summarizing prices, constraints, and weather shifts  
   - Provides auditable context to support trader judgment and client communication

This is not a toy model; it is designed as a **small but realistic slice of an actual gas trading stack**.

---

## Architecture at a glance

At a high level, the system consists of:

- **Data ingestion layer:** pulls pipeline notices, market quotes, ISO data, and weather  
- **Core analytics layer:** imbalance engine, burn forecast models, procurement optimizer  
- **Risk and compliance layer:** risk‑limit checks, authority matrix, and audit logging  
- **Presentation layer:** a web dashboard exposing trader and compliance views

For a visual overview, see:

- [`docs/architecture_diagram.mmd`](docs/architecture_diagram.mmd) — Mermaid source  
- [`docs/architecture_diagram.png`](docs/architecture_diagram.png) — Rendered diagram

---

## Key modules

- **`src/data_ingestion/`**  
  Handles external data sources: pipeline ops data, weather, market quotes, ISO/LMP data.

- **`src/models/`**  
  Houses the imbalance engine, gas burn forecasting models, and the procurement optimizer.  
  These modules are directly informed by the gas pipeline and storage optimization literature.

- **`src/risk/`**  
  Implements risk limits (by pipeline, hub, counterparty) and an authority matrix.  
  All optimization outputs flow through this layer before being surfaced to users.

- **`src/intelligence/`**  
  Structures market color, counterparty metadata, and generates the **Morning Gas Brief**.

- **`src/ui/`**  
  Provides a dashboard built with a lightweight web framework (e.g., Plotly Dash or FastAPI + frontend) that presents trader‑centric and compliance‑centric views.

---

## Research‑backed methodology

The design of NG‑SIP is grounded in the academic and practitioner literature on:

- Transient gas pipeline flow and intra‑day scheduling  
- Gas–power market coordination under uncertainty  
- Storage and trading optimization under price and operational constraints  

A detailed methodology, including model choices and references, is documented in:

- [`docs/methodology.md`](docs/methodology.md)

---

## Getting started

**Prerequisites:**

- Python 3.10+  
- PostgreSQL (or another relational database)  
- Docker (optional but recommended for reproducible deployment)

**Setup:**

```bash
git clone https://github.com/<your-handle>/NG-SIP.git
cd NG-SIP
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
