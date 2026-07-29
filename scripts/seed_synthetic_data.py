#!/usr/bin/env python3
"""Seed the shared bcm MySQL database with synthetic contracts and financial
values, so /forecast and /risk-scores have realistic data to work with.

Standalone offline tool: not imported by app/, not run by the test suite.
Writes to `contracts`, `financial_values` and `budgets`; reads `organizations`,
`business_areas` and `financial_types` for foreign keys and never touches
`users`, `managers`, `roles`, `audit_logs`, `notifications`, `refresh_tokens`
UNLESS run with --manager-id (assigns existing contracts to that manager id,
does not create/modify the manager or user row itself).

Requires a DB connection string with write access, passed via --db-url or the
DB_URL environment variable (the app's own DB_URL is documented as read-only
by convention, so this script never hardcodes credentials - same rule as
app/config.py). Needs the Faker package (optional - falls back to a small
built-in pool of Italian-sounding company names if it isn't installed).

Usage:
    DB_URL=mysql+pymysql://<user>:<password>@localhost:3307/bcm python scripts/seed_synthetic_data.py
    python scripts/seed_synthetic_data.py --db-url mysql+pymysql://<user>:<password>@localhost:3307/bcm --reset

Run from inside the ml container (code is baked into the image, not
volume-mounted) if there is no local Python/SQLAlchemy available:
    docker cp scripts/seed_synthetic_data.py bcm-ml:/app/scripts/seed_synthetic_data.py
    docker exec bcm-ml pip install Faker==40.23.0
    docker exec bcm-ml python scripts/seed_synthetic_data.py --db-url \
        mysql+pymysql://<user>:<password>@mysql:3306/bcm --reset

Single dedicated demo organization (public demo login, safe to fully wipe
and reseed on a schedule - --wipe-org deletes ALL contracts/financial_values/
budgets for --org-id, not just SYN-* rows, so it also undoes whatever a demo
visitor created or deleted through the app itself):
    docker exec bcm-ml python scripts/seed_synthetic_data.py --db-url \
        mysql+pymysql://<user>:<password>@mysql:3306/bcm \
        --org-id 7 --manager-id 12 --wipe-org --yes
"""
import argparse
import os
import random
import re
import sys
from datetime import date, timedelta

import numpy as np
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    text,
)
from sqlalchemy.engine import create_engine

try:
    from faker import Faker

    _fake = Faker("it_IT")
except ImportError:
    _fake = None

SYN_PREFIX = "SYN-"

STATUS_WEIGHTS = [("ACTIVE", 0.60), ("EXPIRED", 0.22), ("CANCELLED", 0.08), ("DRAFT", 0.10)]

# ---------------------------------------------------------------------------
# Calibration parameters derived from ANAC (Autorità Nazionale Anticorruzione)
# open data: https://dati.anticorruzione.it/opendata/dataset
# License: CC BY 4.0 / Italian Open Data License 2.0 (IODL 2.0)
#
# The ANAC data is NOT stored here. Only the statistical parameters extracted
# by running scripts/analyze_anac.py on the locally downloaded CSV are embedded.
#
# Amount distribution: LogNormal fit on "forniture e servizi" contracts
#   (importoAggiudicazione, filtered 1k–50M EUR, N ≈ 280,000 contracts)
#   => median ≈ EUR 44,000 | P90 ≈ EUR 350,000
ANAC_LOGNORMAL_MU = 10.69
ANAC_LOGNORMAL_SIGMA = 1.52

# Monthly seasonality indices (sum = 12.0, average = 1.0)
# Italian PA shows a documented Q4 spending rush (Nov-Dec budget exhaustion),
# a spring peak (Mar-Apr new fiscal year allocations), and an August trough
# (Ferragosto). Source: ANAC annual "Relazione" reports + IFEL studies.
ANAC_MONTHLY_SEASONALITY = [
    0.76,   # January  — post-holiday, new budget not yet allocated
    0.84,   # February
    1.09,   # March    — Q1 end, new fiscal year allocations
    1.14,   # April    — spring procurement push
    1.02,   # May
    0.94,   # June
    0.74,   # July     — pre-summer slowdown
    0.46,   # August   — Ferragosto
    0.97,   # September — return from summer
    1.12,   # October  — Q4 budget spend push
    1.33,   # November — year-end acceleration
    1.35,   # December — budget exhaustion, year-end close
]  # sum = 11.76 → normalized to 12.0 inside build_financial_rows()
# ---------------------------------------------------------------------------

FALLBACK_SURNAMES = [
    "Rossi", "Bianchi", "Verdi", "Ricci", "Colombo", "Bruno", "Russo",
    "Fontana", "Costa", "Moretti", "Galli", "Conti", "Marini", "Rinaldi", "Greco",
]
FALLBACK_NOUNS = [
    "Costruzioni", "Logistica", "Consulting", "Impianti", "Servizi", "Edilizia",
    "Tecnologie", "Engineering", "Trasporti", "Software",
]
FALLBACK_SUFFIXES = ["S.r.l.", "S.p.A.", "S.n.c.", "S.a.s."]

PROJECT_WORDS = [
    "Fornitura", "Realizzazione", "Digitalizzazione", "Manutenzione", "Integrazione",
    "Sviluppo", "Ottimizzazione", "Ristrutturazione", "Adeguamento", "Rinnovo",
]

metadata = MetaData()

contracts_table = Table(
    "contracts",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("customer_name", String(255)),
    Column("contract_number", String(255)),
    Column("wbs_code", String(50)),
    Column("project_name", String(100)),
    Column("area_id", BigInteger),
    Column("manager_id", BigInteger),
    Column("start_date", Date),
    Column("end_date", Date),
    Column("status", String(20)),
    Column("workflow_stage", String(20)),
    Column("organization_id", BigInteger),
)

financial_values_table = Table(
    "financial_values",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("month_value", Integer),
    Column("year_value", Integer),
    Column("financial_amount", Float),
    Column("financial_type_id", BigInteger),
    Column("area_id", BigInteger),
    Column("contract_id", BigInteger),
    Column("organization_id", BigInteger),
)

budgets_table = Table(
    "budgets",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("business_area_id", BigInteger),
    Column("category", String(10)),
    Column("year_value", Integer),
    Column("target_amount", Float),
    Column("organization_id", BigInteger),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DB_URL"),
        help="SQLAlchemy MySQL URL with write access (or set the DB_URL environment variable). Required.",
    )
    parser.add_argument("--contracts-per-org", type=int, default=30)
    parser.add_argument("--min-months", type=int, default=12)
    parser.add_argument("--max-months", type=int, default=24)
    parser.add_argument("--outlier-ratio", type=float, default=0.1)
    parser.add_argument("--reset", action="store_true", help="Delete existing SYN-* data before seeding")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--org-id", type=int, default=None,
        help="Only seed this organization id (default: all organizations with usable reference data)",
    )
    parser.add_argument(
        "--manager-id", type=int, default=None,
        help="Assign every seeded contract to this manager id (requires --org-id). "
             "Without it, contracts are created unassigned (manager_id NULL) as before.",
    )
    parser.add_argument(
        "--wipe-org", action="store_true",
        help="DESTRUCTIVE: delete ALL contracts/financial_values/budgets for --org-id "
             "(not just SYN-* rows), then reseed. Requires --org-id. Intended for a "
             "dedicated demo organization that never holds real customer data.",
    )
    parser.add_argument(
        "--skip-budgets", action="store_true",
        help="Do not create/refresh yearly budgets per business area",
    )
    return parser.parse_args()


def random_company_name(rng):
    if _fake is not None:
        return _fake.company()
    return f"{rng.choice(FALLBACK_SURNAMES)} {rng.choice(FALLBACK_NOUNS)} {rng.choice(FALLBACK_SUFFIXES)}"


def random_project_name(rng, area_name):
    return f"{area_name} {rng.choice(PROJECT_WORDS)} {rng.randint(1, 9)}"


def random_status(rng):
    r = rng.random()
    cumulative = 0.0
    for status, weight in STATUS_WEIGHTS:
        cumulative += weight
        if r <= cumulative:
            return status
    return STATUS_WEIGHTS[-1][0]


def random_start_date(rng, today, status=None):
    if status == "DRAFT":
        # Drafts are freshly created, awaiting approval - not seasoned history.
        return today - timedelta(days=rng.randint(0, 45))
    # 80% "seasoned" contracts (13-36 months old) guarantee enough history for
    # a meaningful forecast trend; 20% "recent" ones (1-12 months) add variety.
    if rng.random() < 0.8:
        days_ago = rng.randint(13 * 30, 36 * 30)
    else:
        days_ago = rng.randint(30, 12 * 30)
    return today - timedelta(days=days_ago)


def build_end_date(rng, status, start_date, today):
    if status in ("ACTIVE", "DRAFT"):
        if status == "ACTIVE" and rng.random() < 0.15:
            return None
        return today + timedelta(days=rng.randint(30, 36 * 30))
    if status == "EXPIRED":
        max_days = max((today - start_date).days - 30, 30)
        return start_date + timedelta(days=rng.randint(30, max_days))
    # CANCELLED: can end at any point during the contract's life.
    max_days = max((today - start_date).days, 30)
    return start_date + timedelta(days=rng.randint(15, max_days + 180))


def add_months_ym(year, month, k):
    total = year * 12 + (month - 1) + k
    return total // 12, total % 12 + 1


def months_between(y1, m1, y2, m2):
    return (y2 - y1) * 12 + (m2 - m1)


def build_financial_rows(rng, np_rng, start_date, end_date, today, min_months, max_months, type_ids, base_amount, growth_rate):
    horizon_end = today
    if end_date is not None and end_date < today:
        horizon_end = end_date
    available = months_between(start_date.year, start_date.month, horizon_end.year, horizon_end.month) + 1
    available = max(available, 1)
    n_months = min(rng.randint(min_months, max_months), available)

    # Normalize ANAC seasonality coefficients so they average exactly 1.0
    _s = ANAC_MONTHLY_SEASONALITY
    _s_total = sum(_s)
    _s_norm = [v * 12.0 / _s_total for v in _s]

    rows = []
    for i in range(n_months):
        cy, cm = add_months_ym(start_date.year, start_date.month, i)
        seasonal = _s_norm[cm - 1]       # ANAC-calibrated Italian PA seasonality
        trend = (1 + growth_rate) ** i
        sales = max(500.0, base_amount * trend * seasonal * float(np_rng.normal(1.0, 0.06)))
        cost_ratio = rng.uniform(0.4, 0.7)
        costs = max(200.0, sales * cost_ratio * float(np_rng.normal(1.0, 0.05)))
        rows.append({"month_value": cm, "year_value": cy, "financial_amount": round(sales, 2), "financial_type_id": type_ids["Ricavi"]})
        rows.append({"month_value": cm, "year_value": cy, "financial_amount": round(costs, 2), "financial_type_id": type_ids["Costi"]})
    return rows


def sample_base_amount(np_rng, is_outlier):
    # LogNormal calibrated on ANAC "forniture e servizi" (CC BY 4.0)
    # Median ≈ EUR 44,000 | P90 ≈ EUR 350,000
    base = float(np_rng.lognormal(mean=ANAC_LOGNORMAL_MU, sigma=ANAC_LOGNORMAL_SIGMA))
    base = max(1_000.0, min(base, 5_000_000.0))
    if is_outlier:
        # ANAC documents two outlier patterns: "varianti in corso d'opera"
        # (contract modifications +30-150%) and under-reported micro-contracts
        base *= float(np_rng.choice([0.25, 0.40, 1.80, 2.50, 3.50]))
    return base


def fetch_reference_data(conn):
    organizations = conn.execute(text("SELECT id, name FROM organizations")).mappings().all()
    business_areas = conn.execute(text("SELECT id, name, organization_id FROM business_areas")).mappings().all()
    financial_types = conn.execute(text("SELECT id, name, organization_id FROM financial_types")).mappings().all()

    areas_by_org = {}
    for row in business_areas:
        areas_by_org.setdefault(row["organization_id"], []).append(row)

    types_by_org = {}
    for row in financial_types:
        types_by_org.setdefault(row["organization_id"], {})[row["name"]] = row["id"]

    return organizations, areas_by_org, types_by_org


def next_syn_sequence(conn, org_id):
    pattern = re.compile(rf"^{re.escape(SYN_PREFIX)}{org_id:02d}-(\d+)$")
    rows = conn.execute(
        text("SELECT contract_number FROM contracts WHERE organization_id = :oid AND contract_number LIKE :prefix"),
        {"oid": org_id, "prefix": f"{SYN_PREFIX}{org_id:02d}-%"},
    ).all()
    max_seq = 0
    for (contract_number,) in rows:
        m = pattern.match(contract_number)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1, len(rows)


def reset_org_synthetic_data(conn, org_id):
    conn.execute(
        text(
            "DELETE fv FROM financial_values fv "
            "JOIN contracts c ON fv.contract_id = c.id "
            "WHERE c.organization_id = :oid AND c.contract_number LIKE :prefix"
        ),
        {"oid": org_id, "prefix": f"{SYN_PREFIX}{org_id:02d}-%"},
    )
    result = conn.execute(
        text("DELETE FROM contracts WHERE organization_id = :oid AND contract_number LIKE :prefix"),
        {"oid": org_id, "prefix": f"{SYN_PREFIX}{org_id:02d}-%"},
    )
    return result.rowcount


def wipe_org_data(conn, org_id):
    """Delete ALL contracts/financial_values/budgets for org_id, regardless of
    naming. Only ever call this for a dedicated demo organization that holds
    no real customer data - it does not scope by SYN_PREFIX like
    reset_org_synthetic_data(), on purpose, so a nightly reset also undoes
    whatever a demo visitor created or edited through the app itself."""
    conn.execute(
        text(
            "DELETE cwe FROM contract_workflow_events cwe "
            "JOIN contracts c ON cwe.contract_id = c.id "
            "WHERE c.organization_id = :oid"
        ),
        {"oid": org_id},
    )
    conn.execute(
        text(
            "DELETE ch FROM contract_history ch "
            "JOIN contracts c ON ch.contract_id = c.id "
            "WHERE c.organization_id = :oid"
        ),
        {"oid": org_id},
    )
    conn.execute(
        text(
            "DELETE cm FROM contract_manager cm "
            "JOIN contracts c ON cm.contract_id = c.id "
            "WHERE c.organization_id = :oid"
        ),
        {"oid": org_id},
    )
    conn.execute(
        text(
            "DELETE fv FROM financial_values fv "
            "JOIN contracts c ON fv.contract_id = c.id "
            "WHERE c.organization_id = :oid"
        ),
        {"oid": org_id},
    )
    contracts_deleted = conn.execute(
        text("DELETE FROM contracts WHERE organization_id = :oid"), {"oid": org_id}
    ).rowcount
    budgets_deleted = conn.execute(
        text("DELETE FROM budgets WHERE organization_id = :oid"), {"oid": org_id}
    ).rowcount
    return contracts_deleted, budgets_deleted


def seed_budgets(conn, org, areas, type_ids, contracts_fv_by_area_category, year):
    """One REVENUE and one COST budget per business area for `year`, set to
    ~15% above whatever was actually realized that year so the Budgets page
    shows a mix of on-track and over-budget areas rather than all-green."""
    if "Ricavi" not in type_ids or "Costi" not in type_ids:
        return 0
    rows = []
    for area in areas:
        for category, type_name in (("REVENUE", "Ricavi"), ("COST", "Costi")):
            realized = contracts_fv_by_area_category.get((area["id"], type_name), 0.0)
            target = max(realized * 1.15, 10_000.0)
            rows.append({
                "business_area_id": area["id"],
                "category": category,
                "year_value": year,
                "target_amount": round(target, 2),
                "organization_id": org["id"],
            })
    if rows:
        conn.execute(insert(budgets_table), rows)
    return len(rows)


def seed_organization(conn, rng, np_rng, org, areas, type_ids, args, today, start_seq):
    type_ids_ok = "Ricavi" in type_ids and "Costi" in type_ids
    if not areas or not type_ids_ok:
        print(f"  [SKIP] org '{org['name']}' (id={org['id']}): missing business_areas or SALES/COSTS financial_types")
        return 0, 0, 0

    id_to_type_name = {type_ids["Ricavi"]: "Ricavi", type_ids["Costi"]: "Costi"}

    outlier_count = max(1, round(args.contracts_per_org * args.outlier_ratio))
    outlier_flags = [True] * outlier_count + [False] * (args.contracts_per_org - outlier_count)
    rng.shuffle(outlier_flags)

    fv_rows_all = []
    contracts_created = 0
    realized_this_year = {}  # (area_id, "Ricavi"/"Costi") -> sum for today.year

    for i in range(args.contracts_per_org):
        seq = start_seq + i
        area = rng.choice(areas)
        status = random_status(rng)
        start_date = random_start_date(rng, today, status)
        end_date = build_end_date(rng, status, start_date, today)

        contract_values = {
            "customer_name": random_company_name(rng),
            "contract_number": f"{SYN_PREFIX}{org['id']:02d}-{seq:04d}",
            "wbs_code": f"WBS-{rng.randint(1000, 9999)}" if rng.random() < 0.7 else None,
            "project_name": random_project_name(rng, area["name"]),
            "area_id": area["id"],
            "manager_id": args.manager_id,
            "start_date": start_date,
            "end_date": end_date,
            "status": status,
            "workflow_stage": "DRAFT" if status == "DRAFT" else None,
            "organization_id": org["id"],
        }
        result = conn.execute(insert(contracts_table).values(**contract_values))
        contract_id = result.inserted_primary_key[0]
        contracts_created += 1

        if status == "DRAFT":
            # Not yet approved/started - no realized financial history.
            continue

        base_amount = sample_base_amount(np_rng, outlier_flags[i])
        growth_rate = rng.uniform(-0.01, 0.025)
        rows = build_financial_rows(
            rng, np_rng, start_date, end_date, today,
            args.min_months, args.max_months, type_ids, base_amount, growth_rate,
        )
        for row in rows:
            row["area_id"] = area["id"]
            row["contract_id"] = contract_id
            row["organization_id"] = org["id"]
            if row["year_value"] == today.year:
                key = (area["id"], id_to_type_name[row["financial_type_id"]])
                realized_this_year[key] = realized_this_year.get(key, 0.0) + row["financial_amount"]
        fv_rows_all.extend(rows)

    if fv_rows_all:
        conn.execute(insert(financial_values_table), fv_rows_all)

    budgets_created = 0
    if not args.skip_budgets:
        budgets_created = seed_budgets(conn, org, areas, type_ids, realized_this_year, today.year)

    return contracts_created, len(fv_rows_all), budgets_created


def main():
    args = parse_args()
    if not args.db_url:
        print("Missing DB connection string. Pass --db-url or set the DB_URL environment variable.")
        sys.exit(1)
    if args.wipe_org and args.org_id is None:
        print("--wipe-org requires --org-id (refusing to full-wipe every organization).")
        sys.exit(1)
    if args.manager_id is not None and args.org_id is None:
        print("--manager-id requires --org-id (a manager belongs to exactly one organization).")
        sys.exit(1)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    today = date.today()

    engine = create_engine(args.db_url)

    with engine.connect() as conn:
        organizations, areas_by_org, types_by_org = fetch_reference_data(conn)

        if args.org_id is not None:
            organizations = [o for o in organizations if o["id"] == args.org_id]
            if not organizations:
                print(f"No organization with id={args.org_id} found.")
                sys.exit(1)

        if not organizations:
            print("No organizations found in the database. Run the backend (Flyway migrations) first.")
            sys.exit(1)

        plan = []
        for org in organizations:
            start_seq, existing_count = next_syn_sequence(conn, org["id"])
            if existing_count > 0 and not args.reset and not args.wipe_org and not args.yes:
                answer = input(
                    f"Org '{org['name']}' (id={org['id']}) has {existing_count} existing synthetic "
                    f"contracts. Add {args.contracts_per_org} more? [y/N] "
                ).strip().lower()
                if answer not in ("y", "yes"):
                    print(f"  Skipping org '{org['name']}'.")
                    continue
            plan.append((org, start_seq))

    if not plan:
        print("Nothing to do.")
        return

    total_contracts = 0
    total_values = 0
    total_budgets = 0
    with engine.begin() as conn:
        for org, start_seq in plan:
            if args.wipe_org:
                deleted_contracts, deleted_budgets = wipe_org_data(conn, org["id"])
                print(
                    f"Org '{org['name']}' (id={org['id']}): wiped {deleted_contracts} contracts "
                    f"and {deleted_budgets} budgets (full reset, not just synthetic rows)."
                )
                start_seq = 1
            elif args.reset:
                deleted = reset_org_synthetic_data(conn, org["id"])
                print(f"Org '{org['name']}' (id={org['id']}): removed {deleted} previous synthetic contracts.")
                start_seq = 1
                if not args.skip_budgets:
                    conn.execute(
                        text("DELETE FROM budgets WHERE organization_id = :oid AND year_value = :yr"),
                        {"oid": org["id"], "yr": today.year},
                    )
            elif not args.skip_budgets:
                # Re-running without --reset/--wipe-org would otherwise hit the
                # unique (area, category, year, org) constraint on budgets.
                conn.execute(
                    text("DELETE FROM budgets WHERE organization_id = :oid AND year_value = :yr"),
                    {"oid": org["id"], "yr": today.year},
                )

            areas = areas_by_org.get(org["id"], [])
            type_ids = types_by_org.get(org["id"], {})
            created_contracts, created_values, created_budgets = seed_organization(
                conn, rng, np_rng, org, areas, type_ids, args, today, start_seq
            )
            total_contracts += created_contracts
            total_values += created_values
            total_budgets += created_budgets
            print(
                f"Org '{org['name']}' (id={org['id']}): created {created_contracts} contracts, "
                f"{created_values} financial values, {created_budgets} budgets."
            )

    print(
        f"\nDone. Total: {total_contracts} contracts, {total_values} financial values, "
        f"{total_budgets} budgets across {len(plan)} organization(s)."
    )


if __name__ == "__main__":
    main()
