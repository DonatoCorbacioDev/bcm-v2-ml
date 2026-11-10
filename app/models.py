# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

from sqlalchemy import Column, String, Date, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class Counterparty(Base):
    __tablename__ = "counterparties"

    id = Column(String(36), primary_key=True)
    name = Column(String(255))
    type = Column(String(10))
    organization_id = Column(String(36))


class Contract(Base):
    __tablename__ = "contracts"

    id = Column(String(36), primary_key=True)
    counterparty_id = Column(String(36), ForeignKey("counterparties.id"))
    status = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    organization_id = Column(String(36))
    manager_id = Column(String(36))

    # lazy="joined" folds the counterparty into the same SELECT as the
    # contract (a single LEFT OUTER JOIN) instead of one extra query per
    # contract -- compute_risk_scores/_propose_reminder read .counterparty.name
    # on every row, so N+1 here would mean N+1 real queries in production.
    counterparty = relationship("Counterparty", lazy="joined")


class FinancialValue(Base):
    __tablename__ = "financial_values"

    id = Column(String(36), primary_key=True)
    month_value = Column(Integer)
    year_value = Column(Integer)
    financial_amount = Column(Float)
    contract_id = Column(String(36))
    organization_id = Column(String(36))
