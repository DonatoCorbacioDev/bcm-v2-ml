from sqlalchemy import Column, BigInteger, String, Date, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class Counterparty(Base):
    __tablename__ = "counterparties"

    id = Column(BigInteger, primary_key=True)
    name = Column(String(255))
    type = Column(String(10))
    organization_id = Column(BigInteger)


class Contract(Base):
    __tablename__ = "contracts"

    id = Column(BigInteger, primary_key=True)
    counterparty_id = Column(BigInteger, ForeignKey("counterparties.id"))
    status = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    organization_id = Column(BigInteger)
    manager_id = Column(BigInteger)

    # lazy="joined" folds the counterparty into the same SELECT as the
    # contract (a single LEFT OUTER JOIN) instead of one extra query per
    # contract -- compute_risk_scores/_propose_reminder read .counterparty.name
    # on every row, so N+1 here would mean N+1 real queries in production.
    counterparty = relationship("Counterparty", lazy="joined")


class FinancialValue(Base):
    __tablename__ = "financial_values"

    id = Column(BigInteger, primary_key=True)
    month_value = Column(Integer)
    year_value = Column(Integer)
    financial_amount = Column(Float)
    contract_id = Column(BigInteger)
    organization_id = Column(BigInteger)
