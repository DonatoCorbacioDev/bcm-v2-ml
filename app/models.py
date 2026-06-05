from sqlalchemy import Column, BigInteger, String, Date, Float, Integer
from .database import Base


class Contract(Base):
    __tablename__ = "contracts"

    id = Column(BigInteger, primary_key=True)
    customer_name = Column(String(255))
    status = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    organization_id = Column(BigInteger)


class FinancialValue(Base):
    __tablename__ = "financial_values"

    id = Column(BigInteger, primary_key=True)
    month_value = Column(Integer)
    year_value = Column(Integer)
    financial_amount = Column(Float)
    contract_id = Column(BigInteger)
    organization_id = Column(BigInteger)
