
from sqlalchemy import create_engine

db_uri = "mysql+pymysql://avinash:admin@localhost:3306/Chinook"
engine = create_engine(db_uri)
with engine.connect() as conn:
    print("SQLAlchemy connected!")
