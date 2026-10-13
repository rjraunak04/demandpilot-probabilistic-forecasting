from demandpilot.db.models import Base
from demandpilot.db.session import build_engine
if __name__=="__main__":
    engine=build_engine(); Base.metadata.create_all(engine); print("DemandPilot database initialized")
