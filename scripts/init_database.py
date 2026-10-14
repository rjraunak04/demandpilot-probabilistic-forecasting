"""Initialize the DemandPilot database schema."""

from demandpilot.db.models import Base
from demandpilot.db.session import build_engine


def main() -> None:
    engine = build_engine()
    Base.metadata.create_all(engine)
    print("DemandPilot database initialized")


if __name__ == "__main__":
    main()
