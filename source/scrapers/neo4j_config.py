import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "y+8B0fxIcrist")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
