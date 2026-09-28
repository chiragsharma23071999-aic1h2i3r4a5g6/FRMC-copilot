"""Application audit logging (separate from the Model Context Protocol SDK)."""
from models import SessionLocal, Log


class MCPController:
    def __init__(self):
        self.set_params()

    def set_params(self, temperature=0.2, max_tokens=512):
        self.temperature = temperature
        self.max_tokens = max_tokens
        return {"temperature": temperature, "max_tokens": max_tokens}

    def log_event(self, agent_name, action, details=""):
        with SessionLocal.begin() as session:
            log = Log(agent=agent_name, action=action, details=details)
            session.add(log)
        return log

    def get_logs(self):
        with SessionLocal() as session:
            return session.query(Log).order_by(Log.id).all()


mcp = MCPController()
