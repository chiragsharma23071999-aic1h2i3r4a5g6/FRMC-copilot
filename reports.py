"""Compatibility entry point for report generation."""
from backend import ReportService


class ReportGenerator(ReportService):
    pass


report_generator = ReportGenerator()
