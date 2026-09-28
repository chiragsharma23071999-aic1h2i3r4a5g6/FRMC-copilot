from models import SessionLocal, Report


class ReportService:
    def create_report(self, title, content, author="System"):
        for name, value in (("title", title), ("content", content), ("author", author)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        with SessionLocal.begin() as session:
            report = Report(title=title.strip(), content=content, author=author.strip())
            session.add(report)
        return report

    def get_reports(self):
        with SessionLocal() as session:
            return session.query(Report).order_by(Report.id).all()

    def export_report(self, report_id):
        with SessionLocal() as session:
            return session.get(Report, report_id)


report_service = ReportService()
