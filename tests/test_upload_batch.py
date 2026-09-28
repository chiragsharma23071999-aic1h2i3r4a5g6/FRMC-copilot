import io
import tempfile
import unittest
import struct
from pathlib import Path
from unittest.mock import patch

import docx
import pandas as pd
from PyPDF2 import PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject, DecodedStreamObject

import app
import rag_pipeline
from document_policy import check_finance_content


def pdf_bytes(text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output


def xls_bytes():
    # Minimal BIFF2 worksheet, decoded by the installed xlrd reader.
    def record(code, data):
        return struct.pack("<HH", code, len(data)) + data
    data = record(0x0009, struct.pack("<HH", 2, 0x0010))
    data += record(0x0042, struct.pack("<H", 1252))
    for row, values in enumerate((("invoice", "payment"), ("INV003", "300"))):
        for col, value in enumerate(values):
            encoded = value.encode("ascii")
            data += record(0x0004, struct.pack("<HH3sB", row, col, bytes(3), len(encoded)) + encoded)
    return io.BytesIO(data + record(0x000A, b""))


class BatchUploadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        config = patch.dict(app.app.config, UPLOAD_FOLDER=str(self.folder), TESTING=True)
        config.start()
        self.addCleanup(config.stop)
        self.client = app.app.test_client()

    def test_mixed_formats_in_one_request(self):
        word = io.BytesIO()
        document = docx.Document()
        document.add_paragraph("SOX financial reporting controls and audit evidence.")
        document.save(word)
        word.seek(0)
        excel = io.BytesIO()
        pd.DataFrame({"invoice": ["INV001"], "payment": [100], "approval": [True]}).to_excel(excel, index=False)
        excel.seek(0)
        files = [
            (word, "controls.docx"),
            (io.BytesIO(b"FRMC financial risk management and compliance policy."), "policy.txt"),
            (excel, "invoices.xlsx"),
            (xls_bytes(), "legacy.xls"),
            (io.BytesIO(b"invoice,payment,approval\nINV002,200,true"), "payments.csv"),
            (pdf_bytes("Financial audit controls and SOX policy."), "audit.pdf"),
        ]
        with patch("app.build_vectorstore") as build:
            result = self.client.post("/upload", data={"file": files})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(len(result.json["files"]), 6)
        build.assert_called_once()
        self.assertEqual(len(build.call_args.args[0]), 6)
        self.assertEqual(len(list(self.folder.iterdir())), 6)

    def test_nonfinance_rejects_whole_batch_without_embedding(self):
        with patch("app.build_vectorstore") as build:
            result = self.client.post("/upload", data={"files": [
                (io.BytesIO(b"SOX financial controls policy"), "valid.txt"),
                (io.BytesIO(b"Recipe: mix flour and water. Bake bread."), "financial_report.txt"),
            ]})
        self.assertEqual(result.status_code, 400)
        self.assertIn("financial_report.txt", result.json["rejected"][0]["filename"])
        build.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_duplicate_in_batch_and_unsupported_extension(self):
        for names in (["a.txt", "a.TXT"], ["a.txt", "b.pdff"]):
            with patch("app.build_vectorstore") as build:
                response = self.client.post("/upload", data={"file": [
                    (io.BytesIO(b"Financial audit controls"), name) for name in names
                ]})
            self.assertIn(response.status_code, (400, 409))
            build.assert_not_called()
            self.assertEqual(list(self.folder.iterdir()), [])

    def test_index_failure_cleans_entire_batch(self):
        with patch("app.build_vectorstore", side_effect=rag_pipeline.KnowledgeBaseError("Unavailable")):
            response = self.client.post("/upload", data={"file": [
                (io.BytesIO(b"Financial audit controls"), name) for name in ("a.txt", "b.txt")
            ]})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_rebuild_rejects_nonfinance_legacy_document(self):
        (self.folder / "recipe.txt").write_text("Mix flour and bake bread.")
        with patch("app.build_vectorstore") as build:
            response = self.client.post("/build_rag")
        self.assertEqual(response.status_code, 400)
        build.assert_not_called()
        self.assertTrue((self.folder / "recipe.txt").exists())

    def test_batch_limits(self):
        with patch.dict(app.app.config, MAX_UPLOAD_FILES=20):
            response = self.client.post("/upload", data={"file": [
                (io.BytesIO(b"SOX policy"), f"{i}.txt") for i in range(21)
            ]})
        self.assertEqual(response.status_code, 400)
        with patch.dict(app.app.config, MAX_CONTENT_LENGTH=100):
            response = self.client.post("/upload", data={"file": (io.BytesIO(b"x" * 200), "a.txt")})
        self.assertEqual(response.status_code, 413)

    def test_folder_upload_and_remove_then_replace(self):
        with patch("app.build_vectorstore"), patch("app.remove_document_from_index") as remove:
            response = self.client.post("/upload", data={"files": [
                (io.BytesIO(b"Financial audit controls"), "Finance/nested/policy.txt"),
                (io.BytesIO(b"Invoice payment approval"), "Finance/payments.txt"),
            ]})
            self.assertEqual(response.status_code, 200, response.json)
            names = [item["filename"] for item in self.client.get("/documents").json["files"]]
            self.assertEqual(names, ["Finance_nested_policy.txt", "Finance_payments.txt"])
            response = self.client.delete("/documents", json={"filename": names[0]})
            self.assertEqual(response.status_code, 200)
            remove.assert_called_once_with(names[0])
            self.assertFalse((self.folder / names[0]).exists())
            self.assertTrue((self.folder / names[1]).exists())
            response = self.client.post("/upload", data={"file":
                (io.BytesIO(b"Updated financial audit controls"), names[0])})
            self.assertEqual(response.status_code, 200)

    def test_removal_failure_restores_file_and_invalid_paths_rejected(self):
        target = self.folder / "policy.txt"
        target.write_text("Financial audit controls")
        with patch("app.remove_document_from_index", side_effect=rag_pipeline.KnowledgeBaseError("Unavailable")):
            response = self.client.delete("/documents", json={"filename": "policy.txt"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(target.read_text(), "Financial audit controls")
        for name in ("../policy.txt", "..\\policy.txt", "C:\\policy.txt"):
            self.assertEqual(self.client.delete("/documents", json={"filename": name}).status_code, 400)
        self.assertEqual(self.client.delete("/documents", json={"filename": "missing.txt"}).status_code, 404)

    def test_per_file_size_limit_cleans_batch(self):
        with patch.dict(app.app.config, MAX_FILE_SIZE=30), patch("app.build_vectorstore") as build:
            response = self.client.post("/upload", data={"files": [
                (io.BytesIO(b"Financial audit controls"), "small.txt"),
                (io.BytesIO(b"Financial audit controls" * 2), "large.txt"),
            ]})
        self.assertEqual(response.status_code, 413)
        build.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_content_screen(self):
        for text in ("Financial audit controls", "SOX compliance policy", "COSO monitoring controls",
                     "invoice payment approval", "revenue expenses", "FRMC risk assessment"):
            self.assertTrue(check_finance_content(text)["accepted"], text)
        for text in ("Recipe for bread", "Risk of infection. Doctor approval required.",
                     "Approval", "A story about a river bank.", ("Bake bread with flour. " * 200) + "SOX policy"):
            self.assertFalse(check_finance_content(text)["accepted"], text)


if __name__ == "__main__":
    unittest.main()
