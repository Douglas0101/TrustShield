"""ERP-grade store inventory/balance system implemented in Python.

This module implements a Tkinter-based desktop application that coordinates a
transactional inventory counting workflow while remaining faithful to the
requirements described in the architectural specification.  The solution uses a
local SQLite database to persist inventory sessions and scan logs, and simulates
an external product master repository that exposes full-text search (FTS)
capabilities.  The code is organized around repositories and services so that
it can be reused in unit tests or alternative user interfaces.
"""
from __future__ import annotations

import datetime as dt
import logging
import pathlib
import sqlite3
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tkinter import END, Toplevel, ttk, messagebox, StringVar, Tk

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Data classes used for transferring structured information around
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """Represents a row in the product master copy."""

    internal_code: str
    barcode: str
    description: str
    unit_of_measure: str
    current_price: Decimal
    last_known_stock_qtd: Decimal


@dataclass(frozen=True)
class ScanEvent:
    """Represents a transactional scan log entry."""

    log_id: int
    session_id: int
    product_internal_code: str
    scanned_value: str
    count_unit: Decimal
    scan_timestamp: dt.datetime
    price_at_scan: Decimal
    adjustment_reason: Optional[str]


@dataclass(frozen=True)
class DivergenceRow:
    """Represents a single line of the divergence report."""

    internal_code: str
    description: str
    stock_system_qty: Decimal
    counted_qty: Decimal
    difference_qty: Decimal
    unit_price: Decimal
    difference_value: Decimal


# ---------------------------------------------------------------------------
# Repository layer: Product master (simulated external DB with FTS)
# ---------------------------------------------------------------------------


class ProductMasterRepository:
    """Repository responsible for interacting with the product master copy.

    The repository internally uses SQLite with FTS5 to emulate the behaviour of
    an enterprise database that exposes full text search capabilities.  In a
    production deployment this class would be replaced with implementations that
    connect to MySQL/PostgreSQL/etc.
    """

    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._connection.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS product_master (
                internal_code TEXT PRIMARY KEY,
                barcode TEXT,
                description TEXT NOT NULL,
                unit_of_measure TEXT NOT NULL,
                current_price NUMERIC NOT NULL,
                last_known_stock_qtd NUMERIC NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS product_master_search
            USING fts5(internal_code UNINDEXED, description, content='product_master', content_rowid='rowid');
            """
        )
        self._connection.commit()
        self._rebuild_fts_if_needed()

    def _rebuild_fts_if_needed(self) -> None:
        cur = self._connection.cursor()
        cur.execute("SELECT count(*) FROM product_master_search")
        count = cur.fetchone()[0]
        if count == 0:
            LOGGER.info("Rebuilding full-text index for product master copy")
            cur.execute("INSERT INTO product_master_search(product_master_search) VALUES('rebuild')")
            self._connection.commit()

    def upsert_product(self, product: Product) -> None:
        LOGGER.debug("Upserting product %s", product.internal_code)
        cur = self._connection.cursor()
        cur.execute(
            """
            INSERT INTO product_master (internal_code, barcode, description, unit_of_measure, current_price, last_known_stock_qtd)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(internal_code) DO UPDATE SET
                barcode=excluded.barcode,
                description=excluded.description,
                unit_of_measure=excluded.unit_of_measure,
                current_price=excluded.current_price,
                last_known_stock_qtd=excluded.last_known_stock_qtd
            """,
            (
                product.internal_code,
                product.barcode,
                product.description,
                product.unit_of_measure,
                float(product.current_price),
                float(product.last_known_stock_qtd),
            ),
        )

        cur.execute("SELECT rowid FROM product_master WHERE internal_code = ?", (product.internal_code,))
        rowid = cur.fetchone()[0]
        cur.execute("DELETE FROM product_master_search WHERE rowid = ?", (rowid,))
        cur.execute(
            """
            INSERT INTO product_master_search(rowid, internal_code, description)
            VALUES (?, ?, ?)
            """,
            (rowid, product.internal_code, product.description),
        )
        self._connection.commit()

    def get_by_internal_code(self, internal_code: str) -> Optional[Product]:
        cur = self._connection.cursor()
        cur.execute("SELECT * FROM product_master WHERE internal_code = ?", (internal_code,))
        row = cur.fetchone()
        return self._row_to_product(row) if row else None

    def get_by_barcode(self, barcode: str) -> Optional[Product]:
        cur = self._connection.cursor()
        cur.execute("SELECT * FROM product_master WHERE barcode = ?", (barcode,))
        row = cur.fetchone()
        return self._row_to_product(row) if row else None

    def search_by_description_fragment(self, fragment: str, limit: int = 20) -> List[Product]:
        """Searches products using FTS instead of LIKE queries."""
        fragment = fragment.strip()
        if not fragment:
            return []
        cur = self._connection.cursor()
        # Escape the fragment for FTS5 query syntax; we quote the token so that spaces become phrase searches
        cur.execute(
            """
            SELECT pm.*
            FROM product_master pm
            JOIN product_master_search pms ON pm.rowid = pms.rowid
            WHERE product_master_search MATCH ?
            ORDER BY bm25(product_master_search)
            LIMIT ?
            """,
            (f'"{fragment}"', limit),
        )
        rows = cur.fetchall()
        return [self._row_to_product(row) for row in rows]

    def _row_to_product(self, row: sqlite3.Row) -> Product:
        return Product(
            internal_code=row["internal_code"],
            barcode=row["barcode"],
            description=row["description"],
            unit_of_measure=row["unit_of_measure"],
            current_price=Decimal(str(row["current_price"])),
            last_known_stock_qtd=Decimal(str(row["last_known_stock_qtd"])),
        )

    def next_internal_code(self) -> str:
        cur = self._connection.cursor()
        cur.execute("SELECT internal_code FROM product_master ORDER BY CAST(internal_code AS INTEGER) DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return "1"
        try:
            return str(int(row[0]) + 1)
        except ValueError:
            # Fall back to GUID-based code when numeric increment fails.
            return uuid.uuid4().hex[:12]

    def all_products(self) -> Iterable[Product]:
        cur = self._connection.cursor()
        cur.execute("SELECT * FROM product_master")
        for row in cur.fetchall():
            yield self._row_to_product(row)


# ---------------------------------------------------------------------------
# Repository layer: Local transactional database
# ---------------------------------------------------------------------------


class InventoryRepository:
    """Manages inventory sessions and scan logs stored in SQLite."""

    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._connection.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS inventory_session (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                responsible_user_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS scan_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES inventory_session(session_id),
                product_internal_code TEXT NOT NULL,
                scanned_value TEXT NOT NULL,
                count_unit NUMERIC NOT NULL,
                scan_timestamp TEXT NOT NULL,
                price_at_scan NUMERIC NOT NULL,
                adjustment_reason TEXT
            );
            """
        )
        self._connection.commit()

    # Session management -------------------------------------------------
    def create_session(self, responsible_user_id: int) -> int:
        cur = self._connection.cursor()
        now = dt.datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO inventory_session(start_timestamp, status, responsible_user_id) VALUES (?, ?, ?)",
            (now, "Aberto", responsible_user_id),
        )
        self._connection.commit()
        session_id = int(cur.lastrowid)
        LOGGER.info("Started inventory session %s", session_id)
        return session_id

    def update_session_status(self, session_id: int, status: str) -> None:
        cur = self._connection.cursor()
        cur.execute("UPDATE inventory_session SET status = ? WHERE session_id = ?", (status, session_id))
        self._connection.commit()

    def get_session_status(self, session_id: int) -> str:
        cur = self._connection.cursor()
        cur.execute("SELECT status FROM inventory_session WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        return row[0]

    def list_sessions(self) -> List[sqlite3.Row]:
        cur = self._connection.cursor()
        cur.execute("SELECT * FROM inventory_session ORDER BY session_id DESC")
        return cur.fetchall()

    # Scan log -----------------------------------------------------------
    def append_scan_event(
        self,
        session_id: int,
        product_internal_code: str,
        scanned_value: str,
        count_unit: Decimal,
        price_at_scan: Decimal,
        adjustment_reason: Optional[str] = None,
    ) -> int:
        cur = self._connection.cursor()
        cur.execute(
            """
            INSERT INTO scan_log(session_id, product_internal_code, scanned_value, count_unit, scan_timestamp, price_at_scan, adjustment_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                product_internal_code,
                scanned_value,
                float(count_unit),
                dt.datetime.utcnow().isoformat(),
                float(price_at_scan),
                adjustment_reason,
            ),
        )
        self._connection.commit()
        log_id = int(cur.lastrowid)
        LOGGER.debug("Recorded scan event %s for product %s", log_id, product_internal_code)
        return log_id

    def aggregated_counts(self, session_id: int) -> Dict[str, Decimal]:
        cur = self._connection.cursor()
        cur.execute(
            """
            SELECT product_internal_code, SUM(count_unit) as total
            FROM scan_log
            WHERE session_id = ?
            GROUP BY product_internal_code
            """,
            (session_id,),
        )
        totals = {row["product_internal_code"]: Decimal(str(row["total"])) for row in cur.fetchall()}
        LOGGER.debug("Aggregated counts for session %s: %s", session_id, totals)
        return totals

    def fetch_scan_events(self, session_id: int) -> List[ScanEvent]:
        cur = self._connection.cursor()
        cur.execute("SELECT * FROM scan_log WHERE session_id = ? ORDER BY log_id", (session_id,))
        rows = cur.fetchall()
        return [
            ScanEvent(
                log_id=row["log_id"],
                session_id=row["session_id"],
                product_internal_code=row["product_internal_code"],
                scanned_value=row["scanned_value"],
                count_unit=Decimal(str(row["count_unit"])),
                scan_timestamp=dt.datetime.fromisoformat(row["scan_timestamp"]),
                price_at_scan=Decimal(str(row["price_at_scan"])),
                adjustment_reason=row["adjustment_reason"],
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Service layer encapsulating business rules
# ---------------------------------------------------------------------------


class InventoryService:
    """Coordinates business rules and repository access."""

    FRACTIONAL_UNITS = {"KG", "MT"}

    def __init__(self, product_repo: ProductMasterRepository, inventory_repo: InventoryRepository):
        self.product_repo = product_repo
        self.inventory_repo = inventory_repo

    # ------------------------------------------------------------------
    # Product lookup logic with FTS fallback and quick register
    # ------------------------------------------------------------------
    def locate_product(self, scanned_value: str) -> Optional[Product]:
        product = self.product_repo.get_by_internal_code(scanned_value)
        if product:
            return product
        product = self.product_repo.get_by_barcode(scanned_value)
        if product:
            return product
        # Fallback to FTS search on description for textual fragments
        matches = self.product_repo.search_by_description_fragment(scanned_value, limit=1)
        return matches[0] if matches else None

    def quick_register_product(
        self,
        description: str,
        unit_of_measure: str,
        current_price: Decimal,
        last_known_stock_qtd: Decimal = Decimal("0"),
        barcode: Optional[str] = None,
    ) -> Product:
        internal_code = self.product_repo.next_internal_code()
        product = Product(
            internal_code=internal_code,
            barcode=barcode or "",
            description=description,
            unit_of_measure=unit_of_measure,
            current_price=current_price,
            last_known_stock_qtd=last_known_stock_qtd,
        )
        self.product_repo.upsert_product(product)
        LOGGER.info("Quick registered product %s", internal_code)
        return product

    # ------------------------------------------------------------------
    # Counting logic
    # ------------------------------------------------------------------
    def validate_count_unit(self, product: Product, count_unit: Decimal) -> Decimal:
        if product.unit_of_measure in self.FRACTIONAL_UNITS:
            return count_unit
        # Ensure discrete units are rounded and non-fractional
        rounded = count_unit.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        if rounded != count_unit:
            raise ValueError("Produtos com unidade UN não aceitam quantidades fracionadas")
        return rounded

    def process_scan(
        self,
        session_id: int,
        scanned_value: str,
        count_unit: Optional[Decimal] = None,
        price_override: Optional[Decimal] = None,
        adjustment_reason: Optional[str] = None,
    ) -> Tuple[Product, Decimal]:
        if self.inventory_repo.get_session_status(session_id) != "Aberto":
            raise RuntimeError("A sessão não está aberta para contagem")

        product = self.locate_product(scanned_value)
        if not product:
            raise LookupError("Produto não encontrado; considere cadastro rápido")

        effective_count_unit = count_unit or Decimal("1")
        effective_count_unit = self.validate_count_unit(product, effective_count_unit)
        effective_price = price_override or product.current_price

        self.inventory_repo.append_scan_event(
            session_id=session_id,
            product_internal_code=product.internal_code,
            scanned_value=scanned_value,
            count_unit=effective_count_unit,
            price_at_scan=effective_price,
            adjustment_reason=adjustment_reason,
        )

        totals = self.inventory_repo.aggregated_counts(session_id)
        product_total = totals.get(product.internal_code, Decimal("0"))
        return product, product_total

    def record_negative_adjustment(
        self,
        session_id: int,
        product_internal_code: str,
        quantity: Decimal,
        reason: str,
    ) -> None:
        product = self.product_repo.get_by_internal_code(product_internal_code)
        if not product:
            raise LookupError("Produto não encontrado para ajuste")
        negative_qty = -abs(quantity)
        self.inventory_repo.append_scan_event(
            session_id=session_id,
            product_internal_code=product.internal_code,
            scanned_value=product.internal_code,
            count_unit=negative_qty,
            price_at_scan=product.current_price,
            adjustment_reason=reason,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def close_session(self, session_id: int) -> None:
        self.inventory_repo.update_session_status(session_id, "Encerrado")

    def generate_divergence_report(self, session_id: int) -> List[DivergenceRow]:
        events = self.inventory_repo.fetch_scan_events(session_id)
        if not events:
            return []

        totals: Dict[str, Decimal] = {}
        price_map: Dict[str, Decimal] = {}
        for event in events:
            totals[event.product_internal_code] = totals.get(event.product_internal_code, Decimal("0")) + event.count_unit
            price_map.setdefault(event.product_internal_code, event.price_at_scan)

        rows: List[DivergenceRow] = []
        for internal_code, counted_qty in totals.items():
            product = self.product_repo.get_by_internal_code(internal_code)
            if not product:
                LOGGER.warning("Produto %s não encontrado no master durante relatório", internal_code)
                continue
            difference = counted_qty - product.last_known_stock_qtd
            unit_price = price_map.get(internal_code, product.current_price)
            difference_value = (difference * unit_price).quantize(Decimal("0.01"))
            rows.append(
                DivergenceRow(
                    internal_code=internal_code,
                    description=product.description,
                    stock_system_qty=product.last_known_stock_qtd,
                    counted_qty=counted_qty,
                    difference_qty=difference,
                    unit_price=unit_price,
                    difference_value=difference_value,
                )
            )
        return rows

    def export_reports(self, session_id: int, output_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        divergence = self.generate_divergence_report(session_id)
        events = self.inventory_repo.fetch_scan_events(session_id)

        divergence_df = pd.DataFrame([row.__dict__ for row in divergence])
        events_df = pd.DataFrame([
            {
                "log_id": event.log_id,
                "session_id": event.session_id,
                "product_internal_code": event.product_internal_code,
                "scanned_value": event.scanned_value,
                "count_unit": float(event.count_unit),
                "scan_timestamp": event.scan_timestamp.isoformat(),
                "price_at_scan": float(event.price_at_scan),
                "adjustment_reason": event.adjustment_reason,
            }
            for event in events
        ])

        outputs: Dict[str, pathlib.Path] = {}
        divergence_csv = output_dir / f"divergence_session_{session_id}.csv"
        divergence_df.to_csv(divergence_csv, index=False)
        outputs["divergence_csv"] = divergence_csv

        scan_csv = output_dir / f"scan_log_session_{session_id}.csv"
        events_df.to_csv(scan_csv, index=False)
        outputs["scan_log_csv"] = scan_csv

        divergence_xlsx = output_dir / f"divergence_session_{session_id}.xlsx"
        with pd.ExcelWriter(divergence_xlsx, engine="openpyxl") as writer:
            divergence_df.to_excel(writer, sheet_name="Divergencia", index=False)
            events_df.to_excel(writer, sheet_name="ScanLog", index=False)
        outputs["excel"] = divergence_xlsx

        # Placeholder for PDF generation: in production use ReportLab/WeasyPrint
        pdf_placeholder = output_dir / f"divergence_session_{session_id}.pdf"
        pdf_placeholder.write_text(
            "Relatórios PDF profissionais devem ser gerados com bibliotecas como ReportLab ou WeasyPrint."
        )
        outputs["pdf_placeholder"] = pdf_placeholder

        return outputs


# ---------------------------------------------------------------------------
# Tkinter GUI application
# ---------------------------------------------------------------------------


class InventoryApp:
    """Tkinter application that interacts with the inventory service."""

    def __init__(self, root: Tk, service: InventoryService, session_id: int):
        self.root = root
        self.service = service
        self.session_id = session_id

        self.root.title("Inventário ERP Python")
        self.root.geometry("900x500")

        self.scanned_value_var = StringVar()
        self.quantity_var = StringVar(value="1")
        self.feedback_var = StringVar()

        self._build_widgets()

    def _build_widgets(self) -> None:
        mainframe = ttk.Frame(self.root, padding="12")
        mainframe.grid(row=0, column=0, sticky="NSEW")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(mainframe, text="Código/Descrição").grid(row=0, column=0, sticky="W")
        entry = ttk.Entry(mainframe, textvariable=self.scanned_value_var, width=60)
        entry.grid(row=1, column=0, sticky="EW")
        entry.bind("<Return>", self.on_scan_submit)
        entry.focus()

        ttk.Label(mainframe, text="Quantidade").grid(row=0, column=1, sticky="W")
        qty_entry = ttk.Entry(mainframe, textvariable=self.quantity_var, width=10)
        qty_entry.grid(row=1, column=1, sticky="W")

        ttk.Button(mainframe, text="Cadastrar Rápido", command=self.quick_register_dialog).grid(row=1, column=2, padx=6)

        ttk.Label(mainframe, textvariable=self.feedback_var, font=("TkDefaultFont", 10, "bold"), foreground="blue").grid(
            row=2, column=0, columnspan=3, sticky="W", pady=6
        )

        columns = ("internal_code", "description", "counted")
        self.tree = ttk.Treeview(mainframe, columns=columns, show="headings")
        for col, heading in zip(columns, ("Código", "Descrição", "Qtd Contada")):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=200 if col != "counted" else 120)
        self.tree.grid(row=3, column=0, columnspan=3, sticky="NSEW", pady=12)

        mainframe.rowconfigure(3, weight=1)
        mainframe.columnconfigure(0, weight=1)

        ttk.Button(mainframe, text="Encerrar Sessão", command=self.close_session).grid(row=4, column=0, pady=8, sticky="W")
        ttk.Button(mainframe, text="Exportar Relatórios", command=self.export_reports).grid(row=4, column=1, pady=8, sticky="W")

    # Event handlers ----------------------------------------------------
    def on_scan_submit(self, event=None) -> None:  # type: ignore[override]
        value = self.scanned_value_var.get().strip()
        quantity_text = self.quantity_var.get().strip() or "1"
        if not value:
            return
        try:
            count_unit = Decimal(quantity_text.replace(",", "."))
        except Exception:
            messagebox.showerror("Erro", "Quantidade inválida")
            return

        try:
            product, total = self.service.process_scan(self.session_id, value, count_unit=count_unit)
            self._update_tree(product, total)
            self.feedback_var.set(f"{product.description} -> total contado {total}")
        except LookupError:
            if messagebox.askyesno("Produto não encontrado", "Cadastrar produto rapidamente?"):
                self.quick_register_dialog(prefill=value)
        except Exception as exc:  # pragma: no cover - GUI feedback
            LOGGER.exception("Erro ao processar scan")
            messagebox.showerror("Erro", str(exc))
        finally:
            self.scanned_value_var.set("")
            self.quantity_var.set("1")

    def _update_tree(self, product: Product, total: Decimal) -> None:
        if self.tree.exists(product.internal_code):
            self.tree.item(product.internal_code, values=(product.internal_code, product.description, float(total)))
        else:
            self.tree.insert("", END, iid=product.internal_code, values=(product.internal_code, product.description, float(total)))

    def quick_register_dialog(self, prefill: str = "") -> None:
        dialog = Toplevel(self.root)
        dialog.title("Cadastro Rápido")
        dialog.transient(self.root)
        dialog.grab_set()

        description_var = StringVar(value=prefill)
        barcode_var = StringVar()
        uom_var = StringVar(value="UN")
        price_var = StringVar(value="0.00")

        ttk.Label(dialog, text="Descrição").grid(row=0, column=0, sticky="W")
        ttk.Entry(dialog, textvariable=description_var, width=40).grid(row=0, column=1)

        ttk.Label(dialog, text="Código de Barras").grid(row=1, column=0, sticky="W")
        ttk.Entry(dialog, textvariable=barcode_var, width=20).grid(row=1, column=1)

        ttk.Label(dialog, text="Unidade (UN/KG/MT)").grid(row=2, column=0, sticky="W")
        ttk.Entry(dialog, textvariable=uom_var, width=10).grid(row=2, column=1)

        ttk.Label(dialog, text="Preço Atual").grid(row=3, column=0, sticky="W")
        ttk.Entry(dialog, textvariable=price_var, width=10).grid(row=3, column=1)

        def submit() -> None:
            try:
                product = self.service.quick_register_product(
                    description=description_var.get(),
                    unit_of_measure=uom_var.get().upper(),
                    current_price=Decimal(price_var.get().replace(",", ".")),
                    barcode=barcode_var.get() or None,
                )
                self.service.process_scan(self.session_id, product.internal_code)
                dialog.destroy()
            except Exception as exc:  # pragma: no cover - GUI feedback
                LOGGER.exception("Falha no cadastro rápido")
                messagebox.showerror("Erro", str(exc))

        ttk.Button(dialog, text="Salvar", command=submit).grid(row=4, column=0, columnspan=2, pady=6)
        dialog.wait_window()

    def close_session(self) -> None:
        try:
            self.service.close_session(self.session_id)
            messagebox.showinfo("Sessão", "Sessão encerrada com sucesso")
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Erro", str(exc))

    def export_reports(self) -> None:
        try:
            output_dir = pathlib.Path("outputs") / f"session_{self.session_id}"
            outputs = self.service.export_reports(self.session_id, output_dir)
            messagebox.showinfo("Exportação", "Arquivos gerados:\n" + "\n".join(str(path) for path in outputs.values()))
        except Exception as exc:  # pragma: no cover - GUI feedback
            LOGGER.exception("Falha ao exportar relatórios")
            messagebox.showerror("Erro", str(exc))


# ---------------------------------------------------------------------------
# Utility functions for initial data seeding
# ---------------------------------------------------------------------------


def seed_sample_products(product_repo: ProductMasterRepository) -> None:
    if any(True for _ in product_repo.all_products()):
        return
    sample_products = [
        Product("1001", "7891000000001", "Arroz Branco Tipo 1 5kg", "UN", Decimal("25.90"), Decimal("120")),
        Product("1002", "7891000000002", "Feijão Carioca 1kg", "UN", Decimal("9.50"), Decimal("200")),
        Product("1003", "7891000000003", "Açúcar Cristal 2kg", "UN", Decimal("7.20"), Decimal("150")),
        Product("2001", "", "Queijo Prato Fatiado", "KG", Decimal("49.90"), Decimal("32.500")),
        Product("3001", "", "Tecido Algodão Metro", "MT", Decimal("29.90"), Decimal("40.000")),
    ]
    for product in sample_products:
        product_repo.upsert_product(product)
    LOGGER.info("Seeded sample products for demonstration")


# ---------------------------------------------------------------------------
# Main entry point (not executed automatically during import)
# ---------------------------------------------------------------------------


def build_application() -> InventoryApp:
    base_dir = pathlib.Path.cwd()
    product_repo = ProductMasterRepository(base_dir / "product_master.db")
    inventory_repo = InventoryRepository(base_dir / "inventory.db")
    seed_sample_products(product_repo)

    service = InventoryService(product_repo, inventory_repo)
    session_id = inventory_repo.create_session(responsible_user_id=1)

    root = Tk()
    app = InventoryApp(root, service, session_id)
    return app


def main() -> None:
    app = build_application()
    app.root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
