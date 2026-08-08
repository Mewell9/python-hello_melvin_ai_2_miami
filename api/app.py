"""FastAPI inventory service backed by products.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = ROOT / "products.csv"
INVENTORY_MD = ROOT / "inventory.md"

app = FastAPI(title="Coffee Shop Inventory API")


class Product(BaseModel):
    id: int
    name: str
    quantity: int
    unit: str


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=0)
    unit: str = Field(..., min_length=1)


class StockUpdate(BaseModel):
    delta: int


def _ensure_csv() -> None:
    if not PRODUCTS_CSV.exists():
        PRODUCTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with PRODUCTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "quantity", "unit"])
            writer.writeheader()


def _read_products() -> List[Product]:
    _ensure_csv()
    products: List[Product] = []
    with PRODUCTS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            products.append(
                Product(
                    id=int(row["id"]),
                    name=row["name"],
                    quantity=int(row["quantity"]),
                    unit=row["unit"],
                )
            )
    return products


def _write_inventory_markdown(products: List[Product]) -> None:
    """Human-readable table view (CSV remains the source of truth for the API)."""
    lines = [
        "# Inventory",
        "",
        "Synced from `products.csv`. Open this file’s preview to see the table layout.",
        "",
        "| id | name | quantity | unit |",
        "| --- | --- | --- | --- |",
    ]
    for p in products:
        lines.append(f"| {p.id} | {p.name} | {p.quantity} | {p.unit} |")
    lines.append("")
    INVENTORY_MD.write_text("\n".join(lines), encoding="utf-8")


def _write_products(products: List[Product]) -> None:
    with PRODUCTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "quantity", "unit"])
        writer.writeheader()
        for p in products:
            writer.writerow(
                {
                    "id": p.id,
                    "name": p.name,
                    "quantity": p.quantity,
                    "unit": p.unit,
                }
            )
    _write_inventory_markdown(products)


@app.get("/inventory", response_model=List[Product])
def list_inventory() -> List[Product]:
    return _read_products()


@app.post("/inventory", response_model=Product, status_code=201)
def add_product(payload: ProductCreate) -> Product:
    name = payload.name.strip()
    unit = payload.unit.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Product name cannot be empty.")
    if not unit:
        raise HTTPException(status_code=400, detail="Unit cannot be empty.")
    if payload.quantity < 0:
        raise HTTPException(status_code=400, detail="Quantity cannot be negative.")

    products = _read_products()
    next_id = max((p.id for p in products), default=0) + 1
    product = Product(
        id=next_id,
        name=name,
        quantity=payload.quantity,
        unit=unit,
    )
    products.append(product)
    _write_products(products)
    return product


@app.get("/inventory/alerts", response_model=List[Product])
def low_stock_alerts(
    threshold: int = Query(default=10, ge=0),
) -> List[Product]:
    return [p for p in _read_products() if p.quantity < threshold]


@app.patch("/inventory/{product_id}", response_model=Product)
def update_stock(product_id: int, payload: StockUpdate) -> Product:
    products = _read_products()
    for index, product in enumerate(products):
        if product.id == product_id:
            new_quantity = product.quantity + payload.delta
            if new_quantity < 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Insufficient stock for '{product.name}'. "
                        f"Current quantity is {product.quantity}; "
                        f"requested delta is {payload.delta}."
                    ),
                )
            updated = product.model_copy(update={"quantity": new_quantity})
            products[index] = updated
            _write_products(products)
            return updated

    raise HTTPException(
        status_code=404,
        detail=f"Product with id {product_id} not found.",
    )
