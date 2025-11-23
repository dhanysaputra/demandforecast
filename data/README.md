# Data Directory

This folder contains **sample datasets** used by the demand forecasting and inventory simulation pipeline.  
Real company data is **not included** for confidentiality reasons.

The sample CSV files here illustrate the **expected schema** for the forecasting engine.

---

## Files

### `sample_sales.csv`
Represents historical **customer sales orders** grouped by delivery date.

Expected columns:

| Column Name        | Description                               |
|--------------------|-------------------------------------------|
| `DeliveryDate`     | Date when item was delivered to customer   |
| `DeliveredQuantity`| Quantity delivered on that date            |

Example rows:
```bash
DeliveryDate,DeliveredQuantity
2024-01-31,12
2024-02-29,73
2024-03-31,35
```

---

### `sample_purchase.csv`
Represents **purchase order rows** received from suppliers.

Expected columns:

| Column Name        | Description                                            |
|--------------------|--------------------------------------------------------|
| `DeliveryDate`     | Date when PO row was delivered                         |
| `DeliveredQuantity`| Quantity received                                      |
| `IsConfirmed`      | Whether the PO row was confirmed                       |
| `RestQuantity`     | Remaining quantity (0 means fully received)            |
| `OrderedQuantity`  | PO quantity originally ordered                         |

Example rows:
DeliveryDate,DeliveredQuantity,IsConfirmed,RestQuantity,OrderedQuantity
2024-01-31,20,true,0,20
2024-02-29,40,true,0,40
2024-03-31,20,true,0,20

---

## Purpose of Sample Data

These sample datasets allow:

- Running the **entire forecasting pipeline**
- Demonstrating **XGBoost + Croston hybrid forecasting**
- Testing **true demand reconstruction**
- Running **inventory simulation + reorder point** logic
- Verifying **CI/CD test runs** on GitHub Actions

No real company data is stored.  
These files are **synthetic** but mimic the structure of real ERP exports.

---

## Data Privacy

If using real data locally:

- Place them in this folder (e.g., `sales_2024.csv`)
- DO NOT commit them to GitHub  
- Add real data filenames to `.gitignore`

Example:
```bash
data/real
*.xlsx
```

---

## Related Code

The forecasting pipeline reads data through:
```bash
src/data_loader.py
```

To use your own data, modify:

```python
load_data(sales_path="data/my_sales.csv", purchase_path="data/my_purchase.csv")
```
