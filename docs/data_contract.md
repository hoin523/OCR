# OCR Dataset Contract

This repository is public. Private screenshots, receipts, invoices, labels, and
generated OCR results must stay in ignored local folders.

## Drop Zone

Put source images here:

```text
data/private/raw/
```

Supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

Recommended names:

- `receipt_000001.png`
- `invoice_000001.png`
- `transaction_000001.png`

Korean filenames and exported messenger filenames are also accepted. The
pipeline creates a stable safe `image_id` from the original filename plus file
hash.

## Generated Files

Inventory:

```text
results/dataset_inventory.json
```

Baseline OCR outputs:

```text
results/baselines/<image_id>/paddleocr-text.json
results/baselines/<image_id>/paddleocr-vl.json
results/baselines/<image_id>/qwen3-vl.json
```

Manual labels, when ready:

```text
data/private/labels/raw_text/<image_id>.txt
data/private/labels/fields/<image_id>.json
data/private/labels/boxes/<image_id>.json
```

## Field Label Shape

Start with this JSON shape for `labels/fields/<image_id>.json`:

```json
{
  "document_type": "receipt",
  "merchant": "Sample Store",
  "invoice_no": null,
  "order_id": null,
  "datetime": "2026-05-14 17:40",
  "currency": "KRW",
  "subtotal": 14400,
  "tax": null,
  "delivery_fee": 0,
  "discount": 0,
  "total": 14400,
  "payment_method": "Sample Bank",
  "account_tail": "2628",
  "line_items": [
    {
      "name": "Sample Item",
      "quantity": null,
      "unit_price": null,
      "amount": 6800
    }
  ],
  "needs_review": false,
  "review_reasons": []
}
```
