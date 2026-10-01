# NeuroCalc v3 Testing

After activating the virtual environment:

```powershell
pip install -r requirements.txt
python train_model.py
python app.py
```

Then test these in the browser:

- `18% of 2500` → `450`
- `solve 2x + 5 = 15` → `x = 5`
- `derivative of x^3 + 2*x`
- `integrate x^2 from 0 to 2` → `8/3`
- `limit of sin(x)/x as x approaches 0` → `1`
- `determinant of [[1,2],[3,4]]` → `-2`
- `inverse of [[1,2],[3,4]]`
- `mean of 10, 20, 30, 40` → `25`
- `monthly EMI for 500000 at 8% for 5 years`
- `convert 10 km to miles`
- `sin 30 degrees` → `0.5`
- `sqrt 144` → `12`
- `log10 1000` → `3`
- `5 factorial` → `120`

The system intentionally uses deterministic math for answers and ML primarily for intent/language routing.
