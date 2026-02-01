# Bayesian Longitudinal Template

This template provides a Bayesian longitudinal state-space model (PyMC) and an interactive viewer (HTML) with a draggable credible interval slider.

## Files
- index.html: Interactive viewer for posterior draws
- fit_model.py: PyMC model runner (NUTS/HMC) that generates posterior_draws.json
- requirements.txt: Python dependencies
- example_data.csv: Example longitudinal input data
- example_posterior.json: Example posterior draws for quick demo

## 1) Install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run the model
```powershell
python fit_model.py --input example_data.csv --output posterior_draws.json
```

You can also point to a folder of CSVs:
```powershell
python fit_model.py --input C:\path\to\folder --output posterior_draws.json
```

Column names can be overridden:
```powershell
python fit_model.py --input data.csv --id subject_id --group condition --time week --y value
```

If R-hat > 1.01 or ESS < 200, the script stops with a message. Increase `--tune`, `--draws`, or `--target-accept`.

## 3) Open the viewer
Open `index.html` in a browser. Click **Load Posterior Draws** and choose the generated `posterior_draws.json`.

The CI slider updates the time-course band and difference CI lines without re-running MCMC.
The viewer also includes Download SVG/PNG buttons for both plots.

## Input schema
Required columns (names are configurable):
- id: subject identifier
- group: condition / treatment
- time: ordered integer (e.g., week/day)
- y: positive measurement (y <= 0 is rejected)

Missing time points are allowed.

Optional metadata:
- You can add `"control_group": "your-control-name"` under the JSON `meta` to preselect the control group in the viewer.
