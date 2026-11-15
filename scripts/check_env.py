"""Verify the development environment: versions plus a quick smoke test of each core library."""

import importlib
import platform
import sys
import time

LIBRARIES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("requests", "requests"),
    ("beautifulsoup4", "bs4"),
    ("lxml", "lxml"),
]


def print_versions() -> bool:
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform:     {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print()

    ok = True
    for dist_name, module_name in LIBRARIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is None and module_name == "lxml":
                from lxml import etree

                version = ".".join(map(str, etree.LXML_VERSION))
            print(f"  {dist_name:<16} {version}")
        # OSError covers native libs that fail to load; report every failure, not just the first
        except (ImportError, OSError) as exc:
            ok = False
            print(f"  {dist_name:<16} IMPORT FAILED: {exc!r}")
    print()
    return ok


def smoke_models() -> None:
    import lightgbm as lgb
    import numpy as np
    import shap
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    start = time.perf_counter()
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, n_jobs=2, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)
    print(f"  XGBoost  accuracy: {xgb_acc:.3f}  ({time.perf_counter() - start:.2f}s)")

    start = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(n_estimators=50, num_leaves=15, n_jobs=2, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_acc = lgb_model.score(X_test, y_test)
    print(f"  LightGBM accuracy: {lgb_acc:.3f}  ({time.perf_counter() - start:.2f}s)")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:1])
    # SHAP values plus the base value should reconstruct the model's raw margin output
    margin = xgb_model.predict(X_test[:1], output_margin=True)[0]
    reconstructed = float(explainer.expected_value) + float(np.sum(shap_values[0]))
    assert abs(margin - reconstructed) < 1e-3, (
        f"SHAP additivity check failed: {margin} vs {reconstructed}"
    )
    top = int(np.argmax(np.abs(shap_values[0])))
    print(f"  SHAP     shape {shap_values.shape}, top feature f{top}, additivity OK")

    assert xgb_acc > 0.7 and lgb_acc > 0.7, "Model accuracy unexpectedly low"


def smoke_html() -> None:
    from bs4 import BeautifulSoup

    html = """
    <table class="fights">
      <tr><th>Red</th><th>Blue</th><th>Method</th></tr>
      <tr><td>Fighter A</td><td>Fighter B</td><td>KO/TKO</td></tr>
      <tr><td>Fighter C</td><td>Fighter D</td><td>Decision</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "lxml")
    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in soup.select("table.fights tr")[1:]
    ]
    assert rows == [
        ["Fighter A", "Fighter B", "KO/TKO"],
        ["Fighter C", "Fighter D", "Decision"],
    ], rows
    print(f"  BeautifulSoup+lxml parsed {len(rows)} rows: {rows}")


def main() -> int:
    print("=== Environment ===")
    if not print_versions():
        print("One or more imports failed; skipping smoke tests.")
        return 1

    print("=== Smoke tests ===")
    smoke_models()
    smoke_html()
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
