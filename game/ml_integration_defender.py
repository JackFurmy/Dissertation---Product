

from __future__ import annotations
import os, sys, re, tempfile, joblib, requests, numpy as np, pandas as pd
from pathlib import Path
from typing  import Dict, Any, Iterable
from river   import tree, linear_model, naive_bayes, optim, metrics as river_metrics

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel)

DEFENDER_SCENARIOS: Dict[str, Dict[str, Any]] = { 
    "iot/fridge": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FFridge%2Fval_starting.csv?alt=media&token",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FFridge%2Fpart2.csv?alt=media&token"
        ],
        "difficulty": 1.0
    },
    "iot/gps_tracker": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FGPS%20Tracker%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FGPS%20Tracker%2Fpart2.csv?alt=media&token"
        ],
        "difficulty": 1.0
    },
    "iot/garage_door": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FGarage%20Door%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FGarage%20Door%2Fpart2.csv?alt=media&token=",
        ],
        "difficulty": 1.0
    },
    "iot/modbus": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FModbus%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FModbus%2Fpart2.csv?alt=media&token=",
        ],
        "difficulty": 1.0
    },
    "iot/motion_light": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FMotion%20Light%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FMotion%20Light%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "iot/thermostat": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FThermostat%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FThermostat%2Fpart2.csv?alt=media&token=
        ],
        "difficulty": 1.0
    },
    "iot/weather": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FWeather%2Fval_starting.csv?alt=media&token=,
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FIoT%2FWeather%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "linux/process": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FProcess%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FProcess%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "linux/disk": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FDisk%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FDisk%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "linux/memory": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FMemory%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FLinux%2FMemory%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "phishing/url_set": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FPhishing%2FPhishing%20URL%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FPhishing%2FPhishing%20URL%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "phishing/websites": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FPhishing%2FPhishing%20Websites%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FPhishing%2FPhishing%20Websites%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "windows/windows7": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FWindows%2FWindows%207%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FWindows%2FWindows%207%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    },
    "windows/windows10": {
        "val": "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FWindows%2FWindows%2010%2Fval_starting.csv?alt=media&token=",
        "chunks": [
            "https://firebasestorage.googleapis.com/v0/b/mldiss.firebasestorage.app/o/Incremental%2FWindows%2FWindows%2010%2Fpart2.csv?alt=media&token="
        ],
        "difficulty": 1.0
    }
}

USER_DATA_DIR  = Path(os.getenv("LOCALAPPDATA", Path.home())) / ".mldiss_models"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

INCREMENTAL_CONFIG: Dict[str, Dict[str, Any]] = {
    name: {
        "start_url"   : cfg["val"],
        "model_dir"   : USER_DATA_DIR,
        "model_names" : ["HoeffdingTree", "LogisticRegression", "NaiveBayes"],
        "difficulty"  : cfg["difficulty"],
    }
    for name, cfg in DEFENDER_SCENARIOS.items()
}

class IncrementalModelManager:
    _THRESH = 0.05                      

    def __init__(self, scenario: str, user_id: str = "default"):
        if scenario not in INCREMENTAL_CONFIG:
            raise ValueError(f"Unknown defender scenario '{scenario}'")
        self.scenario = scenario
        self.cfg      = INCREMENTAL_CONFIG[scenario]
        self.uid      = user_id
        self.dir      = self.cfg["model_dir"]; self.dir.mkdir(exist_ok=True)

        self._feats: set[str] = set()
        self.models           = self._load_models()

    def predict_one(self, x: Dict[str, Any]) -> int:
        proba = self.models["LogisticRegression"].predict_proba_one(self._unify(x))
        return int(proba.get(1, 0.0) >= self._THRESH)

    def update_one(self, x: Dict[str, Any], y_raw) -> None:
        y = self._safe_label(y_raw)
        if y is None:
            return
        x = self._unify(x)
        for _ in range(5 if y == 1 else 1):
            for m in self.models.values():
                m.learn_one(x, y)

    def score_many(self,
                   X_iter: Iterable[Dict[str, Any]],
                   y_iter: Iterable[int]) -> float:
        f1 = river_metrics.F1()
        lr = self.models["LogisticRegression"]
        for x_raw, y_raw in zip(X_iter, y_iter):
            y = self._safe_label(y_raw)
            if y is None:
                continue
            p = int(lr.predict_proba_one(self._unify(x_raw)).get(1, 0.0) >= self._THRESH)
            f1.update(y, p)
        return f1.get()

    def save(self) -> None:
        safe  = self._scenario_safe()
        pat   = re.compile(rf"(user_{re.escape(self.uid)}_.*|{safe})\.pkl$", re.I)
        removed = 0
        for pkl in self.dir.glob("*.pkl"):
            if pat.search(pkl.name):
                try: pkl.unlink(); removed += 1
                except OSError: pass
        print(f"[INFO] deleted {removed} cached model file(s)" if removed
              else "[INFO] no cached models to delete")

    def _scenario_safe(self) -> str:      return self.scenario.replace("/", "_")
    def _path(self, name: str) -> Path:   return self.dir / f"user_{self.uid}_{name}.pkl"

    def _load_models(self) -> Dict[str, Any]:
        bundle = self._locate_bundle()
        if bundle:
            data = joblib.load(bundle)
            if isinstance(data, dict) and "models" in data:
                print(f"[INFO] loaded model bundle ⇒ {bundle.name}")
                return data["models"]

        models, cold_start = {}, True
        for name in self.cfg["model_names"]:
            pkl = self._path(name)
            if pkl.exists():
                models[name] = joblib.load(pkl); cold_start = False
            else:
                models[name] = (
                    tree.HoeffdingTreeClassifier(grace_period=50, delta=1e-7)
                    if name == "HoeffdingTree" else
                    linear_model.LogisticRegression(optimizer=optim.SGD(0.05), l2=0.0)
                    if name == "LogisticRegression" else
                    naive_bayes.GaussianNB()
                )
        if cold_start:
            self._initial_ingest(models)
        return models

    def _locate_bundle(self) -> Path | None:
        pat = re.compile(re.escape(self._scenario_safe()), re.I)
        for f in sorted(self.dir.glob("*.pkl"), key=os.path.getmtime, reverse=True):
            if pat.search(f.name):
                return f
        return None

    def _initial_ingest(self, models: Dict[str, Any]) -> None:
        df  = _dl_csv(self.cfg["start_url"])
        lbl = "label" if "label" in df.columns else "Label"

        pos, neg = df[df[lbl] == 1], df[df[lbl] == 0]
        n_pos = min(len(pos), 2_000); n_neg = min(len(neg), n_pos)
        df_bal = (df.sample(n=min(len(df), 4_000), random_state=42)
                  if n_pos == 0 or n_neg == 0 else
                  pd.concat([pos.sample(n_pos, random_state=42),
                             neg.sample(n_neg, random_state=42)]))
        for _, row in df_bal.iterrows():
            y = self._safe_label(row[lbl]);   x = self._unify(row.drop(labels=[lbl]).to_dict())
            if y is None: continue
            for m in models.values():
                m.learn_one(x, y)

    @staticmethod
    def _numify(v) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
        if isinstance(v, (int, float, np.integer, np.floating)):
            val = float(v); return np.log10(val+1.) if abs(val) > 1e4 else val
        try:
            val = float(v); return np.log10(val+1.) if abs(val) > 1e4 else val
        except (ValueError, TypeError):
            return (abs(hash(str(v))) % 1_000_000) / 1_000_000.0

    def _unify(self, x_raw: Dict[str, Any]) -> Dict[str, float]:
        x = {f: 0.0 for f in self._feats}
        for f, v in x_raw.items():
            x[f] = self._numify(v)
            if f not in self._feats: self._feats.add(f)
        return x

    @staticmethod
    def _safe_label(v) -> int | None:
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        try: return int(v)
        except (ValueError, TypeError): return None

def _dl_csv(url: str, nrows: int | None = None) -> pd.DataFrame:
    r = requests.get(url, stream=True); r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in r.iter_content(1 << 16):
            tmp.write(chunk)
        return pd.read_csv(tmp.name, nrows=nrows)

def get_model_manager(scenario: str, user_id: str = "default") -> IncrementalModelManager:
    return IncrementalModelManager(scenario, user_id)
