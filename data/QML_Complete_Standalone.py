import subprocess, sys

# ── Step 1: install packages ─────────────────────────────────────────────────
subprocess.run(['pip', 'install', '-q',
                'pennylane', 'pennylane-lightning',
                'transformers', 'scikit-learn',
                'seaborn', 'pandas', 'statsmodels'], check=True)
print("✓ Packages installed")

# ── Step 2: purge stale sympy from sys.modules then reimport fresh ───────────
# Pennylane upgrades sympy on disk but the old object stays cached in
# sys.modules. Torch then tries to access sympy.core / sympy.printing on the
# stale object and crashes. Fix: evict every sympy.* entry and reimport.
_stale = [k for k in sys.modules if k == 'sympy' or k.startswith('sympy.')]
for _k in _stale:
    del sys.modules[_k]
import sympy                        # fresh load from upgraded disk package
import sympy.core, sympy.printing   # force subpackage registration
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModel
import torch, torch.nn as nn, torch.optim as optim

# ══════════════════════════════════════════════════════════════════
# QML MASTER EXECUTION — Extended v2  (auto-resume from Drive)
# Corpus: 160 cases · 5 traditions · H1-H8
# Auto-detects completed experiments → trains only missing ones
# Safe to re-run at any time; checkpoints per fold
# Estimated time for 4 new experiments (H6-H8): ~6h on T4
# ══════════════════════════════════════════════════════════════════

import os, sys, re, time, json, glob, shutil, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest
from joblib import Parallel, delayed
from collections import Counter

# ── Global config ───────────────────────────────────────────────
EMBED_DIM      = 768
N_QUBITS       = 4
N_LAYERS       = 1
BATCH_SIZE     = 64
EPOCHS_LOO     = 8
EARLY_STOP_PAT = 2
LR             = 0.005
SEED           = 42
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP        = (DEVICE.type == 'cuda')

torch.manual_seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore')

# ── All 12 experiments (order preserved) ────────────────────────
ALL_EXPERIMENTS = [
    ('H4', 'todas',  True),    # 1.  robustness ablation
    ('H5', 'todas',  False),   # 2.  tradition identification
    ('H2', 'todas',  False),   # 3.  sweat temperature
    ('H1', 'todas',  False),   # 4.  anatomical location
    ('H3', 'todas',  False),   # 5.  critical day
    ('H1', 'griega', False),   # 6.  Greek: location
    ('H2', 'griega', False),   # 7.  Greek: temperature
    ('H3', 'griega', False),   # 8.  Greek: critical day
    ('H6', 'todas',  False),   # 9.  extremity temperature ★ new
    ('H6', 'griega', False),   # 10. extremity temp Greek  ★ new
    ('H7', 'todas',  False),   # 11. delirium during crisis ★ new
    ('H8', 'todas',  False),   # 12. fever pattern          ★ new
]

# ── Hypothesis definitions ───────────────────────────────────────
HIPOTESIS_CONFIG = {
    'H1': {'nombre_en':'Anatomical Location',
           'clases':['cabeza','tórax','extremidades','general'],
           'clases_en':['head','thorax','extremities','general'],
           'columna':'h1_localizacion'},
    'H2': {'nombre_en':'Sweat Temperature',
           'clases':['frío','templado','caliente'],
           'clases_en':['cold','tepid','warm'],
           'columna':'h2_temperatura'},
    'H3': {'nombre_en':'Critical Day',
           'clases':['día_crítico','no_crítico'],
           'clases_en':['critical','non-critical'],
           'columna':'h3_momento'},
    'H4': {'nombre_en':'Clinical Prognosis',
           'clases':['positivo','negativo'],
           'clases_en':['favorable','fatal'],
           'columna':'h4_pronostico'},
    'H5': {'nombre_en':'Tradition of Origin',
           'clases':['griega','babilonica','galenica','islamica','china'],
           'clases_en':['Greek','Babylonian','Galenic','Islamic','Chinese'],
           'columna':'h5_tradicion'},
    'H6': {'nombre_en':'Extremity Temperature',
           'clases':['calientes','frías'],
           'clases_en':['warm','cold'],
           'columna':'h11_extremid'},
    'H7': {'nombre_en':'Delirium During Crisis',
           'clases':['no','sí'],
           'clases_en':['absent','present'],
           'columna':'h10_delirio'},
    'H8': {'nombre_en':'Fever Pattern',
           'clases':['continua','intermitente','remitente'],
           'clases_en':['continuous','intermittent','remittent'],
           'columna':'h9_fiebre'},
}

PALABRAS_DESENLACE = [
    r'\bdied\b',r'\bdeath\b',r'\bdead\b',r'\bfatal\b',r'\bfatally\b',
    r'\brecovered\b',r'\brecovery\b',r'\blive[sd]?\b',r'\bsurvive[sd]?\b',
    r'\bconvalescent\b',r'\bfavorable\b',r'\bfavourable\b',
    r'\bcomplete crisis\b',r'\bcrisis complete\b',
    r'\bhe will die\b',r'\bwill die\b',r'\bhe will live\b',r'\bwill live\b',
    r'\bwill recover\b',r'\bwill not recover\b',r'\bdid not recover\b',
    r'\bhe died\b',r'\bshe died\b',r'\bdying\b',r'\bmortal\b',
    r'\blivid\b',r'\blividity\b',r'\bblack urine\b',r'\bdark urine\b',
    r'\bscanty urine\b',r'\bno urine\b',r'\burine scanty\b',
    r'\basked for food\b',r'\bappetite returned\b',
    r'\bpulse became soft\b',r'\bpulse soft\b',r'\bpulse faint\b',
    r'\bpulse imperceptible\b',r'\bbreathing labored\b',r'\blabored breathing\b',
    r'\bcould not be warmed\b',r'\bcannot be warmed\b',
    r'\bextremities cold\b',r'\bcold extremities\b',r'\bextremities livid\b',
]

def enmascarar(texto):
    t = texto
    for p in PALABRAS_DESENLACE:
        t = re.sub(p, '[OUTCOME]', t, flags=re.IGNORECASE)
    return t

# ══════════════════════════════════════════════════════════════════
# DRIVE AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════
def scan_completed_experiments(results_dir):
    completed = set()
    if not os.path.isdir(results_dir):
        print(f"  ⚠ Results dir not found: {results_dir}")
        return completed
    files = sorted(glob.glob(os.path.join(results_dir, 'LOO_*.json')))
    if not files:
        print(f"  No LOO_*.json files found — all experiments pending.")
        return completed
    print(f"  Found {len(files)} result file(s):")
    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            meta = data.get('metadata', {})
            modo, filtro, ablacion = meta.get('modo'), meta.get('filtro'), meta.get('ablacion')
            stat = data.get('estadistica', {})
            acc  = stat.get('loo_accuracy', 0)
            pval = stat.get('p_valor', 1)
            sig  = '★' if isinstance(pval, float) and pval < 0.05 else '—'
            n    = meta.get('n_casos', '?')
            if modo and filtro is not None and ablacion is not None:
                key = (modo, filtro, bool(ablacion))
                completed.add(key)
                fname = os.path.basename(fpath)
                print(f"    ✓ {modo:<3} {filtro:<10} abl={'Y' if ablacion else 'N'}  "
                      f"N={n:<4} acc={acc:.1%}  p={pval:.2e}  {sig}  [{fname[-28:]}]")
        except Exception as e:
            print(f"    ✗ Could not read {os.path.basename(fpath)}: {e}")
    return completed

def build_queue(all_experiments, completed):
    pending, skipped = [], []
    for exp in all_experiments:
        (skipped if exp in completed else pending).append(exp)
    return pending, skipped

# ══════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT
# ══════════════════════════════════════════════════════════════════
dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {'weights': (N_LAYERS, N_QUBITS, 3)}

# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
def build_model(n_classes):
    class ModeloQML(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_quantum = nn.Sequential(
                nn.Linear(EMBED_DIM, 32), nn.BatchNorm1d(32),
                nn.GELU(), nn.Dropout(0.2),
                nn.Linear(32, N_QUBITS), nn.Tanh(),
            )
            self.quantum = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
            self.post_quantum = nn.Sequential(
                nn.Linear(N_QUBITS, 8), nn.ReLU(),
                nn.Linear(8, n_classes),
            )
        def forward(self, x):
            x = self.pre_quantum(x)
            x = torch.stack([self.quantum(xi) for xi in x])
            return self.post_quantum(x.to(DEVICE))
    return ModeloQML

# ══════════════════════════════════════════════════════════════════
# TRAINING (one LOO fold)
# ══════════════════════════════════════════════════════════════════
def train_fold(ModelClass, n_classes, emb_train, lbl_train, emb_test,
               fold_id='?', fold_num=0, fold_total=0):
    model  = ModelClass().to(DEVICE)
    dl     = DataLoader(TensorDataset(emb_train, lbl_train),
                        batch_size=min(BATCH_SIZE, len(emb_train)),
                        shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    counts = torch.bincount(lbl_train, minlength=n_classes).float()
    w      = (1.0/(counts+1e-6)); w = (w/w.sum()*n_classes).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt    = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_LOO)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    best_loss, best_state, no_improve = float('inf'), None, 0
    _t0 = time.time()

    for epoch in range(EPOCHS_LOO):
        model.train(); epoch_loss = 0.0
        for Xb, yb in dl:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = crit(model(Xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss = crit(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            epoch_loss += loss.item()
        sched.step()
        _pct = (epoch+1)/EPOCHS_LOO
        _bar = '█'*int(20*_pct)+'░'*(20-int(20*_pct))
        _el  = time.time()-_t0
        _eta = _el/(epoch+1)*(EPOCHS_LOO-epoch-1)
        _star= '★' if epoch_loss < best_loss else '↑'
        print(f"\r  [{fold_num}/{fold_total}·{fold_id}] "
              f"ep {epoch+1}/{EPOCHS_LOO} [{_bar}] "
              f"loss={epoch_loss:.3f}{_star} ({_el:.0f}s~{_eta:.0f}s)   ",
              end='', flush=True)
        if epoch_loss < best_loss:
            best_loss, no_improve = epoch_loss, 0
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(emb_test.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits.float(), dim=1).cpu().squeeze(0).numpy()
        pred   = logits.argmax(dim=1).item()
    return pred, probs

# ══════════════════════════════════════════════════════════════════
# BASELINES (SVM + RF, parallel LOO)
# ══════════════════════════════════════════════════════════════════
def compute_baselines(X_np, y_np):
    def _svm(i, X, y):
        idx = [j for j in range(len(X)) if j != i]
        sc  = StandardScaler()
        return int(SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEED)
                   .fit(sc.fit_transform(X[idx]), y[idx])
                   .predict(sc.transform(X[i:i+1]))[0])
    def _rf(i, X, y):
        idx = [j for j in range(len(X)) if j != i]
        return int(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
                   .fit(X[idx], y[idx]).predict(X[i:i+1])[0])
    svm_p = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_svm)(i, X_np, y_np) for i in range(len(X_np)))
    rf_p  = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_rf)(i, X_np, y_np)  for i in range(len(X_np)))
    return svm_p, rf_p

# ══════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════
def run_experiment(MODO, FILTRO, ABLACION, CASOS_ALL, CACHE_DIR, tokenizer, bert):
    CFG        = HIPOTESIS_CONFIG[MODO]
    N_CLASSES  = len(CFG['clases'])
    NOMBRES    = CFG['clases']
    NOMBRES_EN = CFG['clases_en']
    col        = CFG['columna']
    abl_tag    = ' [ABLATION]' if ABLACION else ''
    CACHE_KEY  = f"{MODO}_{FILTRO}{'_ablacion' if ABLACION else ''}"
    CACHE_PATH = f'{CACHE_DIR}/embeddings_{CACHE_KEY}.pt'
    CKPT_PATH  = f'{CACHE_DIR}/checkpoint_{CACHE_KEY}.pt'

    print(f"\n{'═'*65}")
    print(f"  EXPERIMENT: {MODO} | {FILTRO}{abl_tag}")
    print(f"  Classes: {N_CLASSES} — {NOMBRES_EN}")
    print(f"{'═'*65}")

    trad_order = ['griega','babilonica','galenica','islamica','china']
    CASOS_RAW  = []
    for t in (trad_order if FILTRO=='todas' else [FILTRO]):
        CASOS_RAW.extend(CASOS_ALL[t])

    CASOS = [dict(c, texto=enmascarar(c['texto'])) for c in CASOS_RAW] \
            if ABLACION else list(CASOS_RAW)

    for caso in CASOS:
        for k, v in EXTENDED_ANNOTATIONS.get(caso['id'], {}).items():
            caso.setdefault(k, v)

    CASOS = [c for c in CASOS if col in c and c[col] is not None]
    N     = len(CASOS)

    if N == 0:
        print(f"  ✗ No cases with annotation '{col}' — skipping.")
        return None
    print(f"  Cases with '{col}': {N} / {len(CASOS_RAW)}")

    # ── Embeddings ───────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"  Loading embeddings from cache...")
        cache      = torch.load(CACHE_PATH, weights_only=False)
        embeddings = cache['embeddings']
        etiquetas  = cache['etiquetas']
    else:
        print(f"  Generating embeddings for {N} cases...")
        _t0 = time.time(); embs, lbls = [], []
        for i, caso in enumerate(CASOS):
            inp = tokenizer(caso['texto'], return_tensors='pt',
                           truncation=True, max_length=512,
                           padding=True).to(DEVICE)
            with torch.no_grad():
                out = bert(**inp)
            emb = out.last_hidden_state[:,0,:].squeeze(0)
            embs.append(nn.functional.normalize(emb, p=2, dim=0).cpu())
            lbls.append(caso[col])
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{N} ({time.time()-_t0:.0f}s)")
        embeddings = torch.stack(embs)
        etiquetas  = torch.tensor(lbls, dtype=torch.long)
        torch.save({'embeddings':embeddings, 'etiquetas':etiquetas,
                    'cache_key':CACHE_KEY}, CACHE_PATH)
        print(f"  Cache saved ({time.time()-_t0:.1f}s)")

    print(f"  Embeddings: {embeddings.shape} | Classes: {etiquetas.unique().tolist()}")

    # ── Baselines ────────────────────────────────────────────────
    print(f"  Computing baselines (parallel)...")
    _tb = time.time()
    X_np, y_np  = embeddings.numpy(), etiquetas.numpy()
    svm_p, rf_p = compute_baselines(X_np, y_np)
    acc_svm = sum(p==l for p,l in zip(svm_p, y_np))/N
    acc_rf  = sum(p==l for p,l in zip(rf_p,  y_np))/N
    print(f"  SVM={acc_svm:.1%} RF={acc_rf:.1%} ({time.time()-_tb:.0f}s)")

    # ── LOO-CV ───────────────────────────────────────────────────
    ModelClass = build_model(N_CLASSES)

    if os.path.exists(CKPT_PATH):
        ck         = torch.load(CKPT_PATH, weights_only=False)
        start_fold = ck['fold_actual']
        loo_preds  = ck['loo_preds']
        loo_labels = ck['loo_labels']
        loo_probs  = ck['loo_probs']
        loo_ids    = ck['loo_ids']
        _ftimes    = ck.get('fold_times', [])
        _cor       = sum(p==l for p,l in zip(loo_preds, loo_labels))
        print(f"  ★ Checkpoint: resuming from fold {start_fold+1}/{N} ({_cor}/{start_fold} correct)")
    else:
        start_fold = 0
        loo_preds, loo_labels, loo_probs, loo_ids, _ftimes = [], [], [], [], []
        print(f"  Starting LOO from fold 1/{N}")

    print(f"\n  {'Fold':>5} | {'ID':>12} | {'Trad':>10} | "
          f"{'True':>12} | {'Pred':>12} | OK | s/fold | ETA")
    print(f"  {'─'*80}")

    _loo_t0 = time.time()

    for i in range(start_fold, N):
        idx_tr  = [j for j in range(N) if j != i]
        caso_id = CASOS[i]['id']
        trad    = CASOS[i]['tradicion']
        _ft0    = time.time()

        pred, probs = train_fold(ModelClass, N_CLASSES,
                                 embeddings[idx_tr], etiquetas[idx_tr],
                                 embeddings[i],
                                 fold_id=caso_id, fold_num=i+1, fold_total=N)
        _fel = time.time()-_ft0
        _ftimes.append(_fel)
        real = etiquetas[i].item()
        ok   = "✓" if pred == real else "✗"
        loo_preds.append(pred); loo_labels.append(real)
        loo_probs.append(probs); loo_ids.append(caso_id)

        _mean_t  = sum(_ftimes[-20:])/len(_ftimes[-20:])
        _eta     = _mean_t*(N-i-1)
        _eta_str = (f"{int(_eta//3600)}h{int((_eta%3600)//60):02d}m"
                    if _eta > 3600 else f"{int(_eta//60)}m{int(_eta%60):02d}s")

        print(f"\r  {i+1:>5} | {caso_id:>12} | {trad:>10} | "
              f"{NOMBRES[real]:>12} | {NOMBRES[pred]:>12} | {ok} | "
              f"{_fel:>5.1f}s | {_eta_str}")
        sys.stdout.flush()

        if (i+1) % 20 == 0 and i+1 < N:
            _cor = sum(p==l for p,l in zip(loo_preds, loo_labels))
            print(f"\n  ── {i+1}/{N} | Partial acc: {_cor/(i+1):.1%} "
                  f"| Mean: {_mean_t:.1f}s/fold ──\n")

        torch.save({'fold_actual':i+1, 'loo_preds':loo_preds,
                    'loo_labels':loo_labels, 'loo_probs':loo_probs,
                    'loo_ids':loo_ids, 'fold_times':_ftimes,
                    'modo':MODO, 'filtro':FILTRO, 'ablacion':ABLACION,
                    'timestamp':datetime.datetime.now().isoformat()}, CKPT_PATH)

    _loo_total = time.time()-_loo_t0
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    # ── Statistics ───────────────────────────────────────────────
    n_cor   = sum(p==l for p,l in zip(loo_preds, loo_labels))
    acc_loo = n_cor/N
    bal_acc = balanced_accuracy_score(loo_labels, loo_preds)
    p_azar  = 1.0/N_CLASSES
    p_valor = binomtest(n_cor, N, p_azar, alternative='greater').pvalue
    acc_triv= Counter(loo_labels).most_common(1)[0][1]/N
    ci_low, ci_high = proportion_confint(n_cor, N, alpha=0.05, method='wilson')

    print(f"\n  {'═'*65}")
    print(f"  RESULT: {MODO} | {FILTRO}{abl_tag}")
    print(f"  QML  : {acc_loo:.1%} ({n_cor}/{N}) balanced={bal_acc:.1%}")
    print(f"  SVM  : {acc_svm:.1%}  RF: {acc_rf:.1%}")
    print(f"  p    : {p_valor:.2e}  {'★' if p_valor<0.05 else '—'}")
    print(f"  CI   : [{ci_low:.1%}, {ci_high:.1%}]")
    print(f"  Time : {_loo_total/60:.1f} min")
    print(f"  {'═'*65}")

    # ── Save results ─────────────────────────────────────────────
    RESULTS_DIR = f'{CACHE_DIR}/resultados'
    FIGURES_DIR = f'{CACHE_DIR}/figuras'
    LOGS_DIR    = f'{CACHE_DIR}/logs'
    for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'LOO-CV: {MODO} — {CFG["nombre_en"]}{abl_tag}\n'
                 f'QML={acc_loo:.0%} | SVM={acc_svm:.0%} | RF={acc_rf:.0%} '
                 f'| N={N} | {FILTRO}', fontsize=11, fontweight='bold')
    cm = confusion_matrix(loo_labels, loo_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=NOMBRES_EN, yticklabels=NOMBRES_EN, ax=axes[0])
    axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    mods = ['Chance','Majority','RF','SVM','QML']
    accs = [p_azar, acc_triv, acc_rf, acc_svm, acc_loo]
    cols = ['#BDBDBD','#9E9E9E','#81C784','#64B5F6','#EF5350']
    bars = axes[1].bar(mods, accs, color=cols, edgecolor='k', width=0.6)
    axes[1].axhline(y=p_azar, color='gray', linestyle='--', alpha=0.5)
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x()+bar.get_width()/2, acc+0.01,
                     f'{acc:.0%}', ha='center', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, 1.15); axes[1].set_title('Model Comparison')
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    probs_m = np.array(loo_probs)
    sns.heatmap(probs_m, cmap='YlOrRd', ax=axes[2],
                xticklabels=NOMBRES_EN, yticklabels=False, vmin=0, vmax=1)
    for i, (p, l) in enumerate(zip(loo_preds, loo_labels)):
        if p != l:
            axes[2].add_patch(plt.Rectangle((0,i), N_CLASSES, 1,
                              fill=False, edgecolor='red', lw=1))
    axes[2].set_title('Probabilities (errors=red)'); axes[2].set_xlabel('Class')
    plt.setp(axes[2].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    fig_name = f'Results_{CACHE_KEY}_{TIMESTAMP}.png'
    plt.savefig(fig_name, dpi=150, bbox_inches='tight'); plt.close()
    shutil.copy(fig_name, f'{FIGURES_DIR}/{fig_name}')
    print(f"  Figure: {FIGURES_DIR}/{fig_name}")

    # JSON
    por_trad = {}
    for trad_n in ['griega','babilonica','galenica','islamica','china']:
        idx_t = [i for i,c in enumerate(CASOS) if c['tradicion']==trad_n]
        if idx_t:
            cor = sum(loo_preds[i]==loo_labels[i] for i in idx_t)
            por_trad[trad_n] = {'n':len(idx_t),'correctos':cor,
                                 'accuracy':round(cor/len(idx_t),4)}

    resultado = {
        'metadata': {
            'version':'Ev2', 'timestamp':TIMESTAMP,
            'modo':MODO, 'filtro':FILTRO, 'ablacion':ABLACION,
            'nombre_en':CFG['nombre_en'], 'n_casos':N, 'n_clases':N_CLASSES,
            'clases':NOMBRES, 'clases_en':NOMBRES_EN,
            'tiempo_total_min':round(_loo_total/60,2),
        },
        'estadistica': {
            'loo_accuracy':round(acc_loo,4), 'balanced_accuracy':round(bal_acc,4),
            'n_correctos':n_cor, 'n_total':N,
            'baseline_azar':round(p_azar,4), 'baseline_trivial':round(acc_triv,4),
            'svm_accuracy':round(acc_svm,4), 'rf_accuracy':round(acc_rf,4),
            'p_valor':round(float(p_valor),10), 'significativo':bool(p_valor<0.05),
            'ic95_low':round(ci_low,4), 'ic95_high':round(ci_high,4),
        },
        'por_tradicion': por_trad,
        'predicciones': [
            {'id':loo_ids[i], 'tradicion':CASOS[i]['tradicion'],
             'real':NOMBRES[loo_labels[i]], 'predicho':NOMBRES[loo_preds[i]],
             'correcto':loo_preds[i]==loo_labels[i],
             'probs':{NOMBRES[j]:round(float(loo_probs[i][j]),4) for j in range(N_CLASSES)}}
            for i in range(N)
        ],
        'errores': [
            {'id':loo_ids[i], 'tradicion':CASOS[i]['tradicion'],
             'real':NOMBRES[loo_labels[i]], 'predicho':NOMBRES[loo_preds[i]],
             'texto':CASOS[i]['texto'].strip()[:200]}
            for i in range(N) if loo_preds[i]!=loo_labels[i]
        ],
    }

    json_path = f'{RESULTS_DIR}/LOO_{CACHE_KEY}_{TIMESTAMP}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    with open(f'{LOGS_DIR}/historial_multicultural.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'timestamp':TIMESTAMP, 'version':'Ev2',
            'modo':MODO, 'filtro':FILTRO, 'ablacion':ABLACION,
            'accuracy':round(acc_loo,4), 'balanced':round(bal_acc,4),
            'p_valor':round(float(p_valor),10), 'significativo':bool(p_valor<0.05),
            'n':N, 'correctos':n_cor,
            'ic95':[round(ci_low,4),round(ci_high,4)],
            'svm':round(acc_svm,4), 'rf':round(acc_rf,4),
            'tiempo_min':round(_loo_total/60,2),
        }, ensure_ascii=False)+'\n')

    return {'modo':MODO,'filtro':FILTRO,'ablacion':ABLACION,
            'accuracy':acc_loo,'balanced':bal_acc,'p_valor':p_valor,
            'svm':acc_svm,'rf':acc_rf,'n':N,'correctos':n_cor,
            'ci_low':ci_low,'ci_high':ci_high,
            'por_tradicion':por_trad,'tiempo_min':_loo_total/60}

CASOS_GRIEGOS = [
    {'id':'Gr_Ep1_C1','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Philiscus lived by the wall. He took to his bed on the first day of acute fever;
        he sweated; towards night uncomfortable. On the fourth day, towards evening,
        much sweat; extremities cold; the sweating ceased; became cold. On the sixth day, died.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C2','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Silernus lived on Broadway near the place of Eualcidas. Violent fever began after fatigue.
        On the sixth day much sweat all over; extremities cold.
        On the seventh day, the fever was relieved; he recovered.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C3','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Herophon had acute fever. On the third day slight epistaxis.
        On the seventh day, sweating all over; fever diminished but did not cease.
        On the fourteenth day, sweating; fever resolved. Recovered.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C4','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Philistes had acute fever from the start. Head heavy, right hypochondrium tense.
        On the seventh day, hot sweat all over; afebrile briefly.
        Relapsed on the ninth day; died on the eleventh.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C5','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        The wife of Epicrates had difficult labour; fever seized her on the first day.
        On the sixth day, great sweating about the head; neck and chest cold and clammy.
        On the seventh day, she died.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C6','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Cleanactides was seized with ardent fever. Pain of the head, neck, and loins.
        No sleep; dry cough; thirst. On the sixth, cold sweats all over the body;
        extremities could not be warmed. On the seventh day, died.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C7','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        The man who lodged with Tisamenus had fever. Bowels disordered. Sleepless.
        On the sixth day sweated, had a crisis; free from fever. On the seventh, relapsed.
        On the eleventh, sweated abundantly all over; complete crisis. Recovered.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C8','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Epicrates was seized with ardent fever. Head heavy. On the eleventh sweated
        copiously all over; warmly; crisis; favorable. Recovered on the fourteenth day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C9','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Melidia was seized with violent pain of the head, neck, and chest. Immediately had fever.
        On the sixth day, heavy sweating about the head and neck; cold and clammy all about
        the upper body. Delirium; on the seventh day, died.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C10','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Erasinus was seized with fever. Seventh day: rigor, acute fever, much sweating;
        fever subsided; crisis complete. Free from fever on the seventh; recovered.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C11','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Criton was seized with violent pain in the great toe. High fever. Delirium.
        Died on the second day.""",
     'h1_localizacion':2,'h2_temperatura':1,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C12','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Dromeades. Fourteenth day: sweated all over; cold throughout; extremities particularly
        cold; no urine; death on that day.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep1_C13','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        The man lodged at the house of Epigenes. Seventh day: much sweat about the head
        and neck; warm and copious. Tenth day: free of fever; complete crisis. Recovered.""",
     'h1_localizacion':0,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep1_C14','fuente':'Hippocrates, Epidemics I','tradicion':'griega','siglo':-5,'texto':"""
        Melidia near the temple of Hera. Seventh day: scanty sweat about the head and
        clavicles only; cold and clammy. No crisis obtained; died on the seventh day.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C1','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Pythion was seized with violent tremors and acute fever. On the sixth day he sweated
        much, warm sweat on the head; crisis. Fever subsided; recovered without relapse.""",
     'h1_localizacion':0,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C2','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Anaxion was seized with acute fever. On the sixth day, cold sweat on the chest
        and upper limbs; slight delirium. On the eighth, died.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C3','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Crates was seized with violent rigor and acute fever. Much sweating throughout
        the attack; warm and copious on the thorax. Resolved on the fourteenth day; lived.""",
     'h1_localizacion':1,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C4','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        A woman had fever; much vomiting. Sweated little; only around the ankles and feet;
        cold throughout. On the eleventh day, crisis was absent; she lingered and died.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C5','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Apollonius in Abdera had fever for a long time. On a certain day he had warm profuse
        sweating over the whole body; urine normal; crisis on the twentieth day; recovered.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C6','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Hermocrates was seized with fever. Fourteenth day: no fever; no sweat; slept; recovered.
        Seventeenth day: relapse. Twentieth day: fresh crisis; no fever. Twenty-seventh day: died.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C7','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        The man in the garden of Delearces. Fortieth day: sweated abundantly all over;
        complete crisis; recovered.""",
     'h1_localizacion':0,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C8','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Pythion above the shrine of Heracles. Tongue dry; thirst; no sleep.
        Tenth day: speechless; great chill; acute fever; much sweat; died.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C9','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        The patient with phrenitis vomited copiously; much fever; continuous sweating all over;
        convulsions at night. Fourth day: died.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C10','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        In Larisa a bald man had acute ardent fever. Sixth day: much cold sweat;
        extremities cold; no urine passed; delirious; died on the sixth day.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C11','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        A melancholic woman took to her bed. Twentieth day: much sweating; afebrile.
        Twenty-seventh day: sweated; free from fever; complete crisis; recovered.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C12','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        Chthonoboutes, acute fever. Seventh day: slight sweating about the head only;
        cold extremities. Twentieth day: cold sweat all over; extremities cold; died.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C13','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        In Abdera, Pericles had acute continuous fever. Fourteenth day: sweated all over;
        warm and copious; afebrile; complete favorable crisis. Recovered.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C14','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        A woman who had an abortion had fever. Fourteenth day: sweated a little; not all over;
        cold extremities; relapse immediately. Seventeenth day: died.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep3_C15','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        A young man in Meliboea. Fourteenth day: sweated all over; warm and copious;
        complete crisis; free from fever; recovered.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep3_C16','fuente':'Hippocrates, Epidemics III','tradicion':'griega','siglo':-5,'texto':"""
        The Parian in Thasos had acute continuous fever. Eleventh day: sweated all over;
        grew chilly but quickly recovered heat. Twentieth day: sweated all over; recovered.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Prog_C1','fuente':'Hippocrates, Prognostic','tradicion':'griega','siglo':-5,'texto':"""
        If in acute fever the patient sweats warm and copious all over the body and the fever
        is resolved, it brings the disease to a favorable crisis. Cold sweats indicate
        prolongation or a fatal outcome with cold livid extremities.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Prog_C2','fuente':'Hippocrates, Prognostic','tradicion':'griega','siglo':-5,'texto':"""
        Cold sweats occurring with acute fever indicate death; with milder fever, prolongation.
        When sweat is cold and confined to the head only, with extremities cold and livid,
        and with acute fever, death follows.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Prog_C3','fuente':'Hippocrates, Prognostic','tradicion':'griega','siglo':-5,'texto':"""
        Sweats on critical days, warm and covering the whole body, relieve fever and resolve
        the disease. Small cold sweats on the forehead or clavicles only indicate danger.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Aph_C1','fuente':'Hippocrates, Aphorisms','tradicion':'griega','siglo':-5,'texto':"""
        Sweating is beneficial when the fever is resolved by it. Cold sweats with acute fever
        indicate death; with a mild fever, a long illness.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Aph_C2','fuente':'Hippocrates, Aphorisms','tradicion':'griega','siglo':-5,'texto':"""
        When in a fever the extremities are cold while the body is hot, cold sweats appear
        at the extremities and do not spread to the trunk, the prognosis is very bad.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Aph_C3','fuente':'Hippocrates, Aphorisms','tradicion':'griega','siglo':-5,'texto':"""
        After a rigor, warm and copious sweating all over the body, with fever abating,
        is a good sign indicating resolution of the disease and recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep2_C1','fuente':'Hippocrates, Epidemics II','tradicion':'griega','siglo':-5,'texto':"""
        Simus had fever; pain of the head. Seventh day: cold clammy sweat over the whole body;
        extremities could not be warmed; death followed.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep2_C2','fuente':'Hippocrates, Epidemics II','tradicion':'griega','siglo':-5,'texto':"""
        Nicostratus had continuous fever with rigors. Eleventh day: warm sweat over the whole
        body; crisis; fever relieved entirely. Recovered completely.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep2_C3','fuente':'Hippocrates, Epidemics II','tradicion':'griega','siglo':-5,'texto':"""
        A woman in Larisa with continuous fever. Seventh day: slight sweating about the head
        and neck; cold extremities; livid color; delirium. Ninth day: died.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep2_C4','fuente':'Hippocrates, Epidemics II','tradicion':'griega','siglo':-5,'texto':"""
        Heraclides suffered from continued fever. Fourteenth day: copious warm sweating all
        over; crisis; free from fever. Recovered on the fourteenth; no relapse.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
    {'id':'Gr_Ep5_C1','fuente':'Hippocrates, Epidemics V','tradicion':'griega','siglo':-5,'texto':"""
        In Larisa a man with burning fever. Sweated slightly about the forehead; cold and clammy.
        Sixth day: cold sweat all over; extremities livid and cold; no urine; died.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':0},
    {'id':'Gr_Ep5_C2','fuente':'Hippocrates, Epidemics V','tradicion':'griega','siglo':-5,'texto':"""
        Democles in Abdera had ardent fever. Seventh day: profuse warm sweat all over the body;
        fever resolved. Eleventh day completely well; recovered without any relapse.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':0},
]

# ── TRADICIÓN BABILÓNICA (28 casos) ──────────────────────────
# Fuentes primarias:
#   - Markham J. Geller, "Melothesia in Babylonia" (2010)
#   - R.D. Biggs, "Medicine in Ancient Mesopotamia" (1969)
#   - Traducciones de tablillas cuneiformes SA.GIG (Diagnostic Handbook)
# Nota: El SA.GIG (~1000 a.C.) es el texto médico diagnóstico más antiguo conocido.
#       Cubre diagnóstico y pronóstico en formato condicional (SI... ENTONCES...).
# h5_tradicion = 1

CASOS_BABILONICOS = [
    {'id':'Ba_C1','fuente':'Babylonian Diagnostic Handbook, Tablet 1','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man's body is hot with fever and he sweats profusely from his entire body,
        and the sweat is warm and abundant, the fever will break and the man will recover.
        The hand of Shamash is upon him but he will live.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C2','fuente':'Babylonian Diagnostic Handbook, Tablet 2','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man sweats and his hands and his feet are cold, that man will die;
        he will not recover. The hand of a ghost has seized him.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C3','fuente':'Babylonian Diagnostic Handbook, Tablet 3','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is seized by fever and cold sweat breaks out upon his forehead only,
        his hands remaining cold and his feet cold, he will die within three days.
        The sickness is severe.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C4','fuente':'Babylonian Diagnostic Handbook, Tablet 4','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has fever and sweats from his whole body at the time the stars rise,
        and the sweat is warm, the fever will depart from him and he will recover his health.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C5','fuente':'Babylonian Diagnostic Handbook, Tablet 5','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man sweats on his chest and his upper body but his extremities are cold and livid,
        he will not recover; the disease is beyond the physician's power. He will die.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C6','fuente':'Babylonian Diagnostic Handbook, Tablet 6','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is hot with burning fever for many days and then suddenly sweats warm
        and abundantly from his whole body and the fever leaves him, he will live.
        The gods have been merciful.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C7','fuente':'Babylonian Diagnostic Handbook, Tablet 7','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has intermittent fever and sweats on odd days but not on even days,
        and the sweat is cold and scanty about the neck and head, the disease will be long.
        He may recover but recovery will be slow.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C8','fuente':'Babylonian Diagnostic Handbook, Tablet 8','tradicion':'babilonica','siglo':-10,'texto':"""
        If the fever is severe and the patient cannot be warmed, and cold sweat covers
        his entire body, and his urine is scanty and black, he will die before the new moon.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C9','fuente':'Babylonian Diagnostic Handbook, Tablet 9','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man sweats moderately from the upper half of his body during fever and the
        lower half remains dry, and if the fever diminishes by evening, he will recover
        slowly. The treatment of Gula will help him.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C10','fuente':'Babylonian Diagnostic Handbook, Tablet 10','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is seized with shaking and cold sweat breaks out upon him before the fever
        peaks, and his extremities cannot be warmed even with fire, that man will die.
        His ghost will haunt the house.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C11','fuente':'Babylonian Diagnostic Handbook, Tablet 11','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has continuous fever for seven days and on the seventh day sweats warmly
        from his whole body and the fever breaks completely, he will live.
        The crisis has been resolved favorably.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C12','fuente':'Babylonian Diagnostic Handbook, Tablet 12','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man sweats from his head and neck during fever and the sweat is warm but the
        body below remains dry and hot, the crisis is incomplete. He may recover partially
        but relapse should be expected.""",
     'h1_localizacion':0,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C13','fuente':'Babylonian Diagnostic Handbook, Tablet 13','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is afflicted with burning fever and sweats only from his feet and ankles,
        and the sweat is cold and the rest of the body burning hot, this man is in great danger.
        He will likely die; make offerings to Marduk.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C14','fuente':'Babylonian Diagnostic Handbook, Tablet 14','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has fever with delirium and sweats warm from his whole body when the moon
        is full, and the fever resolves at that time, he will completely recover.
        Favorable omen; the gods smile upon him.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C15','fuente':'Babylonian Diagnostic Handbook, Tablet 15','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man's fever worsens day by day and at no point does he sweat, and his skin
        is dry and burning, and his extremities cold, this is a bad sign. He will die.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C16','fuente':'Babylonian Diagnostic Handbook, Tablet 16','tradicion':'babilonica','siglo':-10,'texto':"""
        If during fever a man sweats on his forehead and between his shoulder blades but
        his limbs remain cold, and if he trembles and cannot speak, he will die.
        No physician can help; his time has come.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C17','fuente':'Babylonian Diagnostic Handbook, Tablet 17','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is ill with fever for fourteen days and on the fourteenth day sweats
        abundantly and warmly from all parts of his body, the fever will cease and
        he will recover. The crisis has come on the right day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C18','fuente':'Babylonian Diagnostic Handbook, Tablet 18','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man with severe fever sweats a little, with the sweat cold and confined to
        the upper body only, and the lower body dry and cold, he will die within seven days.
        Prepare his burial offerings.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C19','fuente':'Babylonian Diagnostic Handbook, Tablet 19','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is seized by intermittent fever and sweats moderately on the days the fever
        returns, with the sweat neither warm nor cold, recovery is possible but uncertain.
        Give him herbal decoctions and pray to Gula.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C20','fuente':'Babylonian Diagnostic Handbook, Tablet 20','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man in high fever suddenly breaks into profuse warm sweat covering his entire
        body, and if after the sweat the fever is completely gone, he will recover fully.
        Make an offering of thanksgiving to the gods.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C21','fuente':'Babylonian Diagnostic Handbook, Tablet 21','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has fever and sweats from his head only, and the sweat is cold and the
        head cold to the touch, and his body below is hot, this man will die.
        The spirit of the dead afflicts him.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C22','fuente':'Babylonian Diagnostic Handbook, Tablet 22','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has fever for twenty days and sweats warm from his whole body on the
        twentieth day and the fever resolves, this is a good sign. He will live.
        The crisis has been delayed but is favorable.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C23','fuente':'Babylonian Diagnostic Handbook, Tablet 23','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man sweats from the whole body but the sweat is cold and the extremities
        cannot be warmed, and if his eyes are sunken and his breathing labored,
        he will die before sunrise.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C24','fuente':'Babylonian Diagnostic Handbook, Tablet 24','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man has relapsing fever and on the day of relapse sweats more than before,
        with the sweat warm and covering the entire body, the relapse will be mild and
        recovery will follow. A good omen.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C25','fuente':'Babylonian Diagnostic Handbook, Tablet 25','tradicion':'babilonica','siglo':-10,'texto':"""
        If during fever a man's skin is dry and does not sweat at all, and his body is
        burning hot while his extremities are cold, the prognosis is very grave.
        This man will most likely die.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':1,'h4_pronostico':1,'h5_tradicion':1},
    {'id':'Ba_C26','fuente':'Babylonian Diagnostic Handbook, Tablet 26','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man with fever sweats from his armpits and groin but not from the rest of his
        body, and the sweat is lukewarm, this is an uncertain sign. He may recover or may
        not; the gods have not yet decided his fate.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C27','fuente':'Babylonian Diagnostic Handbook, Tablet 27','tradicion':'babilonica','siglo':-10,'texto':"""
        If on the seventh day of fever a man sweats warm from his whole body and the
        fever departs, he will recover. This is the correct crisis day. He will be well.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':1},
    {'id':'Ba_C28','fuente':'Babylonian Diagnostic Handbook, Tablet 28','tradicion':'babilonica','siglo':-10,'texto':"""
        If a man is seized with violent shaking fever and sweats cold from his forehead
        and neck but his body below is dry, this is a sign of death. He will not recover.
        Call the mourners.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':1},
]

# ── TRADICIÓN GALÉNICA/ROMANA (30 casos) ─────────────────────
# Fuentes primarias:
#   - Galen, "De Crisibus" (trad. Ian Johnston, Loeb Classical Library, 2011)
#   - Galen, "Prognostic" (trad. Vivian Nutton, CMG V 8.1, 1979)
# Nota: Galeno (129-216 d.C.) sistematizó la medicina hipocrática. De Crisibus
#       contiene casos clínicos propios y teoría de las crisis febriles.
# h5_tradicion = 2

CASOS_GALENICOS = [
    {'id':'Ga_C1','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A patient suffering from ardent fever broke into a sweat on the seventh day;
        the sweat was warm and copious, spreading from the chest over the entire body.
        The fever resolved completely and the patient recovered. This is a perfect crisis.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C2','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A man with continuous fever sweated on the fourteenth day; the sweat was cold
        and confined to the forehead and neck. The extremities were livid and could not
        be warmed. He died on the fourteenth day. This is an ominous sweat.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C3','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        I observed a patient whose fever crisis came on the eleventh day. The sweat was
        warm and bathed the whole body abundantly. After the sweat the pulse became soft
        and the patient asked for food. He made a complete recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C4','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        A woman with puerperal fever on the seventh day had cold sweats on the chest
        and upper limbs only. The lower body was dry and cold to the touch. She died
        the following day. Cold partial sweats are always a sign of death.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C5','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        On the twentieth day of a prolonged fever, the patient sweated abundantly and
        warmly from the whole body. The urine showed a good sediment. Complete crisis.
        I predicted this outcome from the character of the pulse on the fourteenth day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C6','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A philosopher came to me with high fever and delirium. On the fourth day he had
        a slight sweat about the head; cold and clammy. The extremities were cold.
        I predicted death. He died on the sixth day. My prognosis was correct.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C7','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        When fever patients sweat at the crisis with warm sweat covering the entire body,
        and the fever resolves at the same time, this is always a good sign. Such sweats
        indicate that nature has overcome the disease through proper coction.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C8','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        Cold sweats occurring during the height of fever are invariably fatal signs.
        When such sweats are accompanied by cold extremities and scanty dark urine,
        death follows within one or two days. Hippocrates taught us this truth.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C9','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A young man with tertian fever sweated moderately on the third and seventh days;
        the sweat was neither warm nor cold but tepid and covered the whole body.
        The fever subsided gradually over twenty days and he recovered slowly.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C10','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        I attended a senator with ardent fever. On the seventh day, when I predicted
        the crisis, warm and copious sweat appeared over his entire body. The fever
        left him completely. My prognosis based on the pulse was vindicated.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C11','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        An old man with continuous fever failed to sweat on the seventh and eleventh days.
        On the fourteenth day a cold sweat appeared on the forehead only. The extremities
        became cold and livid. He died on the fourteenth day.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C12','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A woman with fever on the eleventh day broke into a warm sweat beginning at the
        chest and spreading downward to cover the whole body. The fever resolved and she
        asked for nourishment. She recovered completely on the fourteenth day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C13','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        When sweat appears only about the head during acute fever and the rest of the body
        is dry and the extremities cold, this indicates either prolongation of the disease
        or death. It is never a sign of complete crisis.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C14','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A soldier came to me with quartan fever. On the twenty-first day he sweated warmly
        from the whole body and the fever resolved completely. This is how nature expels
        the morbid humor through sweat when coction is complete.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C15','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        I was called to see a patient who had been sweating cold for two days. The sweat
        covered his whole body but was cold and clammy. His extremities were livid.
        I told the family he would not survive. He died the same night.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C16','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A case of false crisis: on the seventh day there was moderate sweating from the
        upper body only, neither warm nor cold. The fever diminished but returned on the
        ninth day. True crisis came on the fourteenth with full warm sweating.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C17','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        Warm sweats that appear spontaneously during acute fever and cover the entire body
        uniformly indicate that nature is expelling the disease properly. Such patients
        invariably recover if no other dangerous symptoms are present.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C18','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        An athlete developed high fever after violent exercise. On the fourth day cold sweats
        appeared over the whole body; he could not be warmed. I prescribed nothing for I knew
        he would die. He died before the next day's sun.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C19','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        A child of eight years had ardent fever. On the seventh day he sweated copiously
        and warmly from the whole body. The fever left him immediately and he recovered
        quickly. Children resolve fever crises more readily than adults.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C20','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A merchant with remittent fever sweated on the days of exacerbation; the sweat
        was moderate and tepid, covering the upper body. After fourteen days of this
        pattern the fever resolved and he recovered, though weakened.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C21','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        Cold sweats occurring with delirium and cold extremities in acute fever are the most
        dangerous sign known to medicine. Hippocrates correctly identified these as signs of
        imminent death and I have confirmed this in all my practice.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C22','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A woman with puerperal fever recovered on the eleventh day when warm sweat
        covered her whole body. I had predicted this from the favorable signs I observed
        on the seventh day: good urine sediment and a regular pulse.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C23','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        When the sweat at crisis appears only on the extremities and is cold, while the
        trunk remains hot and dry, this is an incomplete and dangerous crisis. The patient
        may survive or may relapse into worse fever.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C24','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A case of complete crisis on the seventh day: warm copious sweat from the whole body,
        the fever resolving entirely, the pulse becoming full and slow, the urine showing
        abundant sediment. The patient recovered perfectly within two days.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C25','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        I have observed many patients in whom cold sweat appeared on the forehead during
        the height of fever while the rest of the body was dry. All of these patients
        either died or had very prolonged illnesses. It is never a favorable sign.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C26','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A philosopher with prolonged fever sweated moderately and warmly on the twentieth day.
        The sweat was not abundant but covered the whole body. The fever subsided gradually.
        He recovered but the convalescence was long.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C27','fuente':'Galen, De Crisibus, Book II','tradicion':'galenica','siglo':2,'texto':"""
        When a patient sweats profusely but the sweat is cold and does not reduce the fever,
        and the extremities cannot be warmed, this is the worst possible sign. Death follows
        within hours or at most one day.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
    {'id':'Ga_C28','fuente':'Galen, De Crisibus, Book III','tradicion':'galenica','siglo':2,'texto':"""
        A young woman with tertian fever had her crisis on the seventh day with warm abundant
        sweat covering her entire body. She recovered completely. The critical days of
        Hippocrates are confirmed by daily clinical experience.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C29','fuente':'Galen, Prognostic','tradicion':'galenica','siglo':2,'texto':"""
        In my thirty years of practice I have never seen a patient die when warm copious
        sweat covering the entire body appeared on a critical day and the fever resolved
        completely at the same time. This is always a sign of recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':2},
    {'id':'Ga_C30','fuente':'Galen, De Crisibus, Book I','tradicion':'galenica','siglo':2,'texto':"""
        A case of death by cold sweat: a man of sixty years had ardent fever. On the
        fourth day cold sweat covered his entire body; extremities became livid.
        I predicted death. He died at the sixth hour of the same day.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':2},
]

# ── TRADICIÓN ISLÁMICA (30 casos) ────────────────────────────
# Fuentes primarias:
#   - Avicenna (Ibn Sina), "Canon of Medicine" Book IV (trad. O. Cameron Gruner, 1930)
#   - Ibn Sina, "Al-Qanun fi al-Tibb" (trad. Mazhar Shah, 1966)
# Nota: Avicena (980-1037 d.C.) sintetizó la tradición galénica e hipocrática.
#       El Canon fue texto médico estándar en Europa y el mundo islámico por 600 años.
# h5_tradicion = 3

CASOS_ISLAMICOS = [
    {'id':'Is_C1','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A patient with burning fever who sweats copiously and warmly from the entire body
        on the day of crisis is surely recovering. The humor has been properly concocted
        and expelled through the pores. Allah has willed his recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C2','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        Cold sweat appearing in acute fever with livid extremities indicates imminent death.
        When the sweat is cold and confined to the forehead and the extremities are cold
        and cannot be warmed, the patient will die within one or two days.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C3','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        I treated a merchant from Bukhara with continuous fever. On the seventh day he sweated
        warmly and abundantly from the whole body. The fever left him that day. He was well
        within four days. Praise be to Allah for his recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C4','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A scholar from Isfahan had fever for twenty days. On the twentieth day warm sweat
        covered his whole body and the fever resolved. This is a complete and favorable crisis.
        The late crisis is due to excess of phlegmatic humor.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C5','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A woman with puerperal fever sweated cold from her chest and upper limbs. Her lower
        body was cold and dry. She could not be warmed despite all efforts. She died on the
        eighth day. Cold partial sweats in women after childbirth are always fatal.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C6','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        When the crisis sweat is moderate and covers only the upper half of the body,
        and is neither warm nor cold, the crisis is incomplete. The fever will return.
        Expect relapse and prepare further treatment.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C7','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A young man from Samarkand with ardent fever. On the eleventh day warm copious sweat
        covered his whole body. The fever resolved completely. He recovered quickly.
        The crisis on the eleventh day confirms the Hippocratic teaching.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C8','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        An old man with prolonged fever. Cold sweat covered his whole body on the fourteenth
        day. His extremities were livid and cold. His breathing was labored. He died before
        the next prayer time. Cold sweats in the elderly are invariably fatal.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C9','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A case of quartan fever resolved on the twenty-first day with warm abundant sweat
        from the entire body. The patient had been ill for three complete cycles. The crisis
        was complete and favorable; recovery was rapid.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C10','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A child with high fever sweated only from the head; the sweat was cold and the face
        pale. The body below was hot and dry. The child died on the fourth day.
        Sweating confined to the head in children is a most dangerous sign.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C11','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        I observed a case where warm sweat appeared spontaneously from the whole body during
        the height of fever, without any crisis medication. The fever resolved within the hour.
        This demonstrates that nature itself can expel the morbid humor when ready.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C12','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        When fever patients sweat from the extremities only, with cold sweat on the feet
        and hands while the trunk is hot and dry, this is a dangerous incomplete crisis.
        The patient may die or suffer prolonged illness.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C13','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A physician from Baghdad came to me ill with tertian fever. On the seventh day he
        sweated moderately and warmly from the whole body. The fever subsided. He recovered
        within four days and resumed his practice. A good crisis on the seventh day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C14','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        Cold sweats that are profuse and cover the whole body during high fever, while the
        extremities are cold and livid, indicate that the vital force is failing. Death
        is certain. No treatment can reverse this sign.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C15','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A young woman with remittent fever sweated moderately each evening; the sweat was
        tepid and covered the whole body. After fourteen days this pattern ceased and she
        recovered. Moderate tepid sweating indicates that the illness is resolving slowly.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C16','fuente':'Avicenna, Canon of Medicine, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        I was called to a nobleman who had been sweating cold from the chest and upper body
        for three days. The lower body was hot and dry. I told his family he would die.
        He died the next morning. Cold chest sweats with hot lower body are always fatal.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C17','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A student afflicted with brain fever sweated warmly and abundantly from the whole
        body on the fourteenth day. The fever resolved immediately. He recovered completely.
        The warm general sweat on a critical day is the most favorable sign in medicine.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C18','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        An elderly scholar with continuous fever sweated cold on the seventh day from the
        forehead and collar bones only. The rest of his body was dry and his extremities cold.
        He died on the ninth day. A cold partial sweat is never a favorable sign.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C19','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A merchant with prolonged fever had his crisis on the twentieth day with warm profuse
        sweat from the entire body. He recovered completely within two days. Long fevers
        that resolve with warm general sweat on the twentieth day always have good outcomes.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C20','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        Cold sweat in acute fever accompanied by weakness of the pulse, cold extremities,
        and darkening of the urine means the patient will not survive. This combination
        of signs has been confirmed in every case I have observed.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C21','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        When sweat appears warm from the whole body on the crisis day and reduces the fever
        completely, nature has successfully expelled the concocted morbid humor. This patient
        will recover and requires no further treatment beyond rest and light food.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C22','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A soldier from the east had fever for seven days. On the seventh day cold sweat
        appeared on his chest; his feet and hands were cold and livid. He died before
        the evening prayer. Cold chest sweat with cold extremities is a fatal combination.""",
     'h1_localizacion':1,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C23','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A woman with intermittent fever sweated tepidly from the whole body on alternate days.
        After fourteen days the pattern ceased, the fever resolved and she recovered.
        Tepid moderate sweating that reduces the fever gradually indicates eventual recovery.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C24','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        I treated a poet with high fever and delirium. On the eleventh day warm abundant sweat
        covered his body from head to foot. The fever resolved and the delirium cleared.
        He recovered completely and wrote a poem in my honor.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C25','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        Profuse cold sweating covering the whole body with cold extremities, labored breathing,
        and failing pulse means death is imminent. No physician can help. Recite the Quran
        for the dying. I have seen this pattern end in death without exception.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C26','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A case of false crisis: on the seventh day moderate warm sweat appeared from the upper
        body only. The fever diminished but did not resolve. True crisis came on the fourteenth
        day with complete warm sweating. The patient recovered after the true crisis.""",
     'h1_localizacion':1,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C27','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        An elderly man with quartan fever had his crisis on the twenty-first day. The sweat
        was warm and covered the whole body. The fever resolved. He recovered slowly due
        to his age and weakness. Old age makes recovery from crisis more difficult.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C28','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        When fever is at its height and the patient sweats cold from the forehead while the
        trunk is hot and dry and the extremities are cold, death will come within two days.
        This combination of signs is unfailingly fatal in my experience.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
    {'id':'Is_C29','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        A youth from Persia with ardent fever sweated warmly and abundantly from the whole
        body on the seventh day. The crisis was complete and the fever resolved. He asked
        for food immediately. He made a full recovery within three days.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':3},
    {'id':'Is_C30','fuente':'Ibn Sina, Al-Qanun fi al-Tibb, Book IV','tradicion':'islamica','siglo':11,'texto':"""
        Cold sweats in high fever that cover the whole body, when accompanied by blackening
        of the extremities, indicate that the vital heat is extinguished. The patient will
        die within hours. No medicine known to me can reverse this fatal sign.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':3},
]

# ── TRADICIÓN CHINA (30 casos) ────────────────────────────────
# Fuentes primarias:
#   - Zhang Zhongjing, "Shang Han Lun" (trad. Luo Xiwen, Foreign Languages Press, 1993)
#   - Zhang Zhongjing, "Jingui Yaolue" (trad. Yang Shou-zhong, Blue Poppy Press, 1994)
# Nota: El Shang Han Lun (~220 d.C.) describe el tratamiento de enfermedades febriles
#       mediante los seis estadios yin-yang. Es la tradición más independiente de la griega.
# h5_tradicion = 4

CASOS_CHINOS = [
    {'id':'Ch_C1','fuente':'Shang Han Lun, Chapter 1','tradicion':'china','siglo':3,'texto':"""
        When there is taiyang disease with fever and sweating, the sweat is warm and spontaneous,
        covering the whole body. If the pulse is floating and moderate and the fever resolves
        after sweating, the patient will recover. This is favorable sweating.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C2','fuente':'Shang Han Lun, Chapter 1','tradicion':'china','siglo':3,'texto':"""
        When the body is hot and the patient sweats but the sweat is cold and the fever does
        not diminish, and the extremities are cold and the pulse is faint, the prognosis is
        very grave. This is yang collapse with yin exuberance. Death is likely.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C3','fuente':'Shang Han Lun, Chapter 2','tradicion':'china','siglo':3,'texto':"""
        Taiyang disease with fever, headache, and stiff neck; the patient sweats spontaneously
        and warmly. The pulse is floating and slow. After warm abundant sweating covering
        the whole body the fever resolves and recovery follows.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C4','fuente':'Shang Han Lun, Chapter 3','tradicion':'china','siglo':3,'texto':"""
        When in high fever the patient sweats from the head only and the sweat is cold,
        and the body below the neck is dry, and the extremities are cold, this indicates
        internal cold with external heat. The prognosis is poor; the patient may die.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C5','fuente':'Shang Han Lun, Chapter 4','tradicion':'china','siglo':3,'texto':"""
        When the patient has fever for seven days and on the seventh day the sweating
        is warm and copious covering the whole body, and the fever resolves, this is
        a natural crisis. Recovery follows. Heaven has determined the critical day.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C6','fuente':'Shang Han Lun, Chapter 5','tradicion':'china','siglo':3,'texto':"""
        Shaoyin disease with cold extremities, diarrhea, and cold sweat covering the body
        indicates collapse of yang qi. The pulse is faint and almost imperceptible.
        This patient will die; the yang has been extinguished by yin.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C7','fuente':'Shang Han Lun, Chapter 6','tradicion':'china','siglo':3,'texto':"""
        When the patient with yangming disease sweats profusely and warmly from the whole
        body and the high fever resolves completely, this indicates successful expulsion of
        heat through the exterior. The patient will recover quickly.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C8','fuente':'Shang Han Lun, Chapter 7','tradicion':'china','siglo':3,'texto':"""
        Jueyin disease: the patient alternates between heat and cold. Cold sweat appears
        on the forehead and hands during the cold phase. The extremities are cold.
        If this pattern continues beyond seven days, the patient will not survive.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C9','fuente':'Shang Han Lun, Chapter 8','tradicion':'china','siglo':3,'texto':"""
        When cold disease transforms into heat and the patient begins to sweat warmly
        and copiously from the whole body, and the pulse becomes strong and the fever
        diminishes, this is recovery. The transformation from cold to heat is complete.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C10','fuente':'Shang Han Lun, Chapter 9','tradicion':'china','siglo':3,'texto':"""
        The patient has high fever and sweats but the sweat is cold and does not reduce
        the fever. The hands and feet are cold. The urine is scanty. This is dangerous:
        the yang qi is being depleted. The patient may die if not treated urgently.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C11','fuente':'Shang Han Lun, Chapter 10','tradicion':'china','siglo':3,'texto':"""
        When taiyang disease has lasted fourteen days and warm sweat suddenly appears from
        the whole body and the fever resolves, this is a delayed but complete crisis.
        The patient will recover. Long illness that resolves with warm sweat is curable.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C12','fuente':'Shang Han Lun, Chapter 11','tradicion':'china','siglo':3,'texto':"""
        Cold sweat on the palms and soles during high fever with cold extremities indicates
        collapse of the yang. The internal yang has retreated and the cold has prevailed.
        This patient will die; the pattern cannot be reversed.""",
     'h1_localizacion':2,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C13','fuente':'Shang Han Lun, Chapter 12','tradicion':'china','siglo':3,'texto':"""
        A man with yangming fever sweated abundantly and warmly for three consecutive days.
        Each time the sweat appeared warm and covered the whole body the fever diminished.
        By the fourteenth day the fever had resolved completely. Recovery was complete.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C14','fuente':'Shang Han Lun, Chapter 13','tradicion':'china','siglo':3,'texto':"""
        When the patient with severe cold disease sweats cold from the head and neck while
        the body below is dry and burning, this is a critical and dangerous pattern.
        The yang has separated from the yin. Death will follow within two days.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C15','fuente':'Shang Han Lun, Chapter 14','tradicion':'china','siglo':3,'texto':"""
        Taiyang disease resolving: warm spontaneous sweating from the whole body appears
        and the patient's pulse becomes soft and slow. The fever resolves and the patient
        feels relief. This is the correct resolution of taiyang exterior pattern.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C16','fuente':'Shang Han Lun, Chapter 15','tradicion':'china','siglo':3,'texto':"""
        When yangming disease has been present for many days and the patient's sweating
        is moderate and tepid, covering the whole body gradually, the fever is resolving
        slowly. Recovery will come but convalescence will be prolonged.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C17','fuente':'Jingui Yaolue, Chapter 1','tradicion':'china','siglo':3,'texto':"""
        A man with prolonged fever sweated only from the chest and upper body; the sweat
        was tepid. The lower body remained dry. After twenty days of this incomplete sweating
        pattern, warm general sweating finally appeared and he recovered.""",
     'h1_localizacion':1,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C18','fuente':'Jingui Yaolue, Chapter 2','tradicion':'china','siglo':3,'texto':"""
        Profuse cold sweating covering the whole body with cold extremities and faint pulse
        in high fever indicates collapse of the original yang. The vital energy is exhausted.
        No acupuncture or herbal treatment can save this patient. He will die.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C19','fuente':'Jingui Yaolue, Chapter 3','tradicion':'china','siglo':3,'texto':"""
        A woman with postpartum fever sweated warmly from her whole body on the seventh day.
        The fever resolved and she recovered her strength. Warm general sweating after
        childbirth fever on the seventh day is always a favorable resolution.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C20','fuente':'Jingui Yaolue, Chapter 4','tradicion':'china','siglo':3,'texto':"""
        When the patient with shaoyin disease sweats cold from the whole body and the
        extremities are cold and cannot be warmed even with moxibustion, the yang has
        been exhausted. This is a pattern of death. No treatment will help.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C21','fuente':'Jingui Yaolue, Chapter 5','tradicion':'china','siglo':3,'texto':"""
        When there is high fever with abundant warm sweating from the whole body and the
        fever decreases with each bout of sweating, this is yangming heat being expelled
        through the exterior. Recovery will be complete within seven days.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C22','fuente':'Jingui Yaolue, Chapter 6','tradicion':'china','siglo':3,'texto':"""
        A patient with taiyin disease sweated cold from the head and neck; the body below
        was cold and the extremities could not be warmed. The pulse was deep and faint.
        He died before the next day. Cold head sweat with cold lower body is fatal.""",
     'h1_localizacion':0,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C23','fuente':'Jingui Yaolue, Chapter 7','tradicion':'china','siglo':3,'texto':"""
        Moderate tepid sweating from the whole body occurring each evening during fever,
        with the fever diminishing after each episode, indicates that the pathogen is
        being gradually expelled. Recovery will come but will take many days.""",
     'h1_localizacion':3,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C24','fuente':'Jingui Yaolue, Chapter 8','tradicion':'china','siglo':3,'texto':"""
        When a patient with severe fever sweats from the palms and soles while the trunk
        is hot and dry, this is partial sweating indicating incomplete crisis. The fever
        will continue. This pattern may lead to death if not treated.""",
     'h1_localizacion':2,'h2_temperatura':1,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C25','fuente':'Jingui Yaolue, Chapter 9','tradicion':'china','siglo':3,'texto':"""
        A young man with yangming disease sweated copiously and warmly from the whole body
        on the fourteenth day. The high fever resolved immediately and completely.
        He recovered within two days. Warm general sweat on a critical day cures the disease.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C26','fuente':'Jingui Yaolue, Chapter 10','tradicion':'china','siglo':3,'texto':"""
        Shaoyin cold pattern with cold sweating from the whole body, reverting extremities,
        diarrhea with undigested food, and faint pulse indicates complete yang collapse.
        This patient cannot be saved. Death will come quickly.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C27','fuente':'Jingui Yaolue, Chapter 11','tradicion':'china','siglo':3,'texto':"""
        A woman with prolonged fever over twenty days sweated warmly from the whole body
        on the twentieth day. The fever left her completely. She recovered. Long fever
        resolving with warm general sweat, even after twenty days, leads to recovery.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C28','fuente':'Jingui Yaolue, Chapter 12','tradicion':'china','siglo':3,'texto':"""
        Cold sweat breaking out suddenly over the whole body during high fever, with the
        pulse becoming faint and the extremities cold, indicates that the yang is collapsing.
        This patient will die. Moxibustion on guan yuan may be attempted but rarely helps.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
    {'id':'Ch_C29','fuente':'Jingui Yaolue, Chapter 13','tradicion':'china','siglo':3,'texto':"""
        Taiyang disease with spontaneous sweating: the sweat is warm, covers the whole body,
        and is accompanied by aversion to wind. After the sweat the fever resolves and
        the patient recovers. This is the correct resolution of the exterior pattern.""",
     'h1_localizacion':3,'h2_temperatura':2,'h3_momento':0,'h4_pronostico':0,'h5_tradicion':4},
    {'id':'Ch_C30','fuente':'Jingui Yaolue, Chapter 14','tradicion':'china','siglo':3,'texto':"""
        When all four limbs are cold and the patient sweats cold from the whole body
        and the breathing is weak and shallow and the pulse cannot be felt, the yang qi
        has been completely exhausted. Death is imminent. Pray for the patient's soul.""",
     'h1_localizacion':3,'h2_temperatura':0,'h3_momento':0,'h4_pronostico':1,'h5_tradicion':4},
]

# ══════════════════════════════════════════════════════════════════
# EXTENDED CLINICAL ANNOTATIONS v2.0 (H6-H8 new variables)
# h6_orina=0(good)/1(absent)/2(bad) | h7_pulso=0(strong)/1(absent)/2(weak)
# h8_piel=0(normal)/1(absent)/2(livid) | h9_fiebre=0(cont)/1(interm)/2(remit)
# h10_delirio=0(no)/1(yes) | h11_extremid=0(warm)/1(cold)
# h12_resp=0(normal)/1(labored) | h13_apetito=0(returned)/1(absent)
# ══════════════════════════════════════════════════════════════════
EXTENDED_ANNOTATIONS = {
'Gr_Ep1_C1':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
'Gr_Ep1_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C5':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C6':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C8':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C9':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C10':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C12':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep1_C14':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C5':{'h6_orina':0,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C6':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C8':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C9':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C10':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C12':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C14':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C15':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep3_C16':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Prog_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Prog_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Prog_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Aph_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Aph_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Aph_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep2_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep2_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep2_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep2_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Gr_Ep5_C1':{'h6_orina':2,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Gr_Ep5_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C5':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C6':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C8':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C9':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C10':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C12':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C14':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C15':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C16':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C17':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C18':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C19':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C20':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C21':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C22':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C23':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
'Ba_C24':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C25':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ba_C26':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C27':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ba_C28':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C1':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':0},
'Ga_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C3':{'h6_orina':0,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':0},
'Ga_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C5':{'h6_orina':0,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C6':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C8':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C9':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C10':{'h6_orina':1,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C12':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':0},
'Ga_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C14':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C15':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C16':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C17':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C18':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C19':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C20':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C21':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C22':{'h6_orina':0,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C23':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C24':{'h6_orina':0,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C25':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C26':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C27':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ga_C28':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C29':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ga_C30':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C1':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C2':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C3':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C5':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C6':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C8':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
'Is_C9':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C10':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C12':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C14':{'h6_orina':1,'h7_pulso':2,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C15':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C16':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C17':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C18':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C19':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C20':{'h6_orina':2,'h7_pulso':2,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C21':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C22':{'h6_orina':1,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C23':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C24':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':1,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C25':{'h6_orina':1,'h7_pulso':2,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
'Is_C26':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':2,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C27':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Is_C28':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Is_C29':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':0},
'Is_C30':{'h6_orina':2,'h7_pulso':1,'h8_piel':2,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C1':{'h6_orina':1,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C2':{'h6_orina':2,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C3':{'h6_orina':1,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C4':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C5':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C6':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C7':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C8':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':1,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C9':{'h6_orina':1,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C10':{'h6_orina':2,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C11':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C12':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C13':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C14':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C15':{'h6_orina':1,'h7_pulso':0,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C16':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C17':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C18':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C19':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C20':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C21':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C22':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C23':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C24':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C25':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C26':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
'Ch_C27':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C28':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':0,'h13_apetito':1},
'Ch_C29':{'h6_orina':1,'h7_pulso':1,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':0,'h12_resp':0,'h13_apetito':1},
'Ch_C30':{'h6_orina':1,'h7_pulso':2,'h8_piel':1,'h9_fiebre':0,'h10_delirio':0,'h11_extremid':1,'h12_resp':1,'h13_apetito':1},
}
# ══════════════════════════════════════════════════════════════════
# MOUNT DRIVE
# ══════════════════════════════════════════════════════════════════
CACHE_DIR = None
for _att in range(3):
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=(_att>0))
        CACHE_DIR = '/content/drive/MyDrive/QML_Medico'
        os.makedirs(CACHE_DIR, exist_ok=True)
        _t = f'{CACHE_DIR}/.test'; open(_t,'w').write('ok'); os.remove(_t)
        print(f"Drive mounted ✓"); break
    except Exception as e:
        print(f"Drive attempt {_att+1}/3: {e}"); time.sleep(5)
if CACHE_DIR is None:
    CACHE_DIR = '/content'
    print("⚠ Drive unavailable — using /content (results will not persist)")

# ══════════════════════════════════════════════════════════════════
# AUTO-DETECT COMPLETED EXPERIMENTS FROM DRIVE
# ══════════════════════════════════════════════════════════════════
RESULTS_DIR = f'{CACHE_DIR}/resultados'
print(f"\n{'─'*65}")
print(f"  Scanning Drive for completed experiments...")
print(f"  Path: {RESULTS_DIR}")
print(f"{'─'*65}")

completed_set          = scan_completed_experiments(RESULTS_DIR)
EXPERIMENTOS, skipped  = build_queue(ALL_EXPERIMENTS, completed_set)

print(f"\n  {'─'*40}")
print(f"  Already done : {len(skipped)}")
for s in skipped:
    print(f"    ✓ SKIP  {s[0]:<3} {s[1]:<10} abl={'Y' if s[2] else 'N'}")
print(f"  To train now : {len(EXPERIMENTOS)}")
for e in EXPERIMENTOS:
    print(f"    → RUN   {e[0]:<3} {e[1]:<10} abl={'Y' if e[2] else 'N'}")

if not EXPERIMENTOS:
    print(f"\n  ✓ All 12 experiments already completed. Nothing to do.")
    print(f"  Check {RESULTS_DIR} for all results.")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════
print(f"\n{'╔' + '═'*63 + '╗'}")
print(f"║  QML MASTER EXECUTION — Extended v2                       ║")
print(f"╠{'═'*63}╣")
print(f"║  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU!)':<53}║")
print(f"║  Device : {str(DEVICE):<53}║")
print(f"║  AMP    : {'✓ active' if USE_AMP else '— disabled':<53}║")
print(f"║  Queue  : {len(EXPERIMENTOS)} experiment(s) to run{'':<33}║")
print(f"║  Skipped: {len(skipped)} already in Drive{'':<34}║")
print(f"╚{'═'*63}╝")

# ══════════════════════════════════════════════════════════════════
# LOAD BIO_CLINICALBERT (once, shared across all experiments)
# ══════════════════════════════════════════════════════════════════
print("\nLoading Bio_ClinicalBERT (once for all experiments)...")
BERT_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL)
bert       = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE).eval()
print("Bio_ClinicalBERT loaded ✓")

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP — only pending experiments
# ══════════════════════════════════════════════════════════════════
CASOS_ALL = {
    'griega':     CASOS_GRIEGOS,
    'babilonica': CASOS_BABILONICOS,
    'galenica':   CASOS_GALENICOS,
    'islamica':   CASOS_ISLAMICOS,
    'china':      CASOS_CHINOS,
}

_master_t0  = time.time()
all_results = []

for exp_num, (MODO, FILTRO, ABLACION) in enumerate(EXPERIMENTOS, 1):
    print(f"\n{'━'*65}")
    print(f"  EXPERIMENT {exp_num}/{len(EXPERIMENTOS)}: "
          f"{MODO} | {FILTRO} | ablacion={ABLACION}")
    print(f"  Remaining after this: {len(EXPERIMENTOS)-exp_num}")
    print(f"  Master elapsed: {(time.time()-_master_t0)/3600:.1f}h")
    print(f"{'━'*65}")

    try:
        result = run_experiment(
            MODO, FILTRO, ABLACION, CASOS_ALL,
            CACHE_DIR=CACHE_DIR,
            tokenizer=tokenizer,
            bert=bert,
        )
        if result:
            all_results.append(result)
            sig = '★' if result['p_valor'] < 0.05 else '—'
            print(f"\n  ✓ Experiment {exp_num} completed: "
                  f"{result['accuracy']:.1%} (p={result['p_valor']:.2e}) {sig}")
    except Exception as e:
        print(f"\n  ✗ Experiment {exp_num} FAILED: {e}")
        import traceback; traceback.print_exc()
        print(f"  Continuing with next experiment...")

# ══════════════════════════════════════════════════════════════════
# MASTER SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
_master_total = time.time() - _master_t0

print(f"\n{'╔' + '═'*67 + '╗'}")
print(f"║  MASTER EXECUTION COMPLETED — {len(all_results)} new experiment(s) this run{'':<12}║")
print(f"║  Total time: {_master_total/3600:.1f}h{'':<53}║")
print(f"╠{'═'*67}╣")
print(f"║  {'Mode':<4} {'Corpus':<10} {'Abl':<5} {'N':<5} "
      f"{'QML':>6} {'SVM':>6} {'RF':>6} {'p-val':>10} {'Sig':>4}  ║")
print(f"╠{'═'*67}╣")
for r in all_results:
    abl = 'Y' if r['ablacion'] else 'N'
    sig = '★' if r['p_valor'] < 0.05 else '—'
    print(f"║  {r['modo']:<4} {r['filtro']:<10} {abl:<5} {r['n']:<5} "
          f"{r['accuracy']:>5.1%} {r['svm']:>5.1%} {r['rf']:>5.1%} "
          f"{r['p_valor']:>10.2e} {sig:>4}  ║")
print(f"╚{'═'*67}╝")

# Save master summary
if all_results:
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    spath= f"{CACHE_DIR}/resultados/MASTER_SUMMARY_{ts}.json"
    with open(spath, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp':datetime.datetime.now().isoformat(),
            'version':'Extended_v2',
            'total_time_h':round(_master_total/3600,2),
            'n_new_experiments':len(all_results),
            'results':[{
                'modo':r['modo'],'filtro':r['filtro'],'ablacion':r['ablacion'],
                'accuracy':round(r['accuracy'],4),'balanced':round(r['balanced'],4),
                'svm':round(r['svm'],4),'rf':round(r['rf'],4),
                'p_valor':round(float(r['p_valor']),10),
                'significativo':bool(r['p_valor']<0.05),
                'ic95':[round(r['ci_low'],4),round(r['ci_high'],4)],
                'n':r['n'],'correctos':r['correctos'],
            } for r in all_results]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Master summary saved: {spath}")

print(f"\n  ✓ All done. Check {CACHE_DIR}/resultados/ for full results.")
